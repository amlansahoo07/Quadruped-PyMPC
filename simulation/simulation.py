# Description: This script is used to simulate the full model of the robot in mujoco
import pathlib

# Authors:
# Giulio Turrisi, Daniel Ordonez
import time
from os import PathLike
from pprint import pprint

import numpy as np

# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm

# Helper functions for plotting
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco

# PyMPC controller imports
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper

# NEW
import csv
from datetime import datetime

def run_simulation(
    qpympc_cfg,
    process=0,
    num_episodes=500,
    num_seconds_per_episode=60,
    ref_base_lin_vel=(0.0, 4.0),
    ref_base_ang_vel=(-0.4, 0.4),
    friction_coeff=(0.5, 1.0),
    base_vel_command_type="human",
    seed=0,
    render=True,
    recording_path: PathLike = None,
    # --- New optional plotting inputs ---
    plot_points: list[dict] | None = None,   # [{"name": "origin", "pos": (0,0,0), "color": [1,1,1,0.9], "diameter": 0.03}, ...]
    plot_axes: bool = False,                 # draw world XYZ axes at origin
    axes_length: float = 0.5,                # meters
    # --- NEW: timing options ---
    timing_start_name: str = "start",
    timing_end_name: str = "end",
    timing_radius: float = 0.05,             # meters, enter-sphere detection in XY
    timing_use_com: bool = False,            # if True use COM; else base_pos
    allow_multiple_timings: bool = True,     # record multiple passes per episode
    # --- NEW: timing log output ---
    timing_log_path: PathLike | None = None, # when None, will create recording_path/metrics/timing_<ts>.csv
):
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(seed)

    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    robot_leg_joints = qpympc_cfg.robot_leg_joints
    robot_feet_geom_names = qpympc_cfg.robot_feet_geom_names
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]

    # Save all observables available.
    state_obs_names = [] #list(QuadrupedEnv.ALL_OBS)  # + list(IMU.ALL_OBS)

    # Create the quadruped robot environment -----------------------------------------------------------
    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray(ref_base_lin_vel) * hip_height,  # pass a float for a fixed value
        ref_base_ang_vel=ref_base_ang_vel,  # pass a float for a fixed value
        ground_friction_coeff=friction_coeff,  # pass a float for a fixed value
        base_vel_command_type=base_vel_command_type,  # "forward", "random", "forward+rotate", "human"
        state_obs_names=tuple(state_obs_names),  # Desired quantities in the 'state' vec
    )
    pprint(env.get_hyperparameters())
    env.mjModel.opt.gravity[2] = -qpympc_cfg.gravity_constant

    # Some robots require a change in the zero joint-space configuration. If provided apply it
    if qpympc_cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))

    env.reset(random=False)
    if render:
        env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType

    # Initialization of variables used in the main control loop --------------------------------

    # Torque vector
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    # Torque limits
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )

    # Feet positions and Legs order
    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]

    # Create HeightMap -----------------------------------------------------------------------
    if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
        from gym_quadruped.sensors.heightmap import HeightMap

        resolution_vfa = 0.04
        dimension_vfa = 7
        heightmaps = LegsAttr(
            FL=HeightMap(
                n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData
            ),
            FR=HeightMap(
                n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData
            ),
            RL=HeightMap(
                n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData
            ),
            RR=HeightMap(
                n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData
            ),
        )
    else:
        heightmaps = None

    # Quadruped PyMPC controller initialization -------------------------------------------------------------
    # mpc_frequency = qpympc_cfg.simulation_params["mpc_frequency"]
    quadrupedpympc_observables_names = (
        "ref_base_height",
        "ref_base_angles",
        "ref_feet_pos",
        "nmpc_GRFs",
        "nmpc_footholds",
        "swing_time",
        "phase_signal",
        "lift_off_positions",
        # "base_lin_vel_err",
        # "base_ang_vel_err",
        # "base_poz_z_err",
    )

    quadrupedpympc_wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
        quadrupedpympc_observables_names=quadrupedpympc_observables_names,
    )

    # Data recording -------------------------------------------------------------------------------------------
    if recording_path is not None:
        from gym_quadruped.utils.data.h5py import H5Writer

        root_path = pathlib.Path(recording_path)
        root_path.mkdir(exist_ok=True)
        dataset_path = (
            root_path
            / f"{robot_name}/{scene_name}"
            / f"lin_vel={ref_base_lin_vel} ang_vel={ref_base_ang_vel} friction={friction_coeff}"
            / f"ep={num_episodes}_steps={int(num_seconds_per_episode // simulation_dt):d}.h5"
        )
        h5py_writer = H5Writer(
            file_path=dataset_path,
            env=env,
            extra_obs=None,  # TODO: Make this automatically configured. Not hardcoded
        )
        print(f"\n Recording data to: {dataset_path.absolute()}")
    else:
        h5py_writer = None

    # Keep geom_ids to update markers across frames
    point_marker_ids: dict[str, int] = {}   # name -> geom_id
    axes_geom_ids = {"x": -1, "y": -1, "z": -1}

    # Helper to normalize point specs
    def _normalize_point_spec(spec: dict) -> dict:
        name = spec.get("name", "pt")
        pos = spec.get("pos", (0.0, 0.0, 0.0))
        # Allow 2D tuples: (x,y) -> (x,y,0)
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0.0)
        color = spec.get("color", [1.0, 1.0, 1.0, 0.9])
        diameter = float(spec.get("diameter", 0.03))
        return {"name": name, "pos": np.array(pos, dtype=float), "color": np.array(color, dtype=float), "diameter": diameter}

    # Default example if user passed None
    if plot_points is None:
        plot_points = [
            {"name": "origin", "pos": (0.0, 0.0, 0.0), "color": [1, 1, 1, 0.1], "diameter": 0.03},
            # {"name": "start",  "pos": (0.3, -0.2, 0.0), "color": [0.2, 0.8, 0.2, 0.9], "diameter": 0.035},
            # {"name": "end",    "pos": (1.0,  0.4, 0.0), "color": [0.9, 0.2, 0.2, 0.9], "diameter": 0.035},
        ]

    # Pre-normalize specs
    plot_points = [_normalize_point_spec(p) for p in plot_points]

    # --- NEW: locate start/end points if present ---
    def _find_point_xy(points: list[dict], name: str) -> np.ndarray | None:
        for p in points:
            if p["name"].lower() == name.lower():
                pos = p["pos"]
                return np.array([pos[0], pos[1]], dtype=float)
        return None

    start_xy = _find_point_xy(plot_points, timing_start_name)
    end_xy   = _find_point_xy(plot_points, timing_end_name)

    # --- NEW: prepare timing CSV logger ---
    timing_writer = None
    timing_csv_fh = None
    pass_counter = 0
    current_pass_idx = None  # NEW: track the active attempt index
    if start_xy is not None and end_xy is not None:
        ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        if timing_log_path is None:
            # default: put under recording_path/metrics or ./metrics
            base_dir = pathlib.Path(recording_path) / "metrics" if recording_path is not None else pathlib.Path.cwd() / "metrics"
            base_dir.mkdir(parents=True, exist_ok=True)
            timing_log_path = base_dir / f"timing_{ts_str}.csv"
        else:
            timing_log_path = pathlib.Path(timing_log_path)
            timing_log_path.parent.mkdir(parents=True, exist_ok=True)
        timing_csv_fh = open(timing_log_path, mode="w", newline="")
        timing_writer = csv.writer(timing_csv_fh)
        timing_writer.writerow([
            "timestamp", "episode", "pass_idx",
            "start_time_s", "end_time_s", "travel_time_s",
            "start_x", "start_y", "end_x", "end_y",
            "use_com", "radius_m", "robot", "scene", "sim_dt"
        ])
        print(f"[timing] Logging to {timing_log_path}")

    # Storage for timings across episodes
    travel_times: list[float] = []

    # -----------------------------------------------------------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    N_EPISODES = num_episodes
    N_STEPS_PER_EPISODE = int(num_seconds_per_episode // simulation_dt)
    last_render_time = time.time()

    state_obs_history, ctrl_state_history = [], []
    for episode_num in range(N_EPISODES):
        ep_state_history, ep_ctrl_state_history, ep_time = [], [], []

        # --- NEW: reset timing FSM per episode ---
        fsm_state = "waiting_start"   # waiting_start -> timing -> cooldown
        start_time = None
        inside_start_prev = False
        inside_end_prev = False
        current_pass_idx = None  # NEW

        for _ in tqdm(range(N_STEPS_PER_EPISODE), desc=f"Ep:{episode_num:d}-steps:", total=N_STEPS_PER_EPISODE):
            # Update value from SE or Simulator ----------------------
            feet_pos = env.feet_pos(frame="world")
            feet_vel = env.feet_vel(frame='world')
            hip_pos = env.hip_positions(frame="world")
            base_lin_vel = env.base_lin_vel(frame="world")
            base_ang_vel = env.base_ang_vel(frame="base")
            base_ori_euler_xyz = env.base_ori_euler_xyz
            base_pos = env.base_pos
            com_pos = env.com

            # Get the reference base velocity in the world frame
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

            # Get the inertia matrix
            if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
                inertia = env.get_base_inertia().flatten()  # Reflected inertia of base at qpos, in world frame
            else:
                inertia = qpympc_cfg.inertia.flatten()

            # Get the qpos and qvel
            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            # Idx of the leg
            legs_qvel_idx = env.legs_qvel_idx  # leg_name: [idx1, idx2, idx3] ...
            legs_qpos_idx = env.legs_qpos_idx  # leg_name: [idx1, idx2, idx3] ...
            joints_pos = LegsAttr(FL=legs_qvel_idx.FL, FR=legs_qvel_idx.FR, RL=legs_qvel_idx.RL, RR=legs_qvel_idx.RR)

            # Get Centrifugal, Coriolis, Gravity, Friction for the swing controller
            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive

            # Compute feet jacobians
            feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
            feet_jac_dot = env.feet_jacobians_dot(frame='world', return_rot_jac=False)

            # Quadruped PyMPC controller --------------------------------------------------------------
            tau = quadrupedpympc_wrapper.compute_actions(
                com_pos,
                base_pos,
                base_lin_vel,
                base_ori_euler_xyz,
                base_ang_vel,
                feet_pos,
                hip_pos,
                joints_pos,
                heightmaps,
                legs_order,
                simulation_dt,
                ref_base_lin_vel,
                ref_base_ang_vel,
                env.step_num,
                qpos,
                qvel,
                feet_jac,
                feet_jac_dot,
                feet_vel,
                legs_qfrc_passive,
                legs_qfrc_bias,
                legs_mass_matrix,
                legs_qpos_idx,
                legs_qvel_idx,
                tau,
                inertia,
                env.mjData.contact,
            )
            
            # Limit tau between tau_limits
            for leg in ["FL", "FR", "RL", "RR"]:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            # Set control and mujoco step -------------------------------------------------------------------------
            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR

            # action_noise = np.random.normal(0, 2, size=env.mjModel.nu)
            # action += action_noise

            # Apply the action to the environment and evolve sim --------------------------------------------------
            state, reward, is_terminated, is_truncated, info = env.step(action=action)

            # Get Controller state observables
            ctrl_state = quadrupedpympc_wrapper.get_obs()

            # Store the history of observations and control -------------------------------------------------------
            base_poz_z_err = ctrl_state["ref_base_height"] - base_pos[2]
            ctrl_state["base_poz_z_err"] = base_poz_z_err

            ep_state_history.append(state)
            ep_time.append(env.simulation_time)
            ep_ctrl_state_history.append(ctrl_state)

            # --- NEW: timing detector (if both points available) ---
            if start_xy is not None and end_xy is not None:
                # Choose base center vs COM
                center_xy = (com_pos[:2] if timing_use_com else base_pos[:2]).astype(float)
                d_start = np.linalg.norm(center_xy - start_xy)
                d_end   = np.linalg.norm(center_xy - end_xy)
                inside_start = d_start <= timing_radius
                inside_end   = d_end   <= timing_radius

                # Rising-edge enter events
                entered_start = (not inside_start_prev) and inside_start
                entered_end   = (not inside_end_prev)   and inside_end

                if fsm_state == "waiting_start" and entered_start:
                    start_time = env.simulation_time
                    fsm_state = "timing"
                    print(f"[timing] Start passed at t={start_time:.3f}s (ep {episode_num})")

                elif fsm_state == "timing" and entered_end:
                    end_time = env.simulation_time
                    dt = float(end_time - start_time) if start_time is not None else float("nan")
                    travel_times.append(dt)
                    print(f"[timing] End passed at t={end_time:.3f}s (ep {episode_num}) — travel time {dt:.3f}s")
                    # --- NEW: append CSV row ---
                    if timing_writer is not None:
                        timing_writer.writerow([
                            datetime.now().isoformat(timespec="seconds"),
                            episode_num, pass_counter,
                            float(start_time) if start_time is not None else np.nan,
                            float(end_time),
                            dt,
                            float(start_xy[0]), float(start_xy[1]),
                            float(end_xy[0]), float(end_xy[1]),
                            bool(timing_use_com), float(timing_radius),
                            str(robot_name), str(scene_name), float(simulation_dt),
                        ])
                        pass_counter += 1
                    fsm_state = "cooldown"

                # Cooldown: wait until we exit both spheres, then optionally re-arm
                elif fsm_state == "cooldown":
                    if not inside_start and not inside_end:
                        fsm_state = "waiting_start" if allow_multiple_timings else "done"

                inside_start_prev = inside_start
                inside_end_prev = inside_end

            # Render only at a certain frequency -----------------------------------------------------------------
            if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

                # Plot the swing trajectory
                feet_traj_geom_ids = plot_swing_mujoco(
                    viewer=env.viewer,
                    swing_traj_controller=quadrupedpympc_wrapper.wb_interface.stc,
                    swing_period=quadrupedpympc_wrapper.wb_interface.stc.swing_period,
                    swing_time=LegsAttr(
                        FL=ctrl_state["swing_time"][0],
                        FR=ctrl_state["swing_time"][1],
                        RL=ctrl_state["swing_time"][2],
                        RR=ctrl_state["swing_time"][3],
                    ),
                    lift_off_positions=ctrl_state["lift_off_positions"],
                    nmpc_footholds=ctrl_state["nmpc_footholds"],
                    ref_feet_pos=ctrl_state["ref_feet_pos"],
                    early_stance_detector=quadrupedpympc_wrapper.wb_interface.esd,
                    geom_ids=feet_traj_geom_ids,
                )

                # Update and Plot the heightmap
                if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
                    # if(stc.check_apex_condition(current_contact, interval=0.01)):
                    for leg_id, leg_name in enumerate(legs_order):
                        data = heightmaps[
                            leg_name
                        ].data  # .update_height_map(ref_feet_pos[leg_name], yaw=env.base_ori_euler_xyz[2])
                        if data is not None:
                            for i in range(data.shape[0]):
                                for j in range(data.shape[1]):
                                    heightmaps[leg_name].geom_ids[i, j] = render_sphere(
                                        viewer=env.viewer,
                                        position=([data[i][j][0][0], data[i][j][0][1], data[i][j][0][2]]),
                                        diameter=0.01,
                                        color=[0, 1, 0, 0.5],
                                        geom_id=heightmaps[leg_name].geom_ids[i, j],
                                    )

                # Plot the GRF
                for leg_id, leg_name in enumerate(legs_order):
                    feet_GRF_geom_ids[leg_name] = render_vector(
                        env.viewer,
                        vector=feet_GRF[leg_name],
                        pos=feet_pos[leg_name],
                        scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                        color=np.array([0, 1, 0, 0.5]),
                        geom_id=feet_GRF_geom_ids[leg_name],
                    )

                # --- NEW: plot custom points as spheres ---
                for p in plot_points:
                    name, pos, color, diam = p["name"], p["pos"], p["color"], p["diameter"]
                    prev_id = point_marker_ids.get(name, -1)
                    geom_id = render_sphere(
                        viewer=env.viewer,
                        position=pos.tolist(),
                        diameter=diam,
                        color=color.tolist(),
                        geom_id=prev_id,
                    )
                    point_marker_ids[name] = geom_id

                # --- NEW: optional world axes at origin ---
                if plot_axes:
                    origin = np.array([0.0, 0.0, 0.0])
                    axes_geom_ids["x"] = render_vector(
                        env.viewer, vector=np.array([1.0, 0.0, 0.0]), pos=origin, scale=axes_length, color=np.array([1, 0, 0, 0.8]),
                        geom_id=axes_geom_ids["x"],
                    )
                    axes_geom_ids["y"] = render_vector(
                        env.viewer, vector=np.array([0.0, 1.0, 0.0]), pos=origin, scale=axes_length, color=np.array([0, 1, 0, 0.8]),
                        geom_id=axes_geom_ids["y"],
                    )
                    axes_geom_ids["z"] = render_vector(
                        env.viewer, vector=np.array([0.0, 0.0, 1.0]), pos=origin, scale=axes_length, color=np.array([0, 0, 1, 0.8]),
                        geom_id=axes_geom_ids["z"],
                    )

                env.render()
                last_render_time = time.time()

            # Reset the environment if the episode is terminated ------------------------------------------------
            if env.step_num >= N_STEPS_PER_EPISODE or is_terminated or is_truncated:
                if is_terminated:
                    print("Environment terminated")
                else:
                    state_obs_history.append(ep_state_history)
                    ctrl_state_history.append(ep_ctrl_state_history)     

                env.reset(random=True)
                quadrupedpympc_wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))
                # --- NEW: also reset episode timing FSM ---
                fsm_state = "waiting_start"
                start_time = None
                inside_start_prev = False
                inside_end_prev = False

        if h5py_writer is not None:  # Save episode trajectory data to disk.
            ep_obs_history = collate_obs(ep_state_history)  # | collate_obs(ep_ctrl_state_history)
            ep_traj_time = np.asarray(ep_time)[:, np.newaxis]
            h5py_writer.append_trajectory(state_obs_traj=ep_obs_history, time=ep_traj_time)
            pass

    # --- NEW: print summary at end ---
    if start_xy is not None and end_xy is not None and travel_times:
        arr = np.array(travel_times, dtype=float)
        print(f"[timing] Completed {len(arr)} start→end passes. "
              f"mean={arr.mean():.3f}s, min={arr.min():.3f}s, max={arr.max():.3f}s")
        
    # --- NEW: close timing CSV if opened ---
    if timing_csv_fh is not None:
        try:
            timing_csv_fh.flush()
        finally:
            timing_csv_fh.close()

    env.close()
    if h5py_writer is not None:
        return h5py_writer.file_path


def collate_obs(list_of_dicts) -> dict[str, np.ndarray]:
    """Collates a list of dictionaries containing observation names and numpy arrays
    into a single dictionary of stacked numpy arrays.
    """
    if not list_of_dicts:
        raise ValueError("Input list is empty.")

    # Get all keys (assumes all dicts have the same keys)
    keys = list_of_dicts[0].keys()

    # Stack the values per key
    collated = {key: np.stack([d[key] for d in list_of_dicts], axis=0) for key in keys}
    collated = {key: v[:, None] if v.ndim == 1 else v for key, v in collated.items()}
    return collated


if __name__ == "__main__":
    from quadruped_pympc import config as cfg

    qpympc_cfg = cfg
    # Custom changes to the config here:
    pass

    # Run the simulation with the desired configuration.....
    # run_simulation(qpympc_cfg=qpympc_cfg)

    run_simulation(
        qpympc_cfg=qpympc_cfg,
        render=True,
        plot_axes=True,
        axes_length=0.25,
        plot_points=[
            {"name": "origin", "pos": (0, 0, 0), "color": [0.1, 0.1, 0.1, 0.75], "diameter": 0.03},
            {"name": "start",  "pos": (0.5, 0.0, 0.0), "color": [0.9, 0, 1, 0.9], "diameter": 0.05},
            {"name": "end",    "pos": (3.0, 0.0, 0.5), "color": [0.9, 0.5, 0, 0.9], "diameter": 0.05},
        ]
    )

    # run_simulation(num_episodes=1, render=False)
