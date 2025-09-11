import pathlib
from datetime import datetime

import numpy as np

from quadruped_pympc import config as cfg
from simulation.simulation_experiment import run_simulation

def run_case(optimize_crawl: bool, tag: str):
    cfg.mpc_params['optimize_crawl_patterns'] = optimize_crawl
    cfg.simulation_params['scene'] = 'flat'
    cfg.simulation_params['mode'] = 'forward'  # keep ref constant for fairness

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rec_dir = pathlib.Path(f"./recordings/push_flat_{tag}_{ts}")
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Two lateral pushes on the base
    pushes = [
        {"episode": 0, "start_s": 3.0, "duration_s": 0.20, "force_xyz": [0.0, 100.0, 0.0]},
        {"episode": 0, "start_s": 7.0, "duration_s": 0.20, "force_xyz": [0.0, -100.0, 0.0]},
    ]

    # Optional start/end points to log travel time
    pts = [
        {"name": "origin", "pos": (0.0, 0.0, 0.0), "color": [1, 1, 1, 0.1], "diameter": 0.03},
        {"name": "start",  "pos": (0.0, 0.0, 0.0), "color": [0.2, 0.8, 0.2, 0.9], "diameter": 0.035},
        {"name": "end",    "pos": (4.0, 0.0, 0.0), "color": [0.9, 0.2, 0.2, 0.9], "diameter": 0.035},
    ]

    run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=12,
        # Fixed forward speed: 0.875 m/s => pass 0.875 / hip_height (sim multiplies by hip_height internally)
        ref_base_lin_vel=(0.0, 4.0),
        ref_base_ang_vel=(-0.4, 0.4),
        friction_coeff=(0.5, 1.0),
        base_vel_command_type="human",
        seed=4,
        render=True,
        recording_path=rec_dir,                # H5 written here if enabled in your sim
        plot_points=pts,
        plot_axes=True,
        axes_length=0.25,
        timing_start_name="start",
        timing_end_name="end",
        timing_radius=0.05,
        timing_use_com=False,
        timing_log_path=rec_dir / "timing.csv",# success/travel time logged here
        push_schedule=pushes,                  # inject pushes
    )

if __name__ == "__main__":
    # Baseline: fixed crawl
    run_case(optimize_crawl=False, tag="baseline")
    # Adaptive: phase pattern optimization enabled
    run_case(optimize_crawl=True, tag="adaptive")