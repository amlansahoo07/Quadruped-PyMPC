import pathlib
from datetime import datetime

import numpy as np

from quadruped_pympc import config as cfg
from simulation.simulation_experiment import run_simulation

"""
Run a series of push recovery trials while standing in place, comparing:
 - Baseline: fixed crawl gait (no adaptation)
 - Adaptive: phase pattern optimization enabled
"""
def run_experiment(optimize_crawl: bool, push_schedule: list, tag: str, seed: int = 42, render_flag: bool = False):
    # Toggle adaptive crawl
    cfg.mpc_params['optimize_crawl_patterns'] = optimize_crawl
    # Ensure crawl gait is active
    cfg.simulation_params['gait'] = 'crawl'
    # Scene
    cfg.simulation_params['scene'] = 'flat'

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rec_dir = pathlib.Path(f"./recordings/push_stand_{tag}_{ts}")
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Shared outcomes CSV for all trials in this condition
    outcomes_csv = rec_dir / "outcomes.csv"


    # Single trial with specified seed
    print(f"\n[{tag} (seed={seed})")
    trial_dir = rec_dir / f"trial_single"
    trial_dir.mkdir(parents=True, exist_ok=True)

    run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=12,
        # Stand-in-place: zero commanded velocity and yaw rate
        ref_base_lin_vel=0.0,
        ref_base_ang_vel=0.0,
        friction_coeff=(0.5, 1.0),
        base_vel_command_type="human",     # fixed command, no randomization
        seed=seed,
        render=render_flag,
        recording_path=trial_dir,          # H5 saved per trial if enabled in sim
        # No start/end timing needed for this scenario
        timing_log_path=None,
        plot_points=None,
        plot_axes=True,
        axes_length=0.25,
        # Push schedule
        push_schedule=push_schedule,
        # NEW: write per-episode outcome
        episode_outcomes_path=outcomes_csv,
    )

    # Summarize success rate
    successes = 0
    total = 0
    if outcomes_csv.exists():
        import csv
        with open(outcomes_csv, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                total += 1
                successes += int(row.get("success", "0"))
    print(f"\n[{tag}] Success rate: {successes}/{total} episodes")

    return rec_dir, successes, total

if __name__ == "__main__":

    # Two lateral pushes on the base (opposite directions)
    pushes = [
        {"episode": 0, "start_s": 5.0, "duration_s": 0.55, "force_xyz": [0.0, 55.0, 0.0]},
        {"episode": 0, "start_s": 8.0, "duration_s": 0.55, "force_xyz": [0.0, -55.0, 0.0]},
    ]

    # Baseline: fixed crawl (no adaptation)
    base_dir, base_ok, base_tot = run_experiment(optimize_crawl=False, push_schedule=pushes, tag="baseline", seed=0, render_flag=True)
    
    # Adaptive: phase pattern optimization enabled
    # adap_dir, adap_ok, adap_tot = run_experiment(optimize_crawl=True, push_schedule=pushes, tag="adaptive", seed=0, render_flag=True)

    print("\n=== Stand-under-push summary ===")
    print(f"Baseline (fixed crawl):  {base_ok}/{base_tot} successes -> {base_dir}")
    # print(f"Adaptive (phase opt):    {adap_ok}/{adap_tot} successes -> {adap_dir}")