import pathlib
from datetime import datetime

import numpy as np

from quadruped_pympc import config as cfg
from simulation.simulation import run_simulation

"""
Run a series of push recovery trials while standing in place, comparing:
 - Baseline: fixed crawl gait (no adaptation)
 - Adaptive: phase pattern optimization enabled
"""
def run_simulation_mpc(optimize_crawl: bool, tag: str, seed: int = 0, outdir: str = "mpc_results", render_each: bool = False):

    # Toggle adaptive crawl
    cfg.mpc_params['optimize_crawl_patterns'] = optimize_crawl
    # Ensure crawl gait is active
    cfg.simulation_params['gait'] = 'crawl'
    # Scene
    cfg.simulation_params['scene'] = 'flat'
    # Enable logging
    cfg.mpc_params['mpc_logging'] = True
    # Output directory
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.mpc_params['mpc_log_dir'] = f'{outdir}/mpc_statistics/{tag}_{ts}'

    run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=12,
        ref_base_lin_vel=(0.0, 4.0),
        ref_base_ang_vel=(-0.4, 0.4),
        friction_coeff=(0.5, 1.0),
        base_vel_command_type="human",
        seed=seed,
        render=render_each,
    )

if __name__ == "__main__":
    # Baseline: fixed crawl (no adaptation)
    run_simulation_mpc(optimize_crawl=False, tag="baseline", seed=0, outdir="results", render_each=True)

    # Adaptive: phase pattern optimization enabled
    run_simulation_mpc(optimize_crawl=True, tag="adaptive", seed=0, outdir="results", render_each=True)
