import numpy as np

from quadruped_pympc import config as cfg
from quadruped_pympc.helpers.periodic_gait_generator import PeriodicGaitGenerator



class SRBDBatchedControllerInterface:
    """This is an interface for a batched controller that uses the SRBD method to optimize the gait"""

    def __init__(self):
        """Constructor for the SRBD batched controller interface"""

        self.type = cfg.mpc_params['type']
        self.mpc_dt = cfg.mpc_params['dt']
        self.horizon = cfg.mpc_params['horizon']
        self.optimize_step_freq = cfg.mpc_params['optimize_step_freq']
        self.step_freq_available = cfg.mpc_params['step_freq_available']

        # crawl pattern optimization parameters
        self.optimize_crawl_patterns = cfg.mpc_params['optimize_crawl_patterns']
        self.crawl_patterns_available = cfg.mpc_params['crawl_patterns_available']

        from quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_gait_adaptive import Acados_NMPC_GaitAdaptive

        self.batched_controller = Acados_NMPC_GaitAdaptive()

        # in the case of nonuniform discretization, we create a list of dts and horizons for each nonuniform discretization
        if cfg.mpc_params['use_nonuniform_discretization']:
            self.contact_sequence_dts = [cfg.mpc_params['dt_fine_grained'], self.mpc_dt]
            self.contact_sequence_lenghts = [cfg.mpc_params['horizon_fine_grained'], self.horizon]
        else:
            self.contact_sequence_dts = [self.mpc_dt]
            self.contact_sequence_lenghts = [self.horizon]

    def optimize_gait(
        self,
        state_current: dict,
        ref_state: dict,
        inertia: np.ndarray,
        pgg_phase_signal: np.ndarray,
        pgg_step_freq: float,
        pgg_duty_factor: float,
        pgg_gait_type: int,
        optimize_swing: int,
    ) -> float:
        """Optimize the gait using the SRBD method
        TODO: remove the unused arguments, and not pass pgg but rather its values

        Args:
            state_current (dict): The current state of the robot
            ref_state (dict): The reference state of the robot
            inertia (np.ndarray): The inertia of the robot
            pgg_phase_signal (np.ndarray): The periodic gait generator phase signal of the legs (from 0 to 1)
            pgg_step_freq (float): The step frequency of the periodic gait generator
            pgg_duty_factor (float): The duty factor of the periodic gait generator
            pgg_gait_type (int): The gait type of the periodic gait generator
            contact_sequence_dts (np.ndarray): The contact sequence dts
            contact_sequence_lenghts (np.ndarray): The contact sequence lengths
            optimize_swing (int): The flag to optimize the swing

        Returns:
            float: The best sample frequency
        """

        best_sample_freq = pgg_step_freq
        # print("optimize_swing:", optimize_swing)
        if self.optimize_step_freq and optimize_swing == 1:
            print("Optimizing step frequency...")
            contact_sequence_temp = np.zeros((len(self.step_freq_available), 4, self.horizon))
            for j in range(len(self.step_freq_available)):
                pgg_temp = PeriodicGaitGenerator(
                    duty_factor=pgg_duty_factor,
                    step_freq=self.step_freq_available[j],
                    gait_type=pgg_gait_type,
                    horizon=self.horizon,
                )
                pgg_temp.set_phase_signal(pgg_phase_signal)
                contact_sequence_temp[j] = pgg_temp.compute_contact_sequence(
                    contact_sequence_dts=self.contact_sequence_dts,
                    contact_sequence_lenghts=self.contact_sequence_lenghts,
                )
            costs, best_sample_freq = self.batched_controller.compute_batch_control(
                state_current, ref_state, contact_sequence_temp
            )

        return best_sample_freq
    
    def optimize_crawl_pattern(
        self,
        state_current: dict,
        ref_state: dict,
        inertia: np.ndarray,
        pgg_phase_signal: np.ndarray,
        pgg_step_freq: float,
        pgg_duty_factor: float,
        optimize_swing: int,
    ) -> tuple[int, float]:
        """
        Optimize crawl pattern by testing different crawl gait types in parallel.
        
        Returns:
            best_crawl_pattern_idx: Index of the best crawl pattern
            best_cost: Cost of the best pattern
        """
        
        if not self.optimize_crawl_patterns:
            print("Crawl pattern optimization is disabled or not applicable.")
            # Return default pattern (BACKDIAGONALCRAWL)
            return 0, float('inf')
        
        # Generate contact sequences for all crawl patterns
        num_patterns = len(self.crawl_patterns_available)
        contact_sequences_batch = np.zeros((num_patterns, 4, self.horizon))
        
        for j, crawl_gait_type in enumerate(self.crawl_patterns_available):
            pgg_temp = PeriodicGaitGenerator(
                duty_factor=pgg_duty_factor,
                step_freq=pgg_step_freq,
                gait_type=crawl_gait_type,  # Different crawl pattern
                horizon=self.horizon,
            )
            pgg_temp.set_phase_signal(pgg_phase_signal)
            contact_sequences_batch[j] = pgg_temp.compute_contact_sequence(
                contact_sequence_dts=self.contact_sequence_dts,
                contact_sequence_lenghts=self.contact_sequence_lenghts,
            )
        
        # # Use existing batch controller to evaluate all patterns
        # costs, _ = self.batched_controller.compute_batch_control(
        #     state_current, ref_state, contact_sequence_temp
        # )
        
        # # Find best pattern
        # best_pattern_idx = np.argmin(costs)
        # best_cost = costs[best_pattern_idx]

        # Use the new crawl pattern batch controller
        costs, best_pattern_idx = self.batched_controller.compute_batch_control_crawl_patterns(
            state_current, ref_state, contact_sequences_batch
        )
        
        best_cost = costs[best_pattern_idx]
        
        # print(f"Selected crawl pattern {self.crawl_patterns_available[best_pattern_idx]} "
        #     f"(index {best_pattern_idx}) with cost {best_cost:.3f}")
        
        return best_pattern_idx, best_cost
