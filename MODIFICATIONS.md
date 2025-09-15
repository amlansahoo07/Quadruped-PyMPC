# Implementation Modifications for Batch MPC Crawl Adaptation

This document outlines the key modifications made to the Quadruped-PyMPC codebase to implement the Batch MPC-based crawl gait adaptation functionality.

## Configuration Changes

- Added `optimize_crawl_patterns` flag in `config.py` to toggle the crawl pattern optimization
- Implemented crawl pattern representation with phase signal patterns in `config.py`:
  ```python
  'phase_signal_patterns': [
      [0.021, 0.521, 0.771, 0.271],  # FL leads
      [0.521, 0.021, 0.271, 0.771],  # FR leads  
      [0.771, 0.271, 0.021, 0.521],  # RL leads
      [0.271, 0.771, 0.521, 0.021]   # RR leads
  ]
  ```
- Added MPC logging parameters to track decision metrics and performance

## Controller Modifications

### Batch MPC Implementation

- Extended the `Acados_NMPC_GaitAdaptive` class to handle crawl pattern optimization
- Added batched evaluation logic in `compute_batch_control_crawl()` method to evaluate multiple patterns
- Implemented pattern selection based on minimizing MPC cost across candidates
- Integrated with the AcadosOcpBatchSolver to leverage batch solving capabilities:
  ```python
  self.batch_solver = AcadosOcpBatchSolver(
      batch_ocp,
      self.batch,
      verbose=False,
      json_file=self.ocp.code_export_directory + "/centroidal_nmpc_batch" + ".json",
  )
  ```

### Interface Implementation

- Created `optimize_crawl_pattern()` in `SRBDBatchedControllerInterface` to:
  - Initialize multiple periodic gait generators with different phase signals
  - Evaluate each pattern's predicted performance
  - Select optimal pattern based on MPC cost
  - Return best pattern index and relative margin of improvement

### Gait Adaptation Logic

- Modified `QuadrupedPyMPCWrapper` to orchestrate on-the-fly phase signal updates
- Implemented event-triggered optimization at full-stance configurations:
  ```python
  # Store crawl optimization flag when detected (runs at simulation frequency)
  if cfg.mpc_params['optimize_crawl_patterns']:
      if optimize_swing == 1 and step_num > 1000:
          self.pending_crawl_optimization = True
          self.last_optimize_swing_detection_step = step_num
      
      # Reset flag if too much time has passed (prevent stale flags)
      if step_num - self.last_optimize_swing_detection_step > 10:  # ~50ms timeout
          self.pending_crawl_optimization = False
  ```
- Added full stance detection in `SwingTrajectoryController` with lookahead mechanisms:
  ```python
  # Rising edge detection (transition from at least one foot in swing to all feet in stance)
        if np.all(current_contact == 1) and not np.all(previous_contact == 1):
            self.rising_edge_detected = True

        # Wait until first "n lookahead" columns in contact sequence are all in stance contact
        stable_stance = np.all(contact_sequence[:, 0:lookahead] == 1)
        next_leg_lift = not np.all(contact_sequence[:, lookahead] == 1)

        if self.rising_edge_detected and stable_stance and next_leg_lift:
            self.rising_edge_detected = False
            return 1 # Signal to trigger optimization
        else:
            return 0
  ```
- Added seamless phase signal transition mechanism:
  ```python
  if best_cost != float('inf'): 
      # Define the phase signals
      pgg_phase_signals = cfg.mpc_params['phase_signal_patterns']
      
      optimal_phase_signal = pgg_phase_signals[best_pattern_idx]
      leg_names = ['FL', 'FR', 'RL', 'RR']
      leading_leg = leg_names[best_pattern_idx]
      
      # Store pre-transition state for logging
      old_phases = self.wb_interface.pgg.phase_signal.copy()
      old_contact = contact_sequence[:, 0].copy()
      
      # Apply the optimal phase signal (keep same gait type)
      self.wb_interface.pgg.set_phase_signal(np.array(optimal_phase_signal))
      
      # Log the transition
      new_phases = self.wb_interface.pgg.phase_signal
  ```

## Experiment Scripts

- Implemented `experiment1.py` for locomotion performance evaluation
- Created `experiment2.py` for lateral push disturbance testing:
  ```python
  # Two lateral pushes on the base (opposite directions)
  pushes = [
      {"episode": 0, "start_s": 5.0, "duration_s": 0.55, "force_xyz": [0.0, 55.0, 0.0]},
      {"episode": 0, "start_s": 8.0, "duration_s": 0.55, "force_xyz": [0.0, -55.0, 0.0]},
  ]
  ```
- Added data collection and analysis utilities for experimental validation

## Performance Analysis Tools

- Added `aggregate_exp2.py` script for disturbance experiment data aggregation
- Implemented visualization tools for contact sequence analysis
- Created metrics calculation for recovery time, success rate, and attitude stability

These modifications enable the quadruped to dynamically select optimal crawl patterns based on its current state, significantly improving stability and disturbance rejection capabilities with minimal computational overhead.
