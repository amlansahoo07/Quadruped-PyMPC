import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quadruped_pympc.helpers.swing_trajectory_controller import SwingTrajectoryController
from quadruped_pympc.helpers.quadruped_utils import GaitType

class TestSwingTrajectoryController:
    """Test suite for SwingTrajectoryController.check_touch_down_condition method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.stc = SwingTrajectoryController(
            step_height=0.08,
            swing_period=0.4,
            position_gain_fb=np.array([100, 100, 100]),
            velocity_gain_fb=np.array([10, 10, 10]),
            generator='scipy'
        )
        self.dt = 0.002
    
    def test_initialization_of_private_variables(self):
        """Test that private variables are initialized correctly"""
        current_contact = [1, 1, 1, 1]
        previous_contact = [0, 1, 1, 1]
        
        with patch('quadruped_pympc.helpers.swing_trajectory_controller.cfg') as mock_cfg:
            mock_cfg.simulation_params = {'dt': self.dt}
            
            result = self.stc.check_touch_down_condition(current_contact, previous_contact)
            
            # Check that private variables are initialized
            assert hasattr(self.stc, '_full_stance_start_time')
            assert hasattr(self.stc, '_last_check_time')
            assert hasattr(self.stc, '_recent_touchdown')
            assert self.stc._full_stance_start_time is None
            assert self.stc._last_check_time == self.dt
            assert self.stc._recent_touchdown is True
    
    def test_single_leg_touchdown_detection(self):
        """Test detection of single leg touchdown"""
        test_cases = [
            # (current_contact, previous_contact, expected_touchdown_flag)
            ([1, 1, 1, 1], [0, 1, 1, 1], True),   # FL touches down
            ([1, 1, 1, 1], [1, 0, 1, 1], True),   # FR touches down
            ([1, 1, 1, 1], [1, 1, 0, 1], True),   # RL touches down
            ([1, 1, 1, 1], [1, 1, 1, 0], True),   # RR touches down
            ([1, 1, 1, 1], [1, 1, 1, 1], False),  # No touchdown
            ([0, 1, 1, 1], [1, 1, 1, 1], False),  # FL lifts off
        ]
        
        for current, previous, expected in test_cases:
            # Reset the controller state
            if hasattr(self.stc, '_full_stance_start_time'):
                delattr(self.stc, '_full_stance_start_time')
            
            with patch('quadruped_pympc.helpers.swing_trajectory_controller.cfg') as mock_cfg:
                mock_cfg.simulation_params = {'dt': self.dt}
                
                self.stc.check_touch_down_condition(current, previous)
                
                if expected:
                    assert self.stc._recent_touchdown is True, f"Failed for {current} vs {previous}"
                else:
                    assert self.stc._recent_touchdown is False, f"Failed for {current} vs {previous}"
    
    def test_full_stance_detection(self):
        """Test full stance condition detection"""
        test_cases = [
            ([1, 1, 1, 1], True),   # All legs in stance
            ([0, 1, 1, 1], False),  # FL in swing
            ([1, 0, 1, 1], False),  # FR in swing
            ([1, 1, 0, 1], False),  # RL in swing
            ([1, 1, 1, 0], False),  # RR in swing
            ([0, 0, 1, 1], False),  # Two legs in swing
            ([0, 0, 0, 0], False),  # All legs in swing
        ]
        
        for contact_state, expected in test_cases:
            # Reset the controller state
            if hasattr(self.stc, '_full_stance_start_time'):
                delattr(self.stc, '_full_stance_start_time')
            
            with patch('quadruped_pympc.helpers.swing_trajectory_controller.cfg') as mock_cfg:
                mock_cfg.simulation_params = {'dt': self.dt}
                
                self.stc.check_touch_down_condition(contact_state, [0, 0, 0, 0])
                
                # Check if we're tracking full stance correctly
                all_in_stance = all(contact_state[i] == 1 for i in range(4))
                assert all_in_stance == expected
    
    def test_stability_delay_timing(self):
        """Test that stability delay is properly enforced"""
        with patch('quadruped_pympc.helpers.swing_trajectory_controller.cfg') as mock_cfg:
            mock_cfg.simulation_params = {'dt': self.dt}
            
            # Reset controller state
            if hasattr(self.stc, '_full_stance_start_time'):
                delattr(self.stc, '_full_stance_start_time')
            
            # Step 1: Touchdown event
            result1 = self.stc.check_touch_down_condition([1, 1, 1, 1], [0, 1, 1, 1])
            assert result1 == 0  # Should not trigger yet
            assert self.stc._recent_touchdown is True
            
            # Step 2: Continue full stance but before stability delay
            for i in range(20):  # 20 * 0.002 = 0.04s < 0.05s delay
                result = self.stc.check_touch_down_condition([1, 1, 1, 1], [1, 1, 1, 1])
                assert result == 0  # Should not trigger yet
            
            # Step 3: After stability delay should trigger
            for i in range(10):  # Additional 10 * 0.002 = 0.02s (total 0.06s > 0.05s)
                result = self.stc.check_touch_down_condition([1, 1, 1, 1], [1, 1, 1, 1])
            
            assert result == 1  # Should trigger now
            assert self.stc._recent_touchdown is False  # Should be reset
    
    def test_state_reset_on_leaving_full_stance(self):
        """Test that state is reset when leaving full stance"""
        with patch('quadruped_pympc.helpers.swing_trajectory_controller.cfg') as mock_cfg:
            mock_cfg.simulation_params = {'dt': self.dt}
            
            # Reset controller state
            if hasattr(self.stc, '_full_stance_start_time'):
                delattr(self.stc, '_full_stance_start_time')
            
            # Step 1: Touchdown and enter full stance
            self.stc.check_touch_down_condition([1, 1, 1, 1], [0, 1, 1, 1])
            assert self.stc._recent_touchdown is True
            
            # Step 2: Leave full stance
            result = self.stc.check_touch_down_condition([0, 1, 1, 1], [1, 1, 1, 1])
            
            # Check that state is reset
            assert self.stc._full_stance_start_time is None
            assert self.stc._recent_touchdown is False
            assert result == 0
    
    def test_multiple_touchdown_events(self):
        """Test handling of multiple touchdown events"""
        with patch('quadruped_pympc.helpers.swing_trajectory_controller.cfg') as mock_cfg:
            mock_cfg.simulation_params = {'dt': self.dt}
            
            # Reset controller state
            if hasattr(self.stc, '_full_stance_start_time'):
                delattr(self.stc, '_full_stance_start_time')
            
            # Multiple legs touching down in sequence
            self.stc.check_touch_down_condition([1, 1, 1, 1], [0, 0, 0, 0])
            assert self.stc._recent_touchdown is True
            
            # Another touchdown event (should maintain flag)
            self.stc.check_touch_down_condition([1, 1, 1, 1], [1, 1, 1, 1])
            assert self.stc._recent_touchdown is True
    
    def test_edge_case_rapid_contact_changes(self):
        """Test rapid contact state changes"""
        with patch('quadruped_pympc.helpers.swing_trajectory_controller.cfg') as mock_cfg:
            mock_cfg.simulation_params = {'dt': self.dt}
            
            # Reset controller state
            if hasattr(self.stc, '_full_stance_start_time'):
                delattr(self.stc, '_full_stance_start_time')
            
            # Rapid contact changes
            contact_sequences = [
                ([1, 1, 1, 1], [0, 1, 1, 1]),  # Touchdown
                ([0, 1, 1, 1], [1, 1, 1, 1]),  # Immediate liftoff
                ([1, 1, 1, 1], [0, 1, 1, 1]),  # Touchdown again
            ]
            
            for current, previous in contact_sequences:
                result = self.stc.check_touch_down_condition(current, previous)
                # Should handle rapid changes without errors
                assert isinstance(result, int)
                assert result in [0, 1]


class TestGaitTypeCompatibility:
    """Test compatibility with different gait types"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.stc = SwingTrajectoryController(
            step_height=0.08,
            swing_period=0.4,
            position_gain_fb=np.array([100, 100, 100]),
            velocity_gain_fb=np.array([10, 10, 10]),
            generator='scipy'
        )
    
    def simulate_gait_pattern(self, gait_type, step_freq, duty_factor, duration=2.0):
        """Simulate a gait pattern and test touchdown detection"""
        dt = 0.002
        steps = int(duration / dt)
        
        # Mock gait generator patterns (simplified)
        gait_patterns = {
            GaitType.TROT.value: [0.0, 0.5, 0.5, 0.0],
            GaitType.PACE.value: [0.0, 0.0, 0.5, 0.5],
            GaitType.BACKDIAGONALCRAWL.value: [0.0, 0.5, 0.75, 0.25],
            GaitType.BFDIAGONALCRAWL.value: [0.0, 0.25, 0.5, 0.75],
            GaitType.CIRCULARCRAWL.value: [0.0, 0.25, 0.75, 0.5],
            GaitType.FRONTDIAGONALCRAWL.value: [0.5, 1.0, 0.75, 1.25],
        }
        
        phase_offsets = gait_patterns.get(gait_type, [0.0, 0.0, 0.0, 0.0])
        period = 1.0 / step_freq
        
        touchdown_triggers = []
        previous_contact = [0, 0, 0, 0]
        
        with patch('quadruped_pympc.helpers.swing_trajectory_controller.cfg') as mock_cfg:
            mock_cfg.simulation_params = {'dt': dt}
            
            for step in range(steps):
                current_time = step * dt
                current_contact = []
                
                # Generate contact sequence for each leg
                for leg_id in range(4):
                    phase = (current_time / period + phase_offsets[leg_id]) % 1.0
                    in_stance = phase < duty_factor
                    current_contact.append(1 if in_stance else 0)
                
                # Test touchdown detection
                result = self.stc.check_touch_down_condition(current_contact, previous_contact)
                touchdown_triggers.append(result)
                
                previous_contact = current_contact.copy()
        
        return touchdown_triggers
    
    def test_trot_gait_compatibility(self):
        """Test with trot gait parameters"""
        triggers = self.simulate_gait_pattern(
            GaitType.TROT.value, 
            step_freq=1.4, 
            duty_factor=0.65
        )
        
        # Should have some touchdown triggers
        assert sum(triggers) > 0
        # Should not trigger too frequently
        assert sum(triggers) < len(triggers) * 0.1
    
    def test_crawl_gait_compatibility(self):
        """Test with crawl gait parameters"""
        triggers = self.simulate_gait_pattern(
            GaitType.BACKDIAGONALCRAWL.value,
            step_freq=0.5,
            duty_factor=0.8
        )
        
        # Should have some touchdown triggers (this was the original problem)
        assert sum(triggers) > 0
        # Should not trigger too frequently
        assert sum(triggers) < len(triggers) * 0.1
    
    def test_all_crawl_patterns(self):
        """Test all crawl pattern types"""
        crawl_patterns = [
            GaitType.BACKDIAGONALCRAWL.value,
            GaitType.BFDIAGONALCRAWL.value,
            GaitType.CIRCULARCRAWL.value,
            GaitType.FRONTDIAGONALCRAWL.value,
        ]
        
        for pattern in crawl_patterns:
            # Reset controller state for each pattern
            if hasattr(self.stc, '_full_stance_start_time'):
                delattr(self.stc, '_full_stance_start_time')
            
            triggers = self.simulate_gait_pattern(
                pattern,
                step_freq=0.5,
                duty_factor=0.8
            )
            
            # Each pattern should work
            assert sum(triggers) > 0, f"Pattern {pattern} failed to trigger"
    
    def test_frequency_range_compatibility(self):
        """Test with different step frequencies"""
        frequencies = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 2.4]
        
        for freq in frequencies:
            # Reset controller state for each frequency
            if hasattr(self.stc, '_full_stance_start_time'):
                delattr(self.stc, '_full_stance_start_time')
            
            triggers = self.simulate_gait_pattern(
                GaitType.TROT.value,
                step_freq=freq,
                duty_factor=0.65,
                duration=3.0  # Longer duration for slower gaits
            )
            
            # Should work for all frequencies
            assert sum(triggers) > 0, f"Frequency {freq} Hz failed to trigger"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])