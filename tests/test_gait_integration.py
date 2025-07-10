import numpy as np
import sys
import os
from unittest.mock import patch

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quadruped_pympc.helpers.swing_trajectory_controller import SwingTrajectoryController
from quadruped_pympc.helpers.quadruped_utils import GaitType

class GaitIntegrationTest:
    """Integration test for gait patterns with touchdown detection"""
    
    def __init__(self):
        self.stc = SwingTrajectoryController(
            step_height=0.08,
            swing_period=0.4,
            position_gain_fb=np.array([100, 100, 100]),
            velocity_gain_fb=np.array([10, 10, 10]),
            generator='scipy'
        )
        self.dt = 0.002
        self.results = {}
    
    def test_gait_pattern(self, gait_name, gait_params, duration=5.0):
        """Test a specific gait pattern"""
        print(f"\nTesting {gait_name} gait...")
        
        # Reset controller state
        if hasattr(self.stc, '_full_stance_start_time'):
            delattr(self.stc, '_full_stance_start_time')
        
        # Mock the config
        with patch('quadruped_pympc.helpers.swing_trajectory_controller.cfg') as mock_cfg:
            mock_cfg.simulation_params = {'dt': self.dt}
            
            # Simplified gait simulation without requiring PeriodicGaitGenerator
            steps = int(duration / self.dt)
            touchdown_triggers = []
            previous_contact = [0, 0, 0, 0]
            
            for step in range(steps):
                current_time = step * self.dt
                
                # Simple mock contact sequence - you can replace with actual gait logic
                current_contact = [1, 1, 1, 1]  # Simplified for testing
                
                # Test touchdown detection
                touchdown_result = self.stc.check_touch_down_condition(current_contact, previous_contact)
                touchdown_triggers.append(touchdown_result)
                
                previous_contact = current_contact.copy()
            
            total_triggers = sum(touchdown_triggers)
            success = total_triggers > 0
            
            self.results[gait_name] = {
                'total_triggers': total_triggers,
                'success': success
            }
            
            print(f"  Total triggers: {total_triggers}")
            print(f"  Success: {success}")
            
            return touchdown_triggers
    
    def test_all_gaits(self):
        """Test basic gait functionality"""
        print("Testing basic gait functionality...")
        
        # Test with a simple gait
        gait_params = {
            'step_freq': 1.0,
            'duty_factor': 0.6,
            'type': 'trot'
        }
        
        self.test_gait_pattern('basic_test', gait_params)
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*50)
        print("TOUCHDOWN DETECTION TEST REPORT")
        print("="*50)
        
        successful_tests = sum(1 for result in self.results.values() if result.get('success', False))
        total_tests = len(self.results)
        
        for gait_name, result in self.results.items():
            status = "PASS" if result.get('success', False) else "FAIL"
            print(f"{gait_name:<20}: {status}")
        
        print(f"\nSummary: {successful_tests}/{total_tests} tests passed")
        
        return successful_tests == total_tests

def main():
    """Run all integration tests"""
    test = GaitIntegrationTest()
    
    # Run basic tests
    test.test_all_gaits()
    
    # Generate report
    success = test.generate_report()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)