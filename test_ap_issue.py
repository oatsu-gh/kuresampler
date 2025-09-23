#!/usr/bin/env python3
"""
Test script to reproduce the ap becoming zero issue with B flags and g flags.
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to sys.path for local imports
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

def create_test_world_features():
    """Create dummy WORLD features for testing."""
    # Create test data
    n_frames = 100
    n_fft = 1025
    
    f0 = np.random.uniform(100, 300, n_frames)  # F0 values
    sp = np.random.uniform(0.01, 1.0, (n_frames, n_fft))  # Spectrum 
    ap = np.random.uniform(0.01, 0.5, (n_frames, n_fft))  # Aperiodicity - should NOT be all zeros
    
    return f0, sp, ap

def simulate_ap_issue():
    """Simulate the ap becoming zero issue."""
    print("Testing ap issue reproduction...")
    
    # Create test WORLD features
    f0, sp, ap = create_test_world_features()
    
    print(f"Original ap shape: {ap.shape}")
    print(f"Original ap nonzero count: {np.count_nonzero(ap)}")
    print(f"Original ap range: [{np.min(ap):.6f}, {np.max(ap):.6f}]")
    
    # Simulate B flag effect (息成分の強さ)
    # B flag ranges from 0-100, default 50
    b_flag_values = [0, 25, 49, 50, 51, 75, 100]
    
    for b_value in b_flag_values:
        ap_modified = ap.copy()
        
        if b_value <= 49:
            # B0の時非周期性指標が全て0になるように乗算
            # This is likely the source of the bug!
            if b_value == 0:
                ap_modified = np.zeros_like(ap_modified)  # This would cause all zeros
            else:
                factor = b_value / 50.0  # Scale down
                ap_modified = ap_modified * factor
        else:
            # B51-100では1000Hz～5000Hz帯の非周期性指標が全て1になるように加算
            # This might also cause issues
            factor = (b_value - 50) / 50.0
            ap_modified = ap_modified + factor * 0.5  # Add some value
            ap_modified = np.clip(ap_modified, 0, 1)  # Clip to valid range
        
        nonzero_count = np.count_nonzero(ap_modified)
        print(f"B{b_value:2d} - ap nonzero count: {nonzero_count:4d}, range: [{np.min(ap_modified):.6f}, {np.max(ap_modified):.6f}]")
        
        if nonzero_count == 0:
            print(f"  >>> WARNING: B{b_value} caused all ap values to become zero!")

    # Simulate g flag effect (疑似ジェンダー値)
    # g flag ranges from -100 to 100, default 0
    g_flag_values = [-100, -50, 0, 50, 100]
    
    print("\nTesting g flag effects:")
    for g_value in g_flag_values:
        ap_modified = ap.copy()
        
        # g flag might modify spectrum and potentially affect ap
        # This is speculative based on gender modification
        if g_value != 0:
            # Gender modification might affect aperiodicity
            factor = 1.0 + (g_value / 100.0) * 0.2  # Small modification
            ap_modified = ap_modified * factor
            ap_modified = np.clip(ap_modified, 0, 1)
        
        nonzero_count = np.count_nonzero(ap_modified)
        print(f"g{g_value:3d} - ap nonzero count: {nonzero_count:4d}, range: [{np.min(ap_modified):.6f}, {np.max(ap_modified):.6f}]")

if __name__ == '__main__':
    simulate_ap_issue()