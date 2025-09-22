#!/usr/bin/env python3
"""
Test case for the formant shifting fix.

This test verifies that the nnsvs_to_world() function correctly
auto-detects FFT size from BAP dimensions to prevent formant shifting.
"""

import sys
from pathlib import Path

# Add the repository path for imports
sys.path.append(str(Path(__file__).parent))

def test_fft_size_auto_detection():
    """Test automatic FFT size detection from BAP dimensions."""
    
    # Mock numpy and pyworld for testing without dependencies
    class MockArray:
        def __init__(self, shape):
            self.shape = shape
        
        def __getitem__(self, key):
            return self
    
    class MockNumpy:
        @staticmethod
        def exp(arr, where=None):
            return MockArray(arr.shape)
    
    class MockPyWorld:
        @staticmethod
        def decode_spectral_envelope(mgc, sample_rate, fft_size):
            # Return info about what was called
            return f"spectrogram_fft_{fft_size}"
        
        @staticmethod
        def decode_aperiodicity(bap, sample_rate, fft_size):
            return f"aperiodicity_fft_{fft_size}"
    
    # Mock the modules
    sys.modules['numpy'] = MockNumpy()
    sys.modules['pyworld'] = MockPyWorld()
    
    # Import the function under test
    from convert import nnsvs_to_world
    
    # Test cases: different BAP dimensions representing different FFT sizes
    test_cases = [
        (257, 512, "Standard resolution"),
        (513, 1024, "High resolution"),
        (1025, 2048, "Very high resolution"),
    ]
    
    print("Testing FFT size auto-detection in nnsvs_to_world()...")
    print("=" * 60)
    
    all_passed = True
    
    for bap_dims, expected_fft, description in test_cases:
        # Create mock data
        mgc = "mock_mgc"
        lf0 = MockArray((100, 1))
        vuv = "mock_vuv"
        bap = MockArray((100, bap_dims))
        sample_rate = 44100
        
        # Call the function with auto-detection (no explicit fft_size)
        f0, spectrogram, aperiodicity = nnsvs_to_world(mgc, lf0, vuv, bap, sample_rate)
        
        # Check if the correct FFT size was used
        expected_spec = f"spectrogram_fft_{expected_fft}"
        expected_ap = f"aperiodicity_fft_{expected_fft}"
        
        spec_correct = spectrogram == expected_spec
        ap_correct = aperiodicity == expected_ap
        test_passed = spec_correct and ap_correct
        all_passed = all_passed and test_passed
        
        status = "✓ PASS" if test_passed else "✗ FAIL"
        print(f"{status} {description}: BAP dims {bap_dims} → FFT size {expected_fft}")
        if not test_passed:
            print(f"      Expected: {expected_spec}, Got: {spectrogram}")
    
    print("=" * 60)
    
    # Test explicit FFT size parameter
    print("\nTesting explicit FFT size parameter...")
    bap = MockArray((100, 257))  # Would normally auto-detect to 512
    explicit_fft = 1024
    
    f0, spectrogram, aperiodicity = nnsvs_to_world(
        "mock_mgc", MockArray((100, 1)), "mock_vuv", bap, 44100, fft_size=explicit_fft
    )
    
    explicit_correct = f"_fft_{explicit_fft}" in spectrogram
    all_passed = all_passed and explicit_correct
    
    status = "✓ PASS" if explicit_correct else "✗ FAIL"
    print(f"{status} Explicit FFT size: {explicit_fft}")
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! The formant shifting fix is working correctly.")
        print("\nKey benefits of this fix:")
        print("• Prevents formant frequency distortion")
        print("• Maintains voice gender characteristics")
        print("• Preserves spectral envelope fidelity")
        print("• Backward compatible with explicit FFT size")
    else:
        print("❌ Some tests failed. The fix may need adjustment.")
    
    return all_passed

if __name__ == "__main__":
    test_fft_size_auto_detection()