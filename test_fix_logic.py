#!/usr/bin/env python3
"""
Focused test for the ap zero issue - tests the logic of B flag detection.
"""

import re

def test_b_flag_detection():
    """Test the B flag detection logic used in the fix."""
    
    test_cases = [
        # (flag_string, expected_b_zero, description)
        ('', False, 'Empty flags'),
        ('B0', True, 'B0 only'),
        ('B0,g50', True, 'B0 with g flag'),
        ('g50,B0', True, 'g flag with B0'),
        ('B0g50', True, 'B0 concatenated with g'),
        ('B10', False, 'B10 (not B0)'),
        ('B50', False, 'B50 (default)'),
        ('g50', False, 'g flag only'),
        ('B1', False, 'B1 (not B0)'),
        ('B01', False, 'B01 (should be treated as B0? - edge case)'),
        ('eB0', True, 'e flag before B0'),
        ('B0e', True, 'B0 before e flag'),
    ]
    
    def detect_b_zero_flag(flag_str: str) -> bool:
        """Detection logic from the fix - using regex for accurate B0 detection."""
        import re
        flag_str = str(flag_str)
        b_zero_pattern = r'(?:^|[^0-9])B0(?:[^0-9]|$)'
        return re.search(b_zero_pattern, flag_str) is not None
    
    print("Testing B flag detection logic:")
    print("Flag String          | Expected | Detected | Result")
    print("-" * 55)
    
    all_passed = True
    for flag_string, expected, description in test_cases:
        detected = detect_b_zero_flag(flag_string)
        passed = detected == expected
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{flag_string:<18} | {str(expected):<8} | {str(detected):<8} | {status}")
    
    print("-" * 55)
    print(f"Overall result: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    
    return all_passed

def test_ap_restoration_conditions():
    """Test the conditions under which ap should be restored."""
    
    print("\nTesting ap restoration conditions:")
    
    scenarios = [
        # (ap_is_zero, ap_was_nonzero_before, is_b_zero_intended, should_restore, description)
        (True, True, False, True, 'Normal case: ap became zero unexpectedly'),
        (True, True, True, False, 'B0 case: ap=0 is intended'),
        (True, False, False, False, 'ap was already zero before'),
        (False, True, False, False, 'ap is not zero'),
        (True, True, None, True, 'Cannot determine B flag, restore to be safe'),
    ]
    
    print("AP=0 | Was≠0 | B0? | Should Restore | Description")
    print("-" * 60)
    
    for ap_zero, was_nonzero, is_b_zero, should_restore, desc in scenarios:
        # Logic from the fix
        should_restore_actual = (
            ap_zero and was_nonzero and not is_b_zero
        ) if is_b_zero is not None else (
            ap_zero and was_nonzero  # Default to restore if can't determine B flag
        )
        
        passed = should_restore_actual == should_restore
        status = "✓" if passed else "✗"
        
        print(f"{str(ap_zero):<4} | {str(was_nonzero):<5} | {str(is_b_zero):<3} | {str(should_restore_actual):<14} | {desc} {status}")

if __name__ == '__main__':
    print("=== Testing ap zero issue fix logic ===")
    
    flag_test_passed = test_b_flag_detection()
    test_ap_restoration_conditions()
    
    print(f"\n=== Summary ===")
    print(f"B flag detection: {'PASSED' if flag_test_passed else 'FAILED'}")
    print("\nThis test validates the logic used in the fix for detecting B0 flags")
    print("and determining when to restore ap values.")