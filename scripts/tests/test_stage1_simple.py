#!/usr/bin/env python3
"""Simple test to check log output for status preservation"""

import subprocess
import re

# Run a command that will have both success and failure
cmd = [
    "python", "digimon_cli.py", 
    "-c", "Data/Russian_Troll_Sample",
    "-q", "please do social network and content and discourse analysis on the corpus",
    "--react"
]

print("Running DIGIMON with a complex query that will have failures...")
print("Command:", " ".join(cmd))
print("\nLooking for _status and _message preservation in logs...\n")

try:
    # Run the command and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120  # 2 minute timeout
    )
    
    # Search for preservation logs
    output = result.stdout + result.stderr
    
    # Look for preservation messages
    status_preservations = re.findall(r"Preserved _status='(\w+)' for step (\w+)", output)
    message_preservations = re.findall(r"Preserved _message='([^']+)' for step (\w+)", output)
    
    print(f"\nFound {len(status_preservations)} _status preservations:")
    for status, step in status_preservations[:5]:  # Show first 5
        print(f"  - Step {step}: _status='{status}'")
    
    print(f"\nFound {len(message_preservations)} _message preservations:")
    for message, step in message_preservations[:5]:  # Show first 5
        print(f"  - Step {step}: _message='{message[:50]}...'")
    
    # Check for failures
    failure_count = sum(1 for status, _ in status_preservations if status == 'failure')
    success_count = sum(1 for status, _ in status_preservations if status == 'success')
    
    print(f"\nStatus breakdown:")
    print(f"  - Success: {success_count}")
    print(f"  - Failure: {failure_count}")
    
    # Verify we have both successes and failures preserved
    if status_preservations and message_preservations:
        print("\n✓ STAGE 1 PASSED: Both _status and _message fields are being preserved!")
        if failure_count > 0:
            print("✓ Confirmed: Failure statuses are also preserved")
        else:
            print("⚠ Warning: No failures found to test failure preservation")
    else:
        print("\n✗ STAGE 1 FAILED: Status/message preservation not working")
        
except subprocess.TimeoutExpired:
    print("\n✗ Command timed out")
except Exception as e:
    print(f"\n✗ Error: {e}")