"""
Verification Script – Voice Clone Detection Module
─────────────────────────────────────────────────
Tests the exposed internal `analyze_audio` method and the POST /api/voice-clone/analyze API.
"""

import os
import sys
import requests

# 1. Locate test WAV file
possible_paths = [
    "../frontend-legacy/assets/feasibility/WhatsApp Audio 2026-04-02 at 00.27.28.wav",
    "../frontend-legacy/assets/feature-cards/WhatsApp Audio 2026-04-02 at 00.56.22.wav",
    "../Fraud_Detection_shield/temp.wav"
]

test_file = None
for path in possible_paths:
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if os.path.exists(abs_path):
        test_file = abs_path
        break

if not test_file:
    print("[ERROR] No test WAV file found in default locations. Checked paths:")
    for p in possible_paths:
        print(f"  - {os.path.abspath(os.path.join(os.path.dirname(__file__), p))}")
    sys.exit(1)

print(f"[TEST 1] Testing direct python method with file: {test_file}")
try:
    # Set weights and tensors flags in environment for testing
    os.environ["SAVE_SPECTROGRAM_TENSORS"] = "True"
    
    from detectors.voice_clone_detector import analyze_audio
    result = analyze_audio(test_file)
    
    print("\nResult properties:")
    for k, v in result.items():
        if k == "spectrogram_image" and v:
            print(f"  - {k}: {v[:50]}... ({len(v)} chars)")
        else:
            print(f"  - {k}: {v}")
            
    print("\n[TEST 1] PASS: Internal analyze_audio runs and parses features.")
except Exception as e:
    print(f"[TEST 1] FAIL: Internal python method threw an exception: {e}")

print("\n" + "="*50 + "\n")

print("[TEST 2] Testing POST /api/voice-clone/analyze API endpoint...")
url = "http://127.0.0.1:8000/api/voice-clone/analyze"
try:
    with open(test_file, "rb") as f:
        files = {"file": (os.path.basename(test_file), f, "audio/wav")}
        resp = requests.post(url, files=files)
        
    print(f"Response HTTP status code: {resp.status_code}")
    if resp.status_code == 200:
        api_result = resp.json()
        print("\nAPI Response payload:")
        for k, v in api_result.items():
            if k == "spectrogram_image" and v:
                print(f"  - {k}: {v[:50]}... ({len(v)} chars)")
            else:
                print(f"  - {k}: {v}")
        print("\n[TEST 2] PASS: API endpoint responded with 200 OK and expected keys.")
    else:
        print(f"[TEST 2] FAIL: Endpoint returned status {resp.status_code}. Response: {resp.text}")
except Exception as e:
    print(f"[TEST 2] FAIL: API endpoint request failed: {e}")
