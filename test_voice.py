import requests
import sys

API = "http://localhost:8000"

# ── Test Voice ───────────────────────────────────────────────────
def test_voice(file_path):
    print(f"\n[VOICE] Testing: {file_path}")
    with open(file_path, "rb") as f:
        r = requests.post(f"{API}/analyze/voice",
                          files={"audio": ("recording.wav", f, "audio/wav")})
    import json
    print(json.dumps(r.json(), indent=2))

# ── Test Video ───────────────────────────────────────────────────
def test_video(file_path):
    print(f"\n[VIDEO] Testing: {file_path}")
    with open(file_path, "rb") as f:
        r = requests.post(f"{API}/analyze/video",
                          files={"video": (file_path.split("\\")[-1], f, "video/mp4")})
    import json
    print(json.dumps(r.json(), indent=2))

# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    
    # Locate actual WhatsApp voice call files in the workspace
    candidates = [
        r"frontend-legacy/assets/feature-cards/WhatsApp Audio 2026-04-02 at 00.56.22.wav",
        r"frontend-legacy/assets/feasibility/WhatsApp Audio 2026-04-02 at 00.27.28.wav",
        r"frontend-legacy/assets/feature-cards/WhatsApp Audio 2026-04-02 at 01.15.58.wav",
        r"Fraud_Detection_shield/temp.wav"
    ]
    
    selected_voice = None
    if len(sys.argv) > 1:
        selected_voice = sys.argv[1]
    else:
        for c in candidates:
            if os.path.exists(c):
                selected_voice = c
                break
                
    if selected_voice and os.path.exists(selected_voice):
        test_voice(selected_voice)
    else:
        print("Error: No test audio file found.")
        print("Please specify a path or place a test file.")
