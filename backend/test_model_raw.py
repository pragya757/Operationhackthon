import os
import numpy as np
import librosa
from transformers import pipeline

MODEL_NAME = "mo-thecreator/Deepfake-audio-detection"
classifier = pipeline("audio-classification", model=MODEL_NAME, device=-1)

test_files = [
    "../frontend-legacy/assets/feasibility/WhatsApp Audio 2026-04-02 at 00.27.28.wav",
    "../frontend-legacy/assets/feature-cards/WhatsApp Audio 2026-04-02 at 00.56.22.wav",
    "../frontend-legacy/assets/feature-cards/WhatsApp Audio 2026-04-02 at 01.15.58.wav",
    "../Fraud_Detection_shield/temp.wav"
]

print("Model:", MODEL_NAME)
print("Configured Labels:", classifier.model.config.id2label)
print("-" * 60)

for f in test_files:
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f))
    if os.path.exists(abs_path):
        print(f"\nProcessing File: {f}")
        try:
            y, sr = librosa.load(abs_path, sr=16000, mono=True)
            res = classifier({"array": y, "sampling_rate": 16000})
            print("Raw Result:", res)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"File not found: {f}")
