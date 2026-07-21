import os
import numpy as np
import librosa
from transformers import pipeline

models = [
    "mo-thecreator/Deepfake-audio-detection",
    "garystafford/wav2vec2-deepfake-voice-detector",
    "MelodyMachine/Deepfake-audio-detection-V2"
]

test_file = "../frontend-legacy/assets/feasibility/WhatsApp Audio 2026-04-02 at 00.27.28.wav"
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), test_file))

if not os.path.exists(abs_path):
    print("Test file not found.")
    exit(1)

y, sr = librosa.load(abs_path, sr=16000, mono=True)
dummy_input = {"array": y, "sampling_rate": 16000}

for model_name in models:
    print(f"\n==================================================")
    print(f"Loading Model: {model_name}")
    try:
        classifier = pipeline("audio-classification", model=model_name, device=-1)
        print("Configured Labels (id2label):", classifier.model.config.id2label)
        
        # Test on dummy input
        res = classifier(dummy_input)
        print("Raw Prediction on Human Audio:", res)
    except Exception as e:
        print(f"Failed to load or test {model_name}: {e}")
