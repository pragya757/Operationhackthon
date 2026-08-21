"""
evaluate_model.py
==================
Quick evaluation of the trained SpectrogramCNN on both:
  - The WhatsApp-converted dataset (data/whatsapp_dataset/)
  - The original raw dataset (Fake/Fake/ and Real/Real/)

Prints per-file predictions, accuracy, F1, and confusion matrix.

Usage:
  python scripts/evaluate_model.py
"""

import os
import sys
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

import torch
import torch.nn.functional as F
import torchaudio.transforms as T

try:
    import librosa
except ImportError:
    os.system("pip install librosa")
    import librosa

try:
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
except ImportError:
    os.system("pip install scikit-learn")
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ── Architecture (must match training) ────────────────────────────────────────
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.20):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )
    def forward(self, x): return self.block(x)

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(1,   32, 0.20)
        self.block2 = ConvBlock(32,  64, 0.20)
        self.block3 = ConvBlock(64,  128, 0.25)
        self.block4 = ConvBlock(128, 256, 0.30)
        self.gap    = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BACKEND_DIR, "models", "spectrogram_cnn_whatsapp.pth")
SAMPLE_RATE = 16_000
MAX_SAMPLES = 160_000
N_FFT       = 1024
HOP_LENGTH  = 512
N_MELS      = 128

_mel_t = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                            hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
_db_t  = T.AmplitudeToDB(stype="power", top_db=80)


def wav_to_spec(path: str) -> torch.Tensor:
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=10.0)
    n = len(y)
    if n < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - n))
    waveform = torch.from_numpy(y).float().unsqueeze(0)
    mel      = _mel_t(waveform)
    logmel   = _db_t(mel)
    mu, std  = logmel.mean(), logmel.std() + 1e-6
    return ((logmel - mu) / std).unsqueeze(0)   # (1, 1, N_MELS, T)


def load_model(device: str):
    model = SpectrogramCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run: python scripts/train_whatsapp_cnn.py first!")
        sys.exit(1)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_directory(model, device, directory: str, true_label: int, label_name: str):
    files = sorted([f for f in os.listdir(directory) if f.endswith('.wav')])
    preds_all   = []
    probs_all   = []
    labels_all  = []

    print(f"\n--- {label_name.upper()} ({len(files)} files) ---")
    print(f"  {'File':<35} {'Pred':>10}  {'P(Human)':>9}  {'P(AI)':>9}  {'✓/✗':>4}")
    print(f"  {'-'*70}")

    for fname in files:
        path = os.path.join(directory, fname)
        try:
            spec = wav_to_spec(path).to(device)
            with torch.no_grad():
                logits = model(spec)
                probs  = F.softmax(logits, dim=1)[0]
            p_human = float(probs[0])
            p_ai    = float(probs[1])
            pred    = 1 if p_ai >= 0.5 else 0
        except Exception as e:
            print(f"  {fname:<35} ERROR: {e}")
            continue

        pred_name = "AI/Fake" if pred == 1 else "Human"
        correct   = "✓" if pred == true_label else "✗"
        preds_all.append(pred)
        probs_all.append(p_ai)
        labels_all.append(true_label)
        print(f"  {fname:<35} {pred_name:>10}  {p_human:>8.1%}  {p_ai:>8.1%}  {correct:>4}")

    return preds_all, labels_all


def run_evaluation(name: str, fake_dir: str, real_dir: str, model, device):
    print(f"\n{'='*70}")
    print(f"  EVALUATION: {name}")
    print(f"{'='*70}")

    if not os.path.exists(fake_dir) or not os.path.exists(real_dir):
        print(f"  [SKIP] Directories not found")
        return

    fake_preds, fake_labels = evaluate_directory(model, device, fake_dir,  1, "AI/Fake")
    real_preds, real_labels = evaluate_directory(model, device, real_dir,  0, "Human/Real")

    all_preds  = fake_preds  + real_preds
    all_labels = fake_labels + real_labels

    if not all_labels:
        print("  No files evaluated.")
        return

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n  Summary:")
    print(f"  Accuracy : {acc:.1%}")
    print(f"  F1 Macro : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=["Human (0)", "AI/Fake (1)"],
                                 zero_division=0))
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print(f"           Pred:Human  Pred:AI")
    print(f"  Actual:Human    {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"  Actual:AI/Fake  {cm[1][0]:>6}      {cm[1][1]:>6}")


if __name__ == "__main__":
    device = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  SpectrogramCNN Evaluation")
    print(f"{'='*70}")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Device : {device}")

    model = load_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params : {n_params:,}")

    # Evaluate WhatsApp-converted dataset (training domain)
    run_evaluation(
        name     = "WhatsApp-Converted Dataset (Training Domain)",
        fake_dir = os.path.join(BACKEND_DIR, "data", "whatsapp_dataset", "fake"),
        real_dir = os.path.join(BACKEND_DIR, "data", "whatsapp_dataset", "real"),
        model    = model,
        device   = device,
    )

    # Evaluate original raw dataset (out-of-domain test)
    run_evaluation(
        name     = "Original Raw Dataset (Out-of-Domain Test)",
        fake_dir = os.path.join(PROJECT_ROOT, "Fake", "Fake"),
        real_dir = os.path.join(PROJECT_ROOT, "Real", "Real"),
        model    = model,
        device   = device,
    )

    print(f"\n{'='*70}")
    print("  Evaluation complete.")
    print("  Next step: python main.py")
    print(f"{'='*70}\n")
