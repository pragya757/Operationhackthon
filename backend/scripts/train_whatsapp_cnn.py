"""
train_whatsapp_cnn.py
======================
Fine-tune (or train from scratch) the SpectrogramCNN on the
WhatsApp-converted dataset (data/whatsapp_dataset/fake/ + real/).

Architecture: identical to models/voice_clone/model.py
  Conv[32→64→128→256] → GAP → Linear(256) → Linear(2)

Training strategy:
  - Strong data augmentation (time-stretch, pitch-shift, noise, crop)
  - 80/20 stratified train/val split
  - AdamW optimizer + CosineAnnealingLR scheduler
  - Early stopping on val F1 (patience=15)
  - Best model saved to models/spectrogram_cnn_whatsapp.pth

Usage:
  python scripts/train_whatsapp_cnn.py
"""

import os
import sys
import time
import random
import numpy as np

# Fix Windows console encoding (cp1252 can't print Unicode chars like checkmarks)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BACKEND_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio.transforms as T

try:
    import librosa
except ImportError:
    os.system("pip install librosa")
    import librosa

try:
    import soundfile as sf
except ImportError:
    os.system("pip install soundfile")
    import soundfile as sf

try:
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
except ImportError:
    os.system("pip install scikit-learn")
    from sklearn.metrics import f1_score, confusion_matrix, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join(BACKEND_DIR, "data", "whatsapp_dataset")
FAKE_DIR     = os.path.join(DATASET_DIR, "fake")
REAL_DIR     = os.path.join(DATASET_DIR, "real")
OUTPUT_PATH  = os.path.join(BACKEND_DIR, "models", "spectrogram_cnn_whatsapp.pth")

SAMPLE_RATE  = 16_000
MAX_DURATION = 10.0
MAX_SAMPLES  = int(SAMPLE_RATE * MAX_DURATION)
N_FFT        = 1024
HOP_LENGTH   = 512
N_MELS       = 128
POWER        = 2.0

BATCH_SIZE   = 8      # small dataset — small batch
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
PATIENCE     = 15
VAL_SPLIT    = 0.20
SEED         = 42

# ── Augmentation params ───────────────────────────────────────────────────────
AUG_PROB         = 0.80    # probability of applying any augmentation per sample
NOISE_LEVEL      = 0.003   # additive gaussian noise amplitude
PITCH_SHIFT_RANGE = (-2, 2) # semitones
TIME_STRETCH_RANGE = (0.85, 1.15)
RANDOM_CROP_PROB = 0.5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── CNN Architecture (matches spectrogram_detector.py exactly) ─────────────
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
    def forward(self, x):
        return self.block(x)


class SpectrogramCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_dropout=0.20):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 32,  dropout=base_dropout)
        self.block2 = ConvBlock(32,  64,           dropout=base_dropout)
        self.block3 = ConvBlock(64,  128,          dropout=base_dropout + 0.05)
        self.block4 = ConvBlock(128, 256,          dropout=base_dropout + 0.10)
        self.gap    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)


# ── Audio → Spectrogram ────────────────────────────────────────────────────
_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT,
    hop_length=HOP_LENGTH, n_mels=N_MELS, power=POWER
)
_db_transform  = T.AmplitudeToDB(stype="power", top_db=80)


def audio_to_spec(y: np.ndarray) -> torch.Tensor:
    """float32 array → normalized log-mel spectrogram tensor (1, N_MELS, T)"""
    # Pad or crop to MAX_SAMPLES
    n = len(y)
    if n < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - n))
    else:
        y = y[:MAX_SAMPLES]

    waveform = torch.from_numpy(y).float().unsqueeze(0)  # (1, T)
    mel      = _mel_transform(waveform)   # (1, N_MELS, T_frames)
    logmel   = _db_transform(mel)         # (1, N_MELS, T_frames)

    # Per-sample normalization
    mu    = logmel.mean()
    sigma = logmel.std() + 1e-6
    norm  = (logmel - mu) / sigma
    return norm  # (1, N_MELS, T_frames)


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply random augmentations to raw waveform."""
    if random.random() > AUG_PROB:
        return y

    ops = []

    # 1. Additive Gaussian noise
    if random.random() < 0.6:
        noise = np.random.normal(0, NOISE_LEVEL, y.shape).astype(np.float32)
        y = np.clip(y + noise, -1.0, 1.0)
        ops.append("noise")

    # 2. Time stretch
    if random.random() < 0.5:
        rate = random.uniform(*TIME_STRETCH_RANGE)
        try:
            y = librosa.effects.time_stretch(y, rate=rate)
        except Exception:
            pass
        ops.append(f"stretch({rate:.2f})")

    # 3. Pitch shift
    if random.random() < 0.5:
        steps = random.uniform(*PITCH_SHIFT_RANGE)
        try:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        except Exception:
            pass
        ops.append(f"pitch({steps:+.1f})")

    # 4. Random crop (take a random 8-10s window)
    if random.random() < RANDOM_CROP_PROB and len(y) > int(sr * 8):
        max_start = len(y) - int(sr * 8)
        start = random.randint(0, max_start)
        y = y[start:]
        ops.append(f"crop@{start/sr:.1f}s")

    # 5. Volume jitter ±3dB
    if random.random() < 0.4:
        gain_db = random.uniform(-3.0, 3.0)
        gain = 10 ** (gain_db / 20.0)
        y = np.clip(y * gain, -1.0, 1.0)
        ops.append(f"vol({gain_db:+.1f}dB)")

    return y.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────
WINDOWS_PER_FILE = 5   # generate 5 random 10s windows per long recording during training

class WhatsAppDataset(Dataset):
    """
    Loads WAV files from fake/ (label=1) and real/ (label=0).
    During TRAINING: draws WINDOWS_PER_FILE random 10s windows from each long file.
    During VALIDATION: uses the MIDDLE 10s window for stable evaluation.
    """
    def __init__(self, file_label_pairs: list, training: bool = True):
        self.training = training
        if training:
            # Expand: each file becomes WINDOWS_PER_FILE entries
            self.pairs = []
            for path, label in file_label_pairs:
                for _ in range(WINDOWS_PER_FILE):
                    self.pairs.append((path, label))
        else:
            self.pairs = file_label_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        try:
            if self.training:
                # Load full file duration first to know the length
                duration_info = librosa.get_duration(path=path)
                if duration_info > MAX_DURATION:
                    # Pick a random start point leaving room for a full 10s window
                    max_offset = duration_info - MAX_DURATION
                    offset = random.uniform(0, max_offset)
                    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True,
                                        offset=offset, duration=MAX_DURATION)
                else:
                    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
            else:
                # Validation: use middle 10s for stable, reproducible eval
                duration_info = librosa.get_duration(path=path)
                offset = max(0.0, (duration_info - MAX_DURATION) / 2.0)
                y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True,
                                    offset=offset, duration=MAX_DURATION)
        except Exception as e:
            print(f"[Dataset] Failed to load {path}: {e}")
            y   = np.zeros(MAX_SAMPLES, dtype=np.float32)
            sr  = SAMPLE_RATE

        if self.training:
            y = augment(y, sr)

        spec = audio_to_spec(y)  # (1, N_MELS, T)
        return spec, torch.tensor(label, dtype=torch.long)


def load_dataset(training: bool = True, val_split: float = VAL_SPLIT):
    """Load all files, stratified split into train and val."""
    fake_files = [(os.path.join(FAKE_DIR, f), 1)
                  for f in sorted(os.listdir(FAKE_DIR)) if f.endswith('.wav')]
    real_files = [(os.path.join(REAL_DIR, f), 0)
                  for f in sorted(os.listdir(REAL_DIR)) if f.endswith('.wav')]

    print(f"  Fake files : {len(fake_files)}")
    print(f"  Real files : {len(real_files)}")

    # Stratified val split
    n_fake_val = max(1, int(len(fake_files) * val_split))
    n_real_val = max(1, int(len(real_files) * val_split))

    val_pairs   = fake_files[:n_fake_val] + real_files[:n_real_val]
    train_files = fake_files[n_fake_val:] + real_files[n_real_val:]
    random.shuffle(train_files)

    n_train_fake = sum(l for _, l in train_files)
    n_train_real = sum(1-l for _, l in train_files)
    print(f"  Train files: {len(train_files)} files x {WINDOWS_PER_FILE} windows = {len(train_files)*WINDOWS_PER_FILE} samples  (fake={n_train_fake}, real={n_train_real})")
    print(f"  Val files  : {len(val_pairs)}  (fake={sum(l for _,l in val_pairs)}, real={sum(1-l for _,l in val_pairs)})")

    train_ds = WhatsAppDataset(train_files, training=True)
    val_ds   = WhatsAppDataset(val_pairs,   training=False)
    return train_ds, val_ds


# ── Training Loop ─────────────────────────────────────────────────────────────
def train():
    set_seed(SEED)
    device = _select_device()
    print(f"\n{'='*60}")
    print(f"  SpectrogramCNN Training — WhatsApp Dataset")
    print(f"{'='*60}")
    print(f"  Device   : {device}")
    print(f"  Fake dir : {FAKE_DIR}")
    print(f"  Real dir : {REAL_DIR}")
    print(f"  Output   : {OUTPUT_PATH}")
    print()

    # ── Verify dataset exists ─────────────────────────────────────────────
    for d in [FAKE_DIR, REAL_DIR]:
        if not os.path.exists(d):
            print(f"[ERROR] Dataset directory not found: {d}")
            print("Run: python scripts/convert_to_whatsapp.py first!")
            sys.exit(1)
        files = [f for f in os.listdir(d) if f.endswith('.wav')]
        if not files:
            print(f"[ERROR] No WAV files in {d}")
            sys.exit(1)

    train_ds, val_ds = load_dataset()

    # ── Weighted sampler to handle class imbalance ─────────────────────────
    labels = [lbl for _, lbl in train_ds.pairs]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SpectrogramCNN(in_channels=1, num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params:,}")

    # ── Try loading existing model for fine-tuning ─────────────────────────
    if os.path.exists(OUTPUT_PATH):
        try:
            state = torch.load(OUTPUT_PATH, map_location=device, weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=True)
            print(f"  [OK] Fine-tuning from existing: {OUTPUT_PATH}")
        except Exception as e:
            print(f"  Starting fresh (existing model load failed: {e})")
    else:
        print(f"  Starting fresh training")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_f1     = 0.0
    best_epoch  = 0
    no_improve  = 0

    print(f"\n  Training for up to {EPOCHS} epochs (patience={PATIENCE})")
    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val F1':>7}  {'Val Acc':>7}  {'LR':>10}")
    print(f"  {'-'*65}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_spec, batch_label in train_loader:
            batch_spec  = batch_spec.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            logits = model(batch_spec)
            loss   = criterion(logits, batch_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch_spec, batch_label in val_loader:
                batch_spec  = batch_spec.to(device)
                batch_label = batch_label.to(device)
                logits = model(batch_spec)
                loss   = criterion(logits, batch_label)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_label.cpu().numpy())

        val_loss /= max(len(val_loader), 1)

        if len(all_labels) > 0:
            val_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        else:
            val_f1  = 0.0
            val_acc = 0.0

        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        marker = "  [BEST]" if val_f1 > best_f1 else ""
        print(f"  {epoch:>6}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_f1:>7.4f}  {val_acc:>7.2%}  {lr_now:>10.2e}{marker}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            no_improve = 0
            # Save best model as bare state_dict
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_PATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best epoch  : {best_epoch}")
    print(f"  Best val F1 : {best_f1:.4f}")
    print(f"  Model saved : {OUTPUT_PATH}")
    print(f"{'='*60}")

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n  Final Evaluation on Validation Set:")
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for batch_spec, batch_label in val_loader:
            batch_spec  = batch_spec.to(device)
            logits = model(batch_spec)
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_label.cpu().numpy())

    print(classification_report(all_labels, all_preds,
                                 target_names=["Human (0)", "AI/Fake (1)"],
                                 zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"\nNext step: python main.py")


if __name__ == "__main__":
    train()
