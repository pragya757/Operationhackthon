"""
convert_to_whatsapp.py
======================
Converts Fake/ and Real/ WAV dataset into WhatsApp-grade audio.

WhatsApp voice calls use Opus codec at ~8-16 kHz with:
  - narrowband / wideband compression
  - slight codec noise / ringing
  - automatic gain control (AGC) normalization
  - 8kHz downsampling then upcasted to 16kHz for model input

Pipeline per file:
  1. Load original WAV (any sample rate)
  2. Downmix to mono
  3. Resample to 8000 Hz (WhatsApp Opus narrowband rate)
  4. Apply AGC normalization (normalize to -20 dBFS RMS)
  5. Add mild codec noise (simulate Opus quantization artifacts)
  6. Apply bandwidth-limiting EQ (cut >3800 Hz like Opus NB)
  7. Upsample back to 16000 Hz for CNN compatibility
  8. Save to output directory

Usage:
  python scripts/convert_to_whatsapp.py
"""

import os
import sys
import numpy as np
import time

# Add backend root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BACKEND_DIR)

try:
    import soundfile as sf
except ImportError:
    print("Installing soundfile...")
    os.system("pip install soundfile")
    import soundfile as sf

try:
    import librosa
except ImportError:
    print("Installing librosa...")
    os.system("pip install librosa")
    import librosa


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

FAKE_INPUT  = os.path.join(PROJECT_ROOT, "Fake", "Fake")
REAL_INPUT  = os.path.join(PROJECT_ROOT, "Real", "Real")

FAKE_OUTPUT = os.path.join(BACKEND_DIR, "data", "whatsapp_dataset", "fake")
REAL_OUTPUT = os.path.join(BACKEND_DIR, "data", "whatsapp_dataset", "real")

os.makedirs(FAKE_OUTPUT, exist_ok=True)
os.makedirs(REAL_OUTPUT, exist_ok=True)

# ── WhatsApp Simulation Parameters ────────────────────────────────────────────
WHATSAPP_CODEC_SR  = 8000    # Opus narrowband codec rate
MODEL_SR           = 16000   # Target SR for CNN input
TARGET_RMS_DB      = -20.0   # AGC target (-20 dBFS)
CODEC_NOISE_LEVEL  = 0.001   # Quantization noise amplitude
CUTOFF_HZ          = 3800    # Opus NB bandwidth limit (Hz)


def apply_agc(y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Automatic Gain Control — normalize RMS to target_db dBFS."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-8:
        return y  # silence — don't touch
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    gain = min(gain, 100.0)  # safety cap: max +40dB gain
    y = y * gain
    return np.clip(y, -1.0, 1.0)


def apply_bandwidth_limit(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Low-pass filter to simulate Opus narrowband codec bandwidth limit."""
    from scipy.signal import butter, sosfilt
    nyq = sr / 2.0
    if cutoff_hz >= nyq:
        return y  # no filtering needed
    norm_cutoff = cutoff_hz / nyq
    sos = butter(N=4, Wn=norm_cutoff, btype='low', output='sos')
    return sosfilt(sos, y)


def add_codec_noise(y: np.ndarray, level: float = 0.001) -> np.ndarray:
    """Add mild quantization / codec noise (simulates Opus compression artifacts)."""
    noise = np.random.normal(0.0, level, size=y.shape).astype(np.float32)
    return np.clip(y + noise, -1.0, 1.0)


def convert_to_whatsapp(input_path: str, output_path: str) -> bool:
    """Full WhatsApp audio simulation pipeline."""
    try:
        # 1. Load
        y, sr = librosa.load(input_path, sr=None, mono=True)
        print(f"  Loaded: {sr}Hz, {len(y)/sr:.2f}s, max={np.max(np.abs(y)):.3f}")

        # 2. Resample to WhatsApp codec rate (8kHz)
        if sr != WHATSAPP_CODEC_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=WHATSAPP_CODEC_SR)

        # 3. AGC normalization (simulate WhatsApp's audio normalization)
        y = apply_agc(y, TARGET_RMS_DB)

        # 4. Bandwidth limiting (simulate Opus NB 3.8kHz cutoff)
        try:
            y = apply_bandwidth_limit(y, WHATSAPP_CODEC_SR, CUTOFF_HZ)
        except Exception as e:
            print(f"    [warn] bandwidth filter skipped: {e}")

        # 5. Codec noise (simulate Opus quantization artifacts)
        y = add_codec_noise(y, CODEC_NOISE_LEVEL)

        # 6. Upsample to 16kHz (CNN input requires 16kHz)
        y = librosa.resample(y, orig_sr=WHATSAPP_CODEC_SR, target_sr=MODEL_SR)

        # 7. Final clip
        y = np.clip(y, -1.0, 1.0).astype(np.float32)

        # 8. Save
        sf.write(output_path, y, MODEL_SR, subtype='PCM_16')
        print(f"  Saved: {output_path} ({MODEL_SR}Hz, {len(y)/MODEL_SR:.2f}s)")
        return True

    except Exception as e:
        print(f"  [ERROR] {input_path}: {e}")
        return False


def process_directory(input_dir: str, output_dir: str, label: str) -> int:
    """Process all WAV files in a directory."""
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    print(f"\n{'='*60}")
    print(f"Processing {label}: {len(files)} files")
    print(f"  Input  -> {input_dir}")
    print(f"  Output -> {output_dir}")
    print(f"{'='*60}")

    success = 0
    for i, fname in enumerate(sorted(files), 1):
        in_path  = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        print(f"\n[{i}/{len(files)}] {fname}")
        if convert_to_whatsapp(in_path, out_path):
            success += 1

    print(f"\n[OK] {label}: {success}/{len(files)} converted successfully")
    return success


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  WhatsApp Audio Conversion Pipeline")
    print("  Dataset -> WhatsApp Opus Simulation")
    print("="*60)
    print(f"  Fake input  : {FAKE_INPUT}")
    print(f"  Real input  : {REAL_INPUT}")
    print(f"  Fake output : {FAKE_OUTPUT}")
    print(f"  Real output : {REAL_OUTPUT}")
    print(f"  Codec SR    : {WHATSAPP_CODEC_SR} Hz")
    print(f"  Model SR    : {MODEL_SR} Hz")
    print(f"  AGC target  : {TARGET_RMS_DB} dBFS")
    print(f"  BW cutoff   : {CUTOFF_HZ} Hz")

    t0 = time.time()
    n_fake = process_directory(FAKE_INPUT,  FAKE_OUTPUT, "FAKE (AI Voices)")
    n_real = process_directory(REAL_INPUT,  REAL_OUTPUT, "REAL (Human Voices)")
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  CONVERSION COMPLETE in {elapsed:.1f}s")
    print(f"  Fake samples : {n_fake}")
    print(f"  Real samples : {n_real}")
    print(f"  Total        : {n_fake + n_real}")
    print(f"  Output dir   : {os.path.join(BACKEND_DIR, 'data', 'whatsapp_dataset')}")
    print(f"{'='*60}")
    print("\nNext step: python scripts/train_whatsapp_cnn.py")
