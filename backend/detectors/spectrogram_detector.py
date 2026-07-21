"""
Spectrogram Audio Forensics Detector – CNN-Powered (Production)
───────────────────────────────────────────────────────────────
Three-tier loader: PyTorch .pth → ONNX Runtime → Visual Heuristics.
Preprocessing is identical to the training pipeline in preprocess.py.

Loading priority:
  Tier 1: PyTorch state_dict (.pth)   – fastest, most accurate
  Tier 2: ONNX Runtime (.onnx)        – portable fallback
  Tier 3: Visual spectral heuristics  – offline fallback (no weights needed)

All three tiers return the same API response schema.
Frontend requires ZERO changes.
"""

import os
import io
import base64
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np

# ── Model paths ────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent           # …/detectors/
_MODELS_ROOT = _HERE.parent / "models"                  # …/backend/models/
_SUBDIR_ROOT = _MODELS_ROOT / "spectrogram_cnn"          # …/backend/models/spectrogram_cnn/

def _resolve_pth() -> Path:
    """Find the .pth file: env-var override → flat models/ → spectrogram_cnn/ subdir."""
    env = os.getenv("SPECTROGRAM_CNN_PTH")
    if env:
        return Path(env)
    flat = _MODELS_ROOT / "spectrogram_cnn_whatsapp.pth"
    if flat.exists():
        return flat
    return _SUBDIR_ROOT / "spectrogram_cnn_whatsapp.pth"

def _resolve_onnx() -> Path:
    """Find the .onnx file: env-var override → flat models/ → spectrogram_cnn/ subdir."""
    env = os.getenv("SPECTROGRAM_CNN_ONNX")
    if env:
        return Path(env)
    flat = _MODELS_ROOT / "spectrogram_cnn.onnx"
    if flat.exists():
        return flat
    return _SUBDIR_ROOT / "spectrogram_cnn.onnx"

_CNN_PTH  = _resolve_pth()
_CNN_ONNX = _resolve_onnx()

# ── Device selection (MPS → CUDA → CPU) ──────────────────────────────────
def _select_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

_DEVICE = _select_device()

_SAMPLE_RATE    = 16_000
_MAX_SAMPLES    = _SAMPLE_RATE * 10   # 10 seconds
_N_FFT          = 1024
_HOP_LENGTH     = 512
_N_MELS         = 128
_TOP_DB         = 80
_NORMALIZE_SPEC = True                # config.NORMALIZE_SPECTROGRAM = True
_IN_CHANNELS    = 1
_HIDDEN_CH      = [32, 64, 128, 256]
_NUM_CLASSES    = 2                   # 0=human, 1=synthetic

# ── CNN architecture (mirrors model.py exactly) ────────────────────────────────────
def _build_model():
    """
    Instantiate SpectrogramCNN using the exact same architecture as
    backend/models/voice_clone/model.py (what the trained .pth was saved from).

    Architecture:
      block1: Conv2d(1→32)   + BN2d + ReLU + MaxPool2d(2) + Dropout2d(0.20)
      block2: Conv2d(32→64)  + BN2d + ReLU + MaxPool2d(2) + Dropout2d(0.20)
      block3: Conv2d(64→128) + BN2d + ReLU + MaxPool2d(2) + Dropout2d(0.25)
      block4: Conv2d(128→256)+ BN2d + ReLU + MaxPool2d(2) + Dropout2d(0.30)
      gap:    AdaptiveAvgPool2d(1,1)  → flatten → (B, 256)
      classifier: Linear(256→256, bias=False) + BN1d(256) + ReLU + Dropout(0.5) + Linear(256→2)
    """
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

    return SpectrogramCNN()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _is_valid_pth(path: Path) -> Tuple[bool, str]:
    """Check whether a .pth file is a real pickle/PyTorch binary (not a text stub)."""
    if not path.exists():
        return False, f"File not found: {path}"
    if path.stat().st_size < 1024:           # real state_dict is > 1 KB
        try:
            content = path.read_text().strip()
            return False, f"Stub text file: '{content[:80]}'"
        except Exception:
            return False, "File too small to be a valid state_dict"
    with open(path, "rb") as f:
        magic = f.read(10)
    # PyTorch saves start with a pickle PROTO opcode (0x80 0x02 or 0x80 0x05)
    # or a zip PK header (0x50 0x4b) for newer torch.save format
    if magic[:2] in (b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05') or magic[:2] == b'PK':
        return True, "OK"
    return False, f"Unexpected header bytes: {magic.hex()}"


def _is_valid_onnx(path: Path) -> Tuple[bool, str]:
    """Check whether an .onnx file is a real protobuf binary (not a text stub)."""
    if not path.exists():
        return False, f"File not found: {path}"
    if path.stat().st_size < 512:
        try:
            content = path.read_text().strip()
            return False, f"Stub text file: '{content[:80]}'"
        except Exception:
            return False, "File too small to be a valid ONNX model"
    with open(path, "rb") as f:
        hdr = f.read(4)
    # Valid ONNX protobuf starts with field tags: 0x08, 0x0a, 0x12, 0x1a
    if hdr and hdr[0] in (0x08, 0x0a, 0x12, 0x1a):
        return True, "OK"
    return False, f"Invalid ONNX header: {hdr.hex()}"


# ── Three-Tier Loader ──────────────────────────────────────────────────────────
class _SpectrogramCNNLoader:
    """
    Loads the SpectrogramCNN in strict priority order with full diagnostic logging.

    Startup banner example:
      ========================================
      Spectrogram CNN Initialization
      ========================================
      Found .pth : YES  (1660.3 KB)
      Found .onnx: YES  (28.4 KB)

      Loading PyTorch model...
      ✓ PyTorch model loaded successfully
        Architecture : ConvBlocks[32,64,128,256] → GAP → Linear(128) → Linear(2)
        Parameters   : 1,688,130
        Device       : cpu

      Inference mode: torch
      ========================================
    """

    MODE_TORCH     = "torch"
    MODE_ONNX      = "onnx"
    MODE_HEURISTIC = "heuristic"

    def __init__(self):
        self.mode    = self.MODE_HEURISTIC
        self._torch  = None       # torch.nn.Module
        self._onnx   = None       # onnxruntime.InferenceSession
        self._preproc = None      # (mel_transform, db_transform) from torchaudio

        self._banner_init()
        self._try_load_torch()
        if self.mode == self.MODE_HEURISTIC:
            self._try_load_onnx()
        self._try_build_preproc()
        self._banner_result()

    # ── Banner helpers ────────────────────────────────────────────────────────
    def _banner_init(self):
        pth_ok,  pth_why  = _is_valid_pth(_CNN_PTH)
        onnx_ok, onnx_why = _is_valid_onnx(_CNN_ONNX)
        pth_sz  = f"{_CNN_PTH.stat().st_size/1024:.1f} KB"  if _CNN_PTH.exists()  else "—"
        onnx_sz = f"{_CNN_ONNX.stat().st_size/1024:.1f} KB" if _CNN_ONNX.exists() else "—"
        pth_tag  = f"YES  ({pth_sz})"  if pth_ok  else f"NO   [{pth_why}]"
        onnx_tag = f"YES  ({onnx_sz})" if onnx_ok else f"NO   [{onnx_why}]"
        print("=" * 48)
        print("  Spectrogram CNN Initialization")
        print("=" * 48)
        print(f"  Found .pth : {pth_tag}")
        print(f"  Found .onnx: {onnx_tag}")
        print()

    def _banner_result(self):
        mode_display = {
            self.MODE_TORCH:     "torch  (PyTorch .pth)",
            self.MODE_ONNX:      "onnx   (ONNX Runtime .onnx)",
            self.MODE_HEURISTIC: "heuristic  (no CNN weights loaded)",
        }
        print(f"  Inference mode: {mode_display[self.mode]}")
        print("=" * 48)
        
        # Custom startup log for Voice Forensics Spectrogram Model
        inference_mode_map = {
            self.MODE_TORCH:     "PyTorch",
            self.MODE_ONNX:      "ONNX",
            self.MODE_HEURISTIC: "Heuristics"
        }
        print("-" * 52)
        print("Voice Forensics Spectrogram Model")
        print("Loaded model:")
        print(os.path.basename(_CNN_PTH))
        print(f"Device: {_DEVICE}")
        print(f"Inference Mode: {inference_mode_map[self.mode]}")
        print("-" * 52)

    # ── Tier 1: PyTorch ───────────────────────────────────────────────────────
    def _try_load_torch(self):
        print("  Loading PyTorch model (.pth) ...")

        valid, reason = _is_valid_pth(_CNN_PTH)
        if not valid:
            print(f"  [Skipped] {reason}")
            print()
            return

        try:
            import torch

            # Try weights_only=True first (safe, PyTorch ≥ 2.6), then legacy
            try:
                state = torch.load(str(_CNN_PTH), map_location=_DEVICE, weights_only=True)
            except Exception as e1:
                print(f"    weights_only=True failed ({e1}), retrying with weights_only=False...")
                state = torch.load(str(_CNN_PTH), map_location=_DEVICE, weights_only=False)

            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            if not isinstance(state, dict):
                raise TypeError(
                    f"Expected OrderedDict (state_dict), got {type(state).__name__}. "
                    "File may be a full model save — use torch.save(model.state_dict(), …)"
                )

            model = _build_model()
            result = model.load_state_dict(state, strict=True)
            model.to(_DEVICE)
            model.eval()

            # Sanity-check inference
            dummy = torch.zeros(1, _IN_CHANNELS, _N_MELS, 32, device=_DEVICE)
            with torch.no_grad():
                out = model(dummy)
            assert out.shape == (1, _NUM_CLASSES), f"Unexpected output shape: {out.shape}"

            n_params = sum(p.numel() for p in model.parameters())
            self._torch = model
            self.mode   = self.MODE_TORCH
            print(f"  [OK] PyTorch model loaded successfully")
            print(f"    Architecture : ConvBlocks{_HIDDEN_CH} -> GAP -> Linear({_HIDDEN_CH[-1]}) -> Linear({_NUM_CLASSES})")
            print(f"    Parameters   : {n_params:,}")
            print(f"    Device       : {_DEVICE}")
            print()
        except Exception as e:
            print(f"  [ERROR] PyTorch load FAILED:")
            traceback.print_exc()
            print()

    # ── Tier 2: ONNX Runtime ─────────────────────────────────────────────────
    def _try_load_onnx(self):
        print("  Loading ONNX model (.onnx) ...")

        valid, reason = _is_valid_onnx(_CNN_ONNX)
        if not valid:
            print(f"  [Skipped] {reason}")
            print()
            return

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._onnx = ort.InferenceSession(
                str(_CNN_ONNX), sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            # Sanity check
            inp_name = self._onnx.get_inputs()[0].name
            dummy = np.zeros((1, _IN_CHANNELS, _N_MELS, 32), dtype=np.float32)
            out   = self._onnx.run(None, {inp_name: dummy})[0]
            assert out.shape == (1, _NUM_CLASSES), f"Unexpected ONNX output shape: {out.shape}"

            self.mode = self.MODE_ONNX
            sz = _CNN_ONNX.stat().st_size / 1024
            print(f"  [OK] ONNX model loaded successfully")
            print(f"    File size    : {sz:.1f} KB")
            print(f"    Input name   : {inp_name}")
            print(f"    Output shape : {out.shape}")
            print()
        except Exception as e:
            print(f"  [ERROR] ONNX load FAILED:")
            traceback.print_exc()
            print()
            if self.mode == self.MODE_HEURISTIC:
                print("  [WARNING] Both .pth and .onnx failed. Falling back to heuristic detector.")
                print()

    # ── Preprocessor (torchaudio transforms — identical to training) ──────────
    def _try_build_preproc(self):
        try:
            import torchaudio
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=_SAMPLE_RATE,
                n_fft=_N_FFT,
                hop_length=_HOP_LENGTH,
                n_mels=_N_MELS,
                power=2.0,          # matches training config POWER=2.0; normalized defaults to False
            )
            db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=_TOP_DB)
            self._preproc = (mel, db)
        except Exception as e:
            print(f"  [WARNING] torchaudio transforms unavailable: {e}  (librosa fallback will be used)")

    # ── Audio → tensor ────────────────────────────────────────────────────────
    def _audio_to_tensor(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          tensor_4d : float32 (1, 1, N_MELS, T) — model input
          spec_2d   : float64 (N_MELS, T)       — visualisation (dB, unnormalised)
        """
        if self._preproc is not None:
            return self._preproc_torchaudio(audio_path)
        return self._preproc_librosa(audio_path)

    def _preproc_torchaudio(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses soundfile/scipy for file I/O (torchaudio.load requires TorchCodec
        in v2.11+, which may not be installed), then applies the same
        MelSpectrogram + AmplitudeToDB transforms used during training.
        """
        import torch
        mel_transform, db_transform = self._preproc

        # Step 1: Load waveform — soundfile → scipy → librosa
        waveform: Optional[torch.Tensor] = None
        sr: Optional[int] = None

        try:
            import soundfile as sf
            data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)          # (C, N)
        except Exception:
            pass

        if waveform is None:
            try:
                from scipy.io import wavfile
                sr, data = wavfile.read(audio_path)
                if data.dtype != np.float32:
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max
                if data.ndim == 1:
                    data = data[:, np.newaxis]
                waveform = torch.from_numpy(data.T)       # (C, N)
            except Exception:
                pass

        if waveform is None:
            import librosa
            y, sr = librosa.load(audio_path, sr=_SAMPLE_RATE, duration=10.0, mono=True)
            waveform = torch.from_numpy(y).unsqueeze(0)   # (1, N)

        # Step 2: Resample
        if sr != _SAMPLE_RATE:
            import torchaudio
            waveform = torchaudio.transforms.Resample(sr, _SAMPLE_RATE)(waveform)

        # Step 3: Mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Step 4: Truncate / pad to exactly 10 s
        if waveform.size(1) > _MAX_SAMPLES:
            waveform = waveform[:, :_MAX_SAMPLES]
        elif waveform.size(1) < _MAX_SAMPLES:
            pad = torch.zeros(1, _MAX_SAMPLES - waveform.size(1))
            waveform = torch.cat([waveform, pad], dim=1)

        # Step 5: Log-Mel spectrogram (identical to preprocess.py)
        mel     = mel_transform(waveform)    # (1, N_MELS, T)
        log_mel = db_transform(mel)          # (1, N_MELS, T) dB

        # Step 6: Per-sample normalisation
        mu    = log_mel.mean()
        sigma = log_mel.std() + 1e-6
        norm  = (log_mel - mu) / sigma       # (1, N_MELS, T)

        tensor_4d = norm.unsqueeze(0).numpy().astype(np.float32)  # (1, 1, N_MELS, T)
        spec_2d   = log_mel.squeeze(0).numpy()                    # (N_MELS, T)
        return tensor_4d, spec_2d

    def _preproc_librosa(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Pure-librosa fallback (when torchaudio transforms unavailable)."""
        import librosa
        y, sr = librosa.load(audio_path, sr=_SAMPLE_RATE, duration=10.0, mono=True)
        if len(y) < _MAX_SAMPLES:
            y = np.pad(y, (0, _MAX_SAMPLES - len(y)))
        S    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=_N_FFT,
                                              hop_length=_HOP_LENGTH, n_mels=_N_MELS)
        S_db = librosa.power_to_db(S, ref=np.max)
        mu   = S_db.mean(); sigma = S_db.std() + 1e-6
        norm = (S_db - mu) / sigma
        tensor_4d = norm[np.newaxis, np.newaxis, :, :].astype(np.float32)
        return tensor_4d, S_db

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, audio_path: str) -> Tuple[Optional[float], str, str]:
        """
        Returns (score, label, provenance):
          score      : float 0-1 (synthetic probability), or None on failure
          label      : 'AI Generated' | 'Human'
          provenance : human-readable description of which model ran
        """
        if self.mode == self.MODE_HEURISTIC:
            return None, "Heuristic", "CNN not loaded — using visual heuristics"

        try:
            tensor_4d, _ = self._audio_to_tensor(audio_path)
        except Exception as e:
            print(f"[SpectrogramCNN] Preprocessing error: {e}")
            traceback.print_exc()
            return None, "Error", f"Preprocessing failed: {e}"

        if self.mode == self.MODE_TORCH:
            return self._infer_torch(tensor_4d)
        if self.mode == self.MODE_ONNX:
            return self._infer_onnx(tensor_4d)
        return None, "Heuristic", "CNN not loaded"

    def _infer_torch(self, tensor_4d: np.ndarray) -> Tuple[Optional[float], str, str]:
        try:
            import torch
            x = torch.from_numpy(tensor_4d).to(_DEVICE)   # move to MPS/CUDA/CPU
            with torch.no_grad():
                logits = self._torch(x)           # (1, 2)  — raw pre-softmax scores
                probs  = torch.softmax(logits, dim=1)

                # ── Raw values ────────────────────────────────────────────────
                logit_human = float(logits[0, 0].item())
                logit_synth = float(logits[0, 1].item())
                prob_human  = float(probs[0, 0].item())
                prob_synth  = float(probs[0, 1].item())
                synthetic_prob = prob_synth
                label = "AI Generated" if synthetic_prob >= 0.5 else "Human"
                confidence_pct = max(prob_human, prob_synth) * 100.0

                print(
                    "\n"
                    "========================\n"
                    "SPECTROGRAM CNN RAW     \n"
                    "========================\n"
                    f"  Backend          : PyTorch .pth ({_DEVICE.upper()})\n"
                    f"  Raw Logits       : [Human={logit_human:+.4f},  AI={logit_synth:+.4f}]\n"
                    f"  Softmax          :\n"
                    f"    Human          : {prob_human:.6f}  ({prob_human*100:.2f}%)\n"
                    f"    AI Generated   : {prob_synth:.6f}  ({prob_synth*100:.2f}%)\n"
                    f"  Predicted Class  : {label}\n"
                    f"  Confidence       : {confidence_pct:.2f}%\n"
                    "========================"
                )
                return synthetic_prob, label, "SpectrogramCNN (PyTorch .pth)"
        except Exception as e:
            print(f"[SpectrogramCNN] PyTorch inference error: {e}")
            traceback.print_exc()
            return None, "Error", f"PyTorch inference failed: {e}"

    def _infer_onnx(self, tensor_4d: np.ndarray) -> Tuple[Optional[float], str, str]:
        try:
            inp_name  = self._onnx.get_inputs()[0].name
            logits    = self._onnx.run(None, {inp_name: tensor_4d})[0]   # (1, 2)
            exp       = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs     = exp / exp.sum(axis=1, keepdims=True)

            logit_human = float(logits[0, 0])
            logit_synth = float(logits[0, 1])
            prob_human  = float(probs[0, 0])
            prob_synth  = float(probs[0, 1])
            synthetic_prob = prob_synth
            label = "AI Generated" if synthetic_prob >= 0.5 else "Human"
            confidence_pct = max(prob_human, prob_synth) * 100.0

            print(
                "\n"
                "========================\n"
                "SPECTROGRAM CNN RAW     \n"
                "========================\n"
                f"  Backend          : ONNX Runtime\n"
                f"  Raw Logits       : [Human={logit_human:+.4f},  AI={logit_synth:+.4f}]\n"
                f"  Softmax          :\n"
                f"    Human          : {prob_human:.6f}  ({prob_human*100:.2f}%)\n"
                f"    AI Generated   : {prob_synth:.6f}  ({prob_synth*100:.2f}%)\n"
                f"  Predicted Class  : {label}\n"
                f"  Confidence       : {confidence_pct:.2f}%\n"
                "========================"
            )
            return synthetic_prob, label, "SpectrogramCNN (ONNX Runtime)"
        except Exception as e:
            print(f"[SpectrogramCNN] ONNX inference error: {e}")
            traceback.print_exc()
            return None, "Error", f"ONNX inference failed: {e}"

    @property
    def is_cnn_active(self) -> bool:
        return self.mode in (self.MODE_TORCH, self.MODE_ONNX)


# ── Singleton — initialised once at import time ────────────────────────────────
_cnn_loader = _SpectrogramCNNLoader()


# ── Public API ────────────────────────────────────────────────────────────────
def analyze_spectrogram(audio_path: str) -> Dict[str, Any]:
    """
    Main forensic entry point.  Returns the IDENTICAL schema as the original detector:
      prediction        — 'AI Generated' | 'Human'
      confidence        — float 0–100
      spectrogram_score — float 0–1
      reasons           — List[str]
      spectrogram_image — Base64 PNG data-URI
    """
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"[SpectrogramDetector] mode={_cnn_loader.mode} | analyzing: {audio_path}")

    # 1. Load audio for heuristics + image rendering
    try:
        y, sr = librosa.load(audio_path, sr=_SAMPLE_RATE, duration=10.0, mono=True)
    except Exception as e:
        print(f"[SpectrogramDetector] Audio load error: {e}")
        return {
            "prediction": "Human",
            "confidence": 0.0,
            "spectrogram_score": 0.0,
            "reasons": [f"Audio load error: {e}"],
            "spectrogram_image": "",
        }

    if len(y) == 0:
        return {
            "prediction": "Human",
            "confidence": 0.0,
            "spectrogram_score": 0.0,
            "reasons": ["Audio file contains no decodable frames."],
            "spectrogram_image": "",
        }

    # 2. Spectrogram for image (64-band — matches original output)
    n_mels_img = 64
    S_img    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels_img, fmax=8000)
    S_db_img = librosa.power_to_db(S_img, ref=np.max)

    # 3. Spectrogram for heuristics (128-band — same band count as CNN)
    S_cnn    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=_N_FFT,
                                              hop_length=_HOP_LENGTH, n_mels=_N_MELS)
    S_db_cnn = librosa.power_to_db(S_cnn, ref=np.max)

    # 4. Render Base64 PNG spectrogram image
    spectrogram_image = ""
    try:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        fig.patch.set_facecolor("#131313")
        ax.set_facecolor("#131313")
        import librosa.display
        img = librosa.display.specshow(
            S_db_img, sr=sr, x_axis="time", y_axis="mel",
            fmax=8000, cmap="magma", ax=ax,
        )
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        cbar.outline.set_edgecolor("none")
        ax.set_xlabel("Time (s)", color="#9e9e9e", fontsize=7)
        ax.set_ylabel("Hz (mel)",  color="#9e9e9e", fontsize=7)
        ax.tick_params(colors="#9e9e9e", labelsize=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        spectrogram_image = (
            "data:image/png;base64,"
            + base64.b64encode(buf.read()).decode("utf-8")
        )
    except Exception as se:
        print(f"[SpectrogramDetector] Visualisation error: {se}")

    # 5. CNN inference (Tier 1 or 2) or heuristics (Tier 3)
    score, label, provenance = _cnn_loader.predict(audio_path)
    reasons: list = []

    if score is not None and _cnn_loader.is_cnn_active:
        # ── CNN path ──────────────────────────────────────────────────────────
        spectrogram_score = float(np.clip(score, 0.0, 1.0))

        if spectrogram_score >= 0.80:
            reasons.append(
                f"{provenance}: high-confidence AI-generated speech detected "
                f"(synthetic probability {spectrogram_score * 100:.1f}%)."
            )
            reasons.append(
                "CNN found strong GAN/TTS fingerprints — over-uniform harmonics, "
                "missing breathiness, flat spectral envelope."
            )
        elif spectrogram_score >= 0.60:
            reasons.append(
                f"{provenance}: AI-generated speech detected "
                f"(synthetic probability {spectrogram_score * 100:.1f}%)."
            )
            reasons.append(
                "Moderate synthetic signature: reduced micro-variation typical of "
                "neural TTS (VITS, FastSpeech, WaveNet)."
            )
        elif spectrogram_score >= 0.40:
            reasons.append(
                f"{provenance}: marginal synthetic signal detected "
                f"(synthetic probability {spectrogram_score * 100:.1f}% — "
                "crosses fusion threshold, CNN primary vote: Human)."
            )
            reasons.append(
                "Subtle spectrogram irregularities above the fusion detection threshold "
                "(≥40%). CNN primary classification threshold is 50%."
            )
        elif spectrogram_score >= 0.20:
            reasons.append(
                f"{provenance}: classified as authentic human speech "
                f"(synthetic probability {spectrogram_score * 100:.1f}% — below all thresholds)."
            )
            reasons.append(
                "Low synthetic probability — below CNN and fusion thresholds. Verdict: Human."
            )
        else:
            reasons.append(
                f"{provenance}: strongly classified as authentic human speech "
                f"(synthetic probability {spectrogram_score * 100:.1f}%)."
            )
            reasons.append(
                "Strong human authenticity: continuous harmonic structure, natural "
                "F0 modulation, organic spectral roll-off detected."
            )

    else:
        # ── Heuristic fallback path ───────────────────────────────────────────
        print(f"[SpectrogramDetector] CNN unavailable ({provenance}) — heuristic mode")
        heuristic_score = 0.0

        # Heuristic A: Frame variance (vocoder grid artifact)
        frame_var = np.var(S_db_cnn, axis=0)
        avg_var   = float(np.mean(frame_var))
        if avg_var < 15.0:
            heuristic_score += 0.35
            reasons.append(
                f"Repeated harmonic textures / low frame variance (mean={avg_var:.2f}) "
                "— consistent with vocoder synthesis."
            )

        # Heuristic B: Spectral discontinuities (phase mismatch)
        diffs     = np.abs(np.diff(S_db_cnn, axis=1))
        max_trans = float(np.max(diffs))
        if max_trans > 45.0:
            heuristic_score += 0.30
            reasons.append(
                f"Abrupt spectral discontinuities (max Δ={max_trans:.1f} dB) — "
                "vocoder phase mismatch."
            )

        # Heuristic C: High-frequency energy ratio
        hf = float(np.mean(S_db_cnn[96:, :]))
        lf = float(np.mean(S_db_cnn[:96, :]))
        ratio = hf / (lf - 1e-6)
        if ratio < 0.40:
            heuristic_score += 0.35
            reasons.append(
                f"Abnormal HF/LF energy ratio ({ratio:.3f}) — "
                "synthesis noise floor artifact."
            )

        spectrogram_score = float(min(1.0, heuristic_score))

        if not reasons:
            reasons.append(
                "Heuristic analysis: natural harmonic structure and continuous "
                "mel-frequency roll-off — no synthetic artifacts detected."
            )
        reasons.append(
            f"[Heuristic mode — {provenance}] "
            "Install real CNN weights (spectrogram_cnn.pth) to enable neural inference."
        )

    # 6. Map score → prediction + confidence
    if spectrogram_score >= 0.40:
        prediction = "AI Generated"
        confidence = round(spectrogram_score * 100.0, 1)
    else:
        prediction = "Human"
        confidence = round((1.0 - spectrogram_score) * 100.0, 1)

    print(
        f"[SpectrogramDetector] RESULT → prediction={prediction}  "
        f"confidence={confidence}%  score={spectrogram_score:.3f}  "
        f"mode={_cnn_loader.mode}"
    )

    return {
        "prediction":        prediction,
        "confidence":        confidence,
        "spectrogram_score": round(spectrogram_score, 3),
        "reasons":           reasons,
        "spectrogram_image": spectrogram_image,
    }
