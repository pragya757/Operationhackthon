"""
preprocess.py — Production Audio Preprocessing
================================================
Pipeline:
  1. Load .flac / .wav (any sample rate, any channels)
     Uses soundfile as primary loader (torchaudio ≥2.11 requires
     TorchCodec for FLAC on macOS and is unsupported here).
  2. Downmix to mono
  3. Resample to 16 kHz
  4. Pad (zero) or crop to exactly 10 seconds
  5. Compute Log-Mel Spectrogram (128 mel bins)
  6. Convert amplitude → dB scale
  7. Normalize (mean=0, std=1)
  8. Save to .pt cache — never regenerate if cache exists
"""

import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

from . import config


class SpectrogramPreprocessor:
    """
    Converts a raw audio file into a normalized Log-Mel spectrogram tensor.

    Output tensor shape: (n_mels, time_frames)
    For model input add channel dim → (1, n_mels, time_frames)

    Cache strategy:
      - Cache key = basename without extension → <basename>.pt
      - If cache file exists and is a valid tensor → return immediately
      - Cache is always written after successful computation
    """

    def __init__(
        self,
        sample_rate: int = None,
        max_duration: float = None,
        n_fft: int = None,
        hop_length: int = None,
        n_mels: int = None,
        power: float = None,
        normalize: bool = None,
        cache_dir: str = None,
    ):
        self.sample_rate = sample_rate  if sample_rate  is not None else config.SAMPLE_RATE
        self.max_duration = max_duration if max_duration is not None else config.MAX_DURATION
        self.n_fft       = n_fft        if n_fft        is not None else config.N_FFT
        self.hop_length  = hop_length   if hop_length   is not None else config.HOP_LENGTH
        self.n_mels      = n_mels       if n_mels       is not None else config.N_MELS
        self.power       = power        if power        is not None else config.POWER
        self.normalize   = normalize    if normalize    is not None else config.NORMALIZE_SPEC
        self.cache_dir   = cache_dir    if cache_dir    is not None else config.CACHE_DIR

        # Derived
        self.max_samples = int(self.sample_rate * self.max_duration)

        # Ensure cache directory exists
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Build transform pipeline once (reused for every file)
        self._mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
        )
        self._amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, filepath: str) -> str | None:
        if not self.cache_dir:
            return None
        basename = os.path.splitext(os.path.basename(filepath))[0]
        return os.path.join(self.cache_dir, f"{basename}.pt")

    def _load_from_cache(self, cache_file: str) -> torch.Tensor | None:
        """Return cached tensor if the file exists and is valid, else None."""
        if not os.path.exists(cache_file):
            return None
        try:
            tensor = torch.load(cache_file, weights_only=True)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                return tensor
        except Exception:
            # Corrupted cache — delete and recompute
            try:
                os.remove(cache_file)
            except OSError:
                pass
        return None

    def _write_cache(self, cache_file: str, tensor: torch.Tensor):
        """Atomically save tensor to cache (write to tmp then rename)."""
        tmp = cache_file + ".tmp"
        try:
            torch.save(tensor, tmp)
            os.replace(tmp, cache_file)
        except Exception:
            # Non-fatal: training can proceed without cache
            try:
                os.remove(tmp)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Core pipeline stages
    # ------------------------------------------------------------------

    def load_and_align(self, filepath: str) -> torch.Tensor:
        """
        Load audio with soundfile (works on macOS for FLAC without TorchCodec),
        downmix to mono, resample via torchaudio transforms, and enforce exact duration.

        Returns:
            Tensor of shape (max_samples,) at self.sample_rate
        """
        # ── Load with soundfile (float32, channels-last) ──────────────
        data, sr = sf.read(filepath, dtype="float32", always_2d=True)
        # data shape: (samples, channels)
        waveform = torch.from_numpy(data.T)   # → (channels, samples)

        # 1. Downmix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, samples)

        # 2. Resample if needed
        if sr != self.sample_rate:
            waveform = T.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        # 3. Remove channel dim → 1-D waveform
        waveform = waveform.squeeze(0)
        n = waveform.shape[0]

        # 4. Pad (zero) or crop
        if n < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - n))
        elif n > self.max_samples:
            waveform = waveform[: self.max_samples]

        return waveform  # shape: (max_samples,)

    def compute_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized Log-Mel spectrogram from aligned waveform.

        Returns:
            Tensor of shape (n_mels, time_frames)
        """
        # MelSpectrogram expects (channel, time)
        x = waveform.unsqueeze(0)             # (1, T)
        mel = self._mel_transform(x)          # (1, n_mels, time_frames)
        log_mel = self._amplitude_to_db(mel)  # (1, n_mels, time_frames)
        log_mel = log_mel.squeeze(0)          # (n_mels, time_frames)

        # Per-spectrogram mean/std normalisation
        if self.normalize:
            mean = log_mel.mean()
            std  = log_mel.std()
            log_mel = (log_mel - mean) / (std + 1e-6)

        return log_mel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(self, filepath: str, use_cache: bool = True) -> torch.Tensor:
        """
        Full pipeline: load → align → log-mel → normalize → (cache).

        Args:
            filepath   : Absolute path to audio file.
            use_cache  : If True, check cache before processing and save after.

        Returns:
            Tensor of shape (n_mels, time_frames).

        Raises:
            FileNotFoundError : If the audio file does not exist.
            RuntimeError      : If torchaudio cannot decode the file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        cache_file = self._cache_path(filepath) if use_cache else None

        # --- Cache hit ---
        if cache_file:
            cached = self._load_from_cache(cache_file)
            if cached is not None:
                return cached

        # --- Compute ---
        waveform = self.load_and_align(filepath)
        log_mel  = self.compute_log_mel(waveform)

        # --- Cache write ---
        if cache_file:
            self._write_cache(cache_file, log_mel)

        return log_mel
