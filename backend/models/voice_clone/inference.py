"""
inference.py — Single File / Batch Inference
=============================================
Supports:
  - Single audio file prediction
  - Directory of audio files (batch prediction)
  - Returns human-readable label and confidence probability

Usage:
  python -m spectrogram_cnn_training.inference --audio path/to/audio.flac
  python -m spectrogram_cnn_training.inference --dir path/to/audio_dir/
"""

import os
import glob
import torch
import torch.nn.functional as F

# Package-relative imports — work when imported as models.voice_clone.inference
from . import config
from .model import SpectrogramCNN
from .preprocess import SpectrogramPreprocessor
from .utils import get_logger

# Human-readable output labels
LABEL_NAMES = {0: "Human (Bonafide)", 1: "Synthetic (Spoof)"}
SUPPORTED_FORMATS = [".flac", ".wav", ".mp3", ".ogg"]


class Predictor:
    """
    End-to-end inference interface for the Spectrogram CNN.
    
    Loads the model once at initialization and exposes:
      - predict_file(path) → single prediction dict
      - predict_dir(dir_path) → list of prediction dicts
    """

    def __init__(self, checkpoint_path: str = None, device: str = None):
        """
        Args:
            checkpoint_path (str): Path to a .pth checkpoint. Defaults to best_model.pth.
            device (str): Target device. Auto-detects if None.
        """
        self.device = device or config.DEVICE
        self.preprocessor = SpectrogramPreprocessor()
        self.logger = get_logger("Predictor")

        # Load model
        checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH
        self.model = self._load_model(checkpoint_path)
        self.logger.info(f"Predictor ready on device: {self.device}")

    def _load_model(self, checkpoint_path: str) -> SpectrogramCNN:
        """Loads and returns a trained SpectrogramCNN in eval mode."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Please train the model first using train.py."
            )
        model = SpectrogramCNN(in_channels=1, num_classes=2)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # Support both full checkpoint dict (saved by train.py) and bare state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError(f"Unexpected checkpoint type: {type(checkpoint).__name__}")
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_file(self, audio_path: str) -> dict:
        """
        Runs inference on a single audio file.
        
        Args:
            audio_path (str): Absolute path to an audio file.
        
        Returns:
            dict: {
                'file': str,
                'label': str,
                'label_id': int,
                'confidence': float,
                'prob_human': float,
                'prob_synthetic': float
            }
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Preprocess (with cache)
        spectrogram = self.preprocessor.process_file(audio_path)

        # Add batch and channel dimensions: [n_mels, time] → [1, 1, n_mels, time]
        x = spectrogram.unsqueeze(0).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred_id = int(torch.argmax(probs).item())

        return {
            "file": os.path.basename(audio_path),
            "label": LABEL_NAMES[pred_id],
            "label_id": pred_id,
            "confidence": round(float(probs[pred_id].item()), 6),
            "prob_human": round(float(probs[0].item()), 6),
            "prob_synthetic": round(float(probs[1].item()), 6)
        }

    def predict_dir(self, dir_path: str, save_csv: bool = True) -> list:
        """
        Runs batch inference on all audio files in a directory.
        
        Args:
            dir_path (str): Path to directory containing audio files.
            save_csv (bool): If True, saves predictions to outputs/sample_predictions.csv.
        
        Returns:
            List of prediction dicts (one per file).
        """
        # Lazy-import pandas and tqdm — only needed for batch/CLI mode
        try:
            import pandas as pd
        except ImportError:
            pd = None
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a valid directory: {dir_path}")

        # Collect all supported audio files
        audio_files = []
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))

        if not audio_files:
            self.logger.warning(f"No supported audio files found in: {dir_path}")
            return []

        self.logger.info(f"Running inference on {len(audio_files)} files...")
        results = []

        iterator = (_tqdm(audio_files, desc="Inferring", unit="file")
                    if _tqdm else audio_files)
        for fpath in iterator:
            try:
                result = self.predict_file(fpath)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"  Skipping {os.path.basename(fpath)}: {e}")

        if save_csv and results and pd is not None:
            df = pd.DataFrame(results)
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            out_path = config.SAMPLE_PREDICTIONS
            df.to_csv(out_path, index=False)
            self.logger.info(f"Predictions saved to: {out_path}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spectrogram CNN Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=str, help="Path to a single audio file")
    group.add_argument("--dir", type=str, help="Path to a directory of audio files")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pth checkpoint (default: best_model.pth)")
    args = parser.parse_args()

    predictor = Predictor(checkpoint_path=args.checkpoint)

    if args.audio:
        result = predictor.predict_file(args.audio)
        print(f"\n=== Prediction ===")
        print(f"  File       : {result['file']}")
        print(f"  Label      : {result['label']}")
        print(f"  Confidence : {result['confidence']:.4f}")
        print(f"  P(Human)   : {result['prob_human']:.4f}")
        print(f"  P(Synthetic): {result['prob_synthetic']:.4f}")
    elif args.dir:
        results = predictor.predict_dir(args.dir)
        print(f"\n=== Batch Inference Summary ===")
        print(f"  Total files    : {len(results)}")
        human_count = sum(1 for r in results if r['label_id'] == 0)
        spoof_count = sum(1 for r in results if r['label_id'] == 1)
        print(f"  Human (Bonafide): {human_count}")
        print(f"  Synthetic (Spoof): {spoof_count}")
