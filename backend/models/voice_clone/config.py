import os
import torch

# =====================================================================
# DEVICE CONFIGURATION
# =====================================================================
# Automatically detect the most optimal device (Apple Silicon, NVIDIA, or CPU)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# =====================================================================
# AUDIO PREPROCESSING CONFIGURATION
# =====================================================================
SAMPLE_RATE = 16000      # Target sample rate (16 kHz mono)
MAX_DURATION = 10.0      # Maximum duration in seconds (10 seconds)
N_FFT = 1024             # FFT size
HOP_LENGTH = 512         # Hop length
N_MELS = 128             # Number of Mel bins
POWER = 2.0              # Power for Spectrogram (2.0 for Power Spectrogram)
NORMALIZE_SPEC = True    # Enable mean-std normalization of spectrograms

# =====================================================================
# TRAINING HYPERPARAMETERS
# =====================================================================
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_VAL = 5.0
SEED = 42

# =====================================================================
# PATHS CONFIGURATION
# =====================================================================
# Root workspace directory for the ASVspoof2019 dataset
DATASET_ROOT = "/Users/pragya/Downloads/LA"

# ASVspoof2019 Specific subdirectories
TRAIN_AUDIO_DIR = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_train", "flac")
DEV_AUDIO_DIR = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_dev", "flac")
EVAL_AUDIO_DIR = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_eval", "flac")

# Protocol files
TRAIN_PROTOCOL = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTOCOL = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")
EVAL_PROTOCOL = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")

# Project Outputs and Cache
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_DIR = os.path.dirname(PROJECT_ROOT)
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")  # Preprocessed spectrogram cache

# Dynamic checkpoint and output file mappings
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "spectrogram_cnn.pth")
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")

# Output plot and metrics filenames
TRAINING_LOSS_PLOT = os.path.join(OUTPUT_DIR, "training_loss.png")
VALIDATION_LOSS_PLOT = os.path.join(OUTPUT_DIR, "validation_loss.png")
ACCURACY_CURVE_PLOT = os.path.join(OUTPUT_DIR, "accuracy_curve.png")
CONFUSION_MATRIX_PLOT = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
ROC_CURVE_PLOT = os.path.join(OUTPUT_DIR, "roc_curve.png")
CLASSIFICATION_REPORT = os.path.join(OUTPUT_DIR, "classification_report.txt")
EVALUATION_METRICS = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
SAMPLE_PREDICTIONS = os.path.join(OUTPUT_DIR, "sample_predictions.csv")

# Create all project output directories if they don't exist
for directory in [CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)
