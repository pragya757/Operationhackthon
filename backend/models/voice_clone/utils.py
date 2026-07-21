import os
import json
import random
import logging
import numpy as np
import torch

# pandas, matplotlib, and sklearn are only needed for training/evaluation utilities.
# Lazy-import them inside each function so this module loads without those packages.

def set_seed(seed: int = 42):
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # On MPS, manual_seed is sufficient. deterministic operations are set when supported.
    # We do not use torch.use_deterministic_algorithms(True) as some MPS operators don't support it yet.

def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a unified logger configured for console and file output.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def plot_loss(train_losses: list, val_losses: list, save_path: str):
    """
    Plots training and validation losses.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_accuracy(train_accs: list, val_accs: list, save_path: str):
    """
    Plots training and validation accuracy curves.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Accuracy', color='#2ca02c', linewidth=2)
    plt.plot(val_accs, label='Validation Accuracy', color='#d62728', linewidth=2)
    plt.title('Training and Validation Accuracy Curves', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, labels=None):
    """
    Computes and plots a stylized confusion matrix.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    if labels is None:
        labels = ['Human (Bona Fide)', 'Synthetic (Spoof)']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Label grid cells with counts
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12, fontweight='bold')

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: str):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and computes AUC.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#9467bd', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return roc_auc

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, labels=None):
    """
    Generates and saves a detailed text classification report.
    """
    from sklearn.metrics import classification_report
    if labels is None:
        labels = ['Human (Bona Fide)', 'Synthetic (Spoof)']
    
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    with open(save_path, 'w') as f:
        f.write(report)
    return report

def save_metrics_json(metrics: dict, save_path: str):
    """
    Saves validation/evaluation metrics into a JSON file.
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_predictions_csv(filenames: list, true_labels: list, pred_probs: list, pred_labels: list, save_path: str):
    """
    Saves model prediction outputs to a CSV for sample predictions and error analysis.
    """
    import pandas as pd
    df = pd.DataFrame({
        'audio_file': filenames,
        'true_label': true_labels,
        'predicted_probability': pred_probs,
        'predicted_label': pred_labels
    })
    df.to_csv(save_path, index=False)
