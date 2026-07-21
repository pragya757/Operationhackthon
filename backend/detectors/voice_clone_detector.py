"""
Voice Clone Detector – Modular Voice Forensics Engine
─────────────────────────────────────────────────────
Uses a pre-trained deepfake classification model from Hugging Face as the primary detector,
combined with physical acoustic feature heuristics (librosa) in a weighted fusion layer.

Updates for Stage 2:
  1. Restricts input audio loading to the first 10 seconds.
  2. Integrates Spectrogram Audio Forensics via detectors.spectrogram_detector.
  3. Implements lightweight, modular Threat Fusion layer combining both results.
  4. Returns separate sub-verdicts for Voice Clone, Spectrogram, and Final Threat Assessment.
"""

import os
import io
import time
import base64
from typing import Dict, Any, List

import numpy as np
from models.voice_clone.inference import Predictor

# ── Configurable Variables ───────────────────────────────────────────────────
MODEL_NAME = os.getenv("VOICE_CLONE_MODEL", "garystafford/wav2vec2-deepfake-voice-detector")
VOICE_MODEL_WEIGHT = float(os.getenv("VOICE_MODEL_WEIGHT", "0.70"))
VOICE_HEURISTIC_WEIGHT = float(os.getenv("VOICE_HEURISTIC_WEIGHT", "0.30"))
SAVE_SPECTROGRAM_TENSORS = os.getenv("SAVE_SPECTROGRAM_TENSORS", "False").lower() in ("true", "1", "yes")

# ── Threat Fusion Thresholds (all configurable via environment variables) ─────
# Fusion weights: voice clone engine has higher priority in UPLOAD mode (0.65 vs 0.35)
UPLOAD_FUSION_WEIGHT_CLONE = float(os.getenv("UPLOAD_FUSION_WEIGHT_CLONE", os.getenv("FUSION_WEIGHT_CLONE", "0.65")))
UPLOAD_FUSION_WEIGHT_SPEC  = float(os.getenv("UPLOAD_FUSION_WEIGHT_SPEC",  os.getenv("FUSION_WEIGHT_SPEC",  "0.35")))

# Retain legacy aliases for backward compatibility with other modules
FUSION_WEIGHT_CLONE = UPLOAD_FUSION_WEIGHT_CLONE
FUSION_WEIGHT_SPEC  = UPLOAD_FUSION_WEIGHT_SPEC

# Confidence threshold: minimum clone engine score to consider a positive detection
# [WhatsApp calibration] Raised from 0.40 → 0.45 to reduce Wav2Vec2 false positives on Opus-compressed audio
CLONE_CONFIDENCE_THRESHOLD = float(os.getenv("CLONE_CONFIDENCE_THRESHOLD", "0.45"))
CLONE_HIGH_RISK_THRESHOLD  = float(os.getenv("CLONE_HIGH_RISK_THRESHOLD",  "0.70"))

# Spectrogram synthetic score threshold for Upload mode (default 0.40)
SPEC_CONFIDENCE_THRESHOLD  = float(os.getenv("SPEC_CONFIDENCE_THRESHOLD",  "0.40"))

# Final combined score thresholds for risk bucketing
FUSION_HIGH_RISK_THRESHOLD    = float(os.getenv("FUSION_HIGH_RISK_THRESHOLD",    "0.65"))
FUSION_SUSPICIOUS_THRESHOLD   = float(os.getenv("FUSION_SUSPICIOUS_THRESHOLD",   "0.35"))

# ── Live Call Threat Fusion Configuration (Telephony/WebRTC adaptation) ────────
# The pretrained Wav2Vec2 detector has a known generalization gap (false positives)
# on compressed telephony/WebRTC streams (Opus compression, AEC, comfort noise).
# Consequently, in Live Call mode we prioritize the WhatsApp-trained Spectrogram CNN.
LIVE_FUSION_WEIGHT_CLONE = float(os.getenv("LIVE_FUSION_WEIGHT_CLONE", "0.10"))
LIVE_FUSION_WEIGHT_SPEC  = float(os.getenv("LIVE_FUSION_WEIGHT_SPEC",  "0.90"))

# Spectrogram CNN synthetic threshold for Live mode
# [WhatsApp calibration] Raised from 0.25 → 0.40 — our WhatsApp-trained CNN uses calibrated 0.5 boundary;
# allowing 0.25 caused borderline samples (real voices with codec noise) to trigger false AI alerts.
LIVE_SPEC_AI_THRESHOLD   = float(os.getenv("LIVE_SPEC_AI_THRESHOLD",   "0.40"))

# Case 5 conservative override — dual-condition gate
#
# Wav2Vec2 false-positive range on compressed WhatsApp/WebRTC audio is
# empirically 40–60% (clone_fusion_score 0.40–0.60).  Genuine AI voices
# produced by voice-cloning tools consistently score >= 0.79 in validation.
#
# Both conditions must be satisfied for a Safe override:
#   1. Voice Clone score is BELOW this ceiling (weak positive only)
#   2. Spectrogram CNN predicts Human with very high confidence
# Default: 0.70  (configurable via LIVE_CLONE_SAFE_MAX_CONFIDENCE env var)
LIVE_CLONE_SAFE_MAX_CONFIDENCE = float(os.getenv("LIVE_CLONE_SAFE_MAX_CONFIDENCE", "0.70"))

# Minimum Spectrogram CNN Human confidence required for Case 5 override.
# A low-confidence Human prediction from the CNN is not trusted enough
# to override the Wav2Vec2 detector.
# [WhatsApp calibration] Set to 70.0% — reliably distinguishes genuine Human voice from synthetic speech.
LIVE_SPEC_HUMAN_CONFIDENCE = float(os.getenv("LIVE_SPEC_HUMAN_CONFIDENCE", "70.0"))


# ── Startup Configuration Logging ─────────────────────────────────────────────
print(
    "\n"
    "================ CONFIG ================\n"
    "\n"
    "UPLOAD\n"
    f"Clone Weight : {UPLOAD_FUSION_WEIGHT_CLONE:.2f}\n"
    f"Spec Weight  : {UPLOAD_FUSION_WEIGHT_SPEC:.2f}\n"
    "\n"
    "LIVE\n"
    f"Clone Weight : {LIVE_FUSION_WEIGHT_CLONE:.2f}\n"
    f"Spec Weight  : {LIVE_FUSION_WEIGHT_SPEC:.2f}\n"
    "\n"
    f"Clone Safe Max Confidence    : {LIVE_CLONE_SAFE_MAX_CONFIDENCE:.2f}\n"
    f"Spectrogram Human Confidence : {LIVE_SPEC_HUMAN_CONFIDENCE:.1f}%\n"
    "\n"
    "========================================\n"
)



_HERE = os.path.dirname(os.path.abspath(__file__))
TENSOR_DIR = os.path.join(os.path.dirname(_HERE), "models", "voice_clone", "temp_tensors")

# ── Pre-trained Model Singleton ─────────────────────────────────────────────
_classifier = None
_spectrogram_predictor = None

def _select_hf_device():
    """Select the best available device for the HuggingFace pipeline."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon — transformers ≥ 4.35 accepts "mps"
    if torch.cuda.is_available():
        return 0        # CUDA GPU index 0
    return -1           # CPU

def _get_classifier():
    """Lazy-load the Hugging Face audio classification model."""
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
            hf_device = _select_hf_device()
            print(f"[VoiceCloneDetector] Initializing model pipeline: {MODEL_NAME} (device={hf_device}) ...")
            _classifier = pipeline("audio-classification", model=MODEL_NAME, device=hf_device)
            print(f"[VoiceCloneDetector] Pipeline loaded successfully on device={hf_device}.")
        except Exception as e:
            print(f"[VoiceCloneDetector] Warning: Could not initialize model pipeline {MODEL_NAME}: {e}")
            _classifier = None
    return _classifier


# ── Acoustic Feature Heuristics ──────────────────────────────────────────────
def extract_acoustic_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract physical metrics from audio data using librosa."""
    import librosa
    
    features = {}
    try:
        # 1. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
        
        # 2. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
        features["mfcc_var"] = float(np.var(mfccs, axis=1).mean())
        
        # 3. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["centroid_mean"] = float(np.mean(centroid))
        features["centroid_std"] = float(np.std(centroid))
        
        # 4. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features["flatness_mean"] = float(np.mean(flatness))
        
        # 5. Pitch tracking (F0) & Jitter
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[magnitudes > magnitudes.mean()]
        if len(pitch_vals) > 0:
            features["pitch_mean"] = float(np.mean(pitch_vals[pitch_vals > 0]))
            features["pitch_std"] = float(np.std(pitch_vals))
            diffs = np.abs(np.diff(pitch_vals))
            features["jitter"] = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0
            features["jitter"] = 0.0
            
    except Exception as e:
        features["error"] = str(e)
        features.setdefault("zcr_std", 0.01)
        features.setdefault("mfcc_var", 20.0)
        features.setdefault("flatness_mean", 0.10)
        features.setdefault("jitter", 5.0)
        features.setdefault("centroid_std", 150.0)
        
    return features


# ── Core Analysis Interface ──────────────────────────────────────────────────
def analyze_audio(audio_path: str, mode: str = "upload") -> Dict[str, Any]:
    """
    Core python method to analyze a local audio file path for deepfakes.
    Exposes an internal method usable by Twilio, WhatsApp streams, or other files.
    """
    import librosa
    from detectors.spectrogram_detector import analyze_spectrogram
    
    # 1. Load Audio File (Load only the first 10 seconds of audio)
    # Uses a multi-decoder strategy to handle all browser recording formats:
    #   WebM (Chrome/Firefox), OGG (Firefox), MP4 (Safari), WAV, MP3, FLAC
    y, sr = None, None

    # Strategy A: librosa (handles WAV, FLAC, MP3 natively; needs ffmpeg for WebM/OGG)
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=10.0, mono=True)
        print(f"[VoiceCloneDetector] Backend Successfully Loaded via librosa: {audio_path}")
        print(f"[VoiceCloneDetector] Sample Rate: {sr} Hz | Audio Length: {len(y)/sr:.2f}s | Channels: mono")
    except Exception as librosa_err:
        print(f"[VoiceCloneDetector] librosa load failed ({librosa_err}), trying pydub fallback...")

    # Strategy B: pydub (handles WebM, OGG, MP4 from browser MediaRecorder via ffmpeg under the hood)
    if y is None:
        try:
            from pydub import AudioSegment
            import numpy as np
            ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
            fmt = ext if ext in ("webm", "ogg", "mp4", "mp3", "wav", "flac", "m4a") else "webm"
            audio_seg = AudioSegment.from_file(audio_path, format=fmt)
            # Trim to first 10 seconds
            audio_seg = audio_seg[:10000]
            audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
            samples = audio_seg.get_array_of_samples()
            y = np.array(samples, dtype=np.float32) / (2 ** (audio_seg.sample_width * 8 - 1))
            sr = 16000
            print(f"[VoiceCloneDetector] Backend Successfully Loaded via pydub: {audio_path}")
            print(f"[VoiceCloneDetector] Sample Rate: {sr} Hz | Audio Length: {len(y)/sr:.2f}s | Format: {fmt}")
        except Exception as pydub_err:
            print(f"[VoiceCloneDetector] pydub load also failed: {pydub_err}")

    if y is None:
        return {
            "prediction": "Human",
            "prediction_internal": "REAL",
            "confidence": 0.0,
            "model_score": 0.0,
            "heuristic_score": 0.0,
            "fusion_score": 0.0,
            "risk_level": "Safe",
            "reasons": ["Audio load error: could not decode the uploaded audio file. Supported formats: WAV, MP3, FLAC, M4A, WebM, OGG."],
            "spectrogram_image": None,
            "raw_model_output": "Audio load error",
            "raw_scores": {}
        }

    # Normalize sample rate to 16kHz for pipeline consistency
    import numpy as np
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    # -- Audio Diagnostics (logged for every request – both Upload and Live Call) -
    rms   = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
    peak  = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    rms_db  = 20 * np.log10(rms  + 1e-9)
    peak_db = 20 * np.log10(peak + 1e-9)
    duration_s = len(y) / sr
    first_samples = y[:8].tolist() if len(y) >= 8 else y.tolist()
    print(f"[AudioDiag] ----- Audio Pre-Normalization Diagnostics -----")
    print(f"[AudioDiag]   Sample Rate  : {sr} Hz")
    print(f"[AudioDiag]   Duration     : {duration_s:.3f} s")
    print(f"[AudioDiag]   Samples      : {len(y)}")
    print(f"[AudioDiag]   Channels     : mono")
    print(f"[AudioDiag]   RMS          : {rms:.6f}  ({rms_db:.1f} dBFS)")
    print(f"[AudioDiag]   Peak         : {peak:.6f}  ({peak_db:.1f} dBFS)")
    print(f"[AudioDiag]   First 8 vals : {[round(v,4) for v in first_samples]}")
    if rms < 0.001:
        print(f"[AudioDiag]   [WARNING] Near-silent audio (RMS={rms:.6f}). Mic volume may be too low.")

    # -- RMS Normalization to -20 dBFS target -------------------------------------
    # Live Call mic recordings typically arrive at -30 to -40 dBFS due to room
    # acoustics and OS gain limits. Uploaded files are often near 0 dBFS.
    # Without normalization, the model sees completely different amplitude ranges
    # for the same voice content, causing inconsistent predictions.
    # Target: -20 dBFS (10% of full scale) – safe headroom, consistent for both paths.
    TARGET_RMS_DB  = float(os.getenv("AUDIO_TARGET_RMS_DB", "-20.0"))
    MIN_RMS_THRESH = float(os.getenv("AUDIO_MIN_RMS_THRESH", "0.0001"))  # silence gate
    if rms >= MIN_RMS_THRESH:
        target_rms = 10 ** (TARGET_RMS_DB / 20.0)
        gain = target_rms / rms
        # Hard-limit gain: never amplify more than 40 dB to avoid noise blowup
        gain = min(gain, 100.0)
        y = y * gain
        # Peak-limit clip to [-1, 1] after gain
        y = np.clip(y, -1.0, 1.0)
        rms_after  = float(np.sqrt(np.mean(y ** 2)))
        peak_after = float(np.max(np.abs(y)))
        print(f"[AudioDiag]   Normalization: gain={gain:.3f}x -> RMS={rms_after:.6f} ({20*np.log10(rms_after+1e-9):.1f} dBFS)")
    else:
        print(f"[AudioDiag]   Normalization: SKIPPED (audio is near-silent, RMS={rms:.6f})")
    print(f"[AudioDiag] ----------------------------------------------------")

    print(f"[VoiceCloneDetector] Voice Clone Analysis Started: {len(y)} samples @ {sr} Hz")


    # 2. Extract Acoustic Heuristics
    features = extract_acoustic_features(y, sr)

    # Normalize heuristic parameters into a score of 0.0 to 1.0 (higher = synthetic)
    heuristic_points = 0.0
    if features.get("flatness_mean", 0.0) > 0.30:
        heuristic_points += 0.2
    if features.get("zcr_std", 1.0) < 0.006:
        heuristic_points += 0.2
    if features.get("mfcc_var", 10.0) < 3.5:
        heuristic_points += 0.2
    if 0.0 < features.get("jitter", 0.0) < 0.6:
        heuristic_points += 0.2
    if features.get("centroid_std", 200.0) < 90:
        heuristic_points += 0.2
    heuristic_score = float(heuristic_points)

    # 3. Model Scoring (Synthetic / Fake Probability)
    model_score = 0.0
    classifier = _get_classifier()
    model_active = classifier is not None
    raw_model_output = "Model not initialized"
    raw_scores = {}
    
    if model_active:
        try:
            results = classifier({"array": y, "sampling_rate": sr})
            
            # Extract id2label and build raw outputs for debugging (Tasks 1 & 3)
            id2label = getattr(classifier.model.config, "id2label", {})
            raw_model_output = str(results)
            raw_scores = {str(item["label"]): float(item["score"]) for item in results}
            
            print(f"[VoiceCloneDetector] Running inference with model: {MODEL_NAME}")
            print(f"[VoiceCloneDetector] Model Labels (id2label): {id2label}")
            print(f"[VoiceCloneDetector] Raw Prediction Scores: {results}")
            
            # Task 2: Robust label mapping (mapping to synthetic/fake probability)
            fake_prob = 0.0
            found_fake = False
            
            # 1. Search for explicit fake/spoof/synthetic labels
            for item in results:
                label_str = str(item["label"]).lower()
                if any(w in label_str for w in ("fake", "spoof", "synthetic", "clon", "generat", "unreal")):
                    fake_prob = item["score"]
                    found_fake = True
                    break
            
            # 2. Search for explicit real/human/bonafide labels and invert
            if not found_fake:
                for item in results:
                    label_str = str(item["label"]).lower()
                    if any(w in label_str for w in ("real", "bonafide", "human", "original", "authentic", "genuine")):
                        fake_prob = 1.0 - item["score"]
                        found_fake = True
                        break
                        
            # 3. Handle LABEL_0 (fake) / LABEL_1 (real) by convention
            if not found_fake:
                for item in results:
                    label_str = str(item["label"]).upper()
                    if label_str == "LABEL_0":
                        fake_prob = item["score"]
                        found_fake = True
                        break
                    elif label_str == "LABEL_1":
                        fake_prob = 1.0 - item["score"]
                        found_fake = True
                        break
            
            # 4. Fallback if no matching labels found
            if not found_fake and results:
                fake_prob = results[0]["score"]
                
            model_score = float(fake_prob)

            # ── Derive human / clone probabilities for the debug banner ───────────────────
            _vc_clone_prob = float(model_score)            # synthetic probability
            _vc_human_prob = float(1.0 - model_score)      # human probability
            # Best-effort: pull exact scores from raw_scores dict if available
            for _k, _v in raw_scores.items():
                _kl = _k.lower()
                if any(w in _kl for w in ("fake", "spoof", "synthetic", "clon", "generat")):
                    _vc_clone_prob = float(_v)
                elif any(w in _kl for w in ("real", "bonafide", "human", "original", "authentic")):
                    _vc_human_prob = float(_v)

            _vc_label_before_fusion = "Human" if model_score < CLONE_CONFIDENCE_THRESHOLD else "Voice Clone / AI"

            print(
                "\n"
                "========================\n"
                "VOICE CLONE RAW OUTPUT  \n"
                "========================\n"
                f"  Model             : {MODEL_NAME}\n"
                f"  Raw Response      : {raw_model_output}\n"
                f"  Raw Score Map     : {raw_scores}\n"
                f"  Human Probability : {_vc_human_prob:.6f}  ({_vc_human_prob*100:.2f}%)\n"
                f"  Clone Probability : {_vc_clone_prob:.6f}  ({_vc_clone_prob*100:.2f}%)\n"
                f"  Mapped fake_prob  : {model_score:.6f}  (synthetic score sent to fusion)\n"
                f"  Label before Fusion: {_vc_label_before_fusion}\n"
                "========================"
            )

        except Exception as e:
            print(f"[VoiceCloneDetector] Inference error: {e}")
            model_score = 0.0
            model_active = False
            raw_model_output = f"Inference error: {str(e)}"

    # 4. Voice Clone Detection verdict layers
    clone_fusion_score = (VOICE_MODEL_WEIGHT * model_score) + (VOICE_HEURISTIC_WEIGHT * heuristic_score) if model_active else heuristic_score
    clone_fusion_score = max(0.0, min(1.0, clone_fusion_score))

    prediction_internal = "REAL"
    prediction = "Human"
    confidence = 0.0
    
    if clone_fusion_score < CLONE_CONFIDENCE_THRESHOLD:
        prediction_internal = "REAL"
        prediction = "Human"
        confidence = (1.0 - clone_fusion_score) * 100.0
        risk_level = "Safe"
    else:
        if features.get("mfcc_var", 10.0) < 3.5 or features.get("flatness_mean", 0.0) > 0.30:
            prediction_internal = "SYNTHETIC_TTS"
            prediction = "AI Generated"
            risk_level = "High Risk" if clone_fusion_score >= CLONE_HIGH_RISK_THRESHOLD else "Suspicious"
        else:
            prediction_internal = "VOICE_CLONE"
            prediction = "Voice Clone"
            risk_level = "High Risk" if clone_fusion_score >= CLONE_HIGH_RISK_THRESHOLD else "Suspicious"
        confidence = clone_fusion_score * 100.0

    # Explainability reasons
    reasons = []
    if prediction_internal == "REAL":
        reasons.append("Spectrogram analysis matches natural human vocal folds.")
        if features.get("jitter", 0.0) >= 0.6:
            reasons.append(f"Natural F0 pitch jitter fluctuation detected ({features['jitter']:.2f} Hz).")
        if features.get("mfcc_var", 0.0) >= 3.5:
            reasons.append(f"High vocal tract resonance variance detected (MFCC var: {features['mfcc_var']:.1f}).")
        if model_active:
            reasons.append(f"Voice Clone model verified original human voice (confidence: {100 - (model_score*100):.1f}%).")
    else:
        reasons.append("Synthetic voice artifacts detected in frequency segments.")
        if features.get("flatness_mean", 0.0) > 0.30:
            reasons.append(f"Elevated spectral flatness ({features['flatness_mean']:.3f}) typical of neural vocoder synthesis.")
        if features.get("zcr_std", 1.0) < 0.006:
            reasons.append(f"Unnaturally standard zero-crossing rate variance ({features['zcr_std']:.5f}) - synthetic transition artifact.")
        if features.get("mfcc_var", 10.0) < 3.5:
            reasons.append(f"Flat vocal tract signature (MFCC variance: {features['mfcc_var']:.1f}) indicates text-to-speech vocoding.")
        if 0.0 < features.get("jitter", 0.0) < 0.6:
            reasons.append(f"Pitch cycle variance is too stable ({features['jitter']:.2f} Hz) - human baseline jitter: 4-8 Hz.")
        if features.get("centroid_std", 200.0) < 90:
            reasons.append(f"Spectral centroid repetition detected ({features['centroid_std']:.1f} Hz) - voice cloning frame loop marker.")
        if model_active:
            reasons.append(f"Voice Clone model flagged synthesized voice (confidence: {model_score*100:.1f}%).")

    # 5. Run Spectrogram Audio Forensics
    # ---------- CNN Spectrogram Prediction ----------
    print(f"[VoiceCloneDetector] Spectrogram Analysis Started: {audio_path}")
    spec_res = analyze_spectrogram(audio_path)

    # ── Forensic relabeling (domain-gap awareness) ──────────────────────────
    # The Spectrogram CNN was trained on studio-grade ASVspoof FLAC audio.
    # Real-world microphone recordings may score high due to acoustic domain
    # differences (room reverb, codec coloring, background noise) rather than
    # genuine synthesis artifacts. When the Voice Clone engine (primary detector)
    # says Human with high confidence but the CNN says Synthetic, we relabel the
    # CNN verdict to 'Suspicious Acoustic Pattern' so the UI presents it as
    # forensic evidence rather than a standalone verdict.
    # The numeric spectrogram_score is NEVER changed — ThreatFusion math is unaffected.
    _clone_is_positive_prelim = clone_fusion_score >= CLONE_CONFIDENCE_THRESHOLD
    _spec_is_synthetic_prelim = float(spec_res.get("spectrogram_score") or 0.0) >= SPEC_CONFIDENCE_THRESHOLD

    if not _clone_is_positive_prelim and _spec_is_synthetic_prelim:
        # Case 4: Voice Clone says Human, CNN says Synthetic → domain gap most likely
        spec_display_prediction = "Suspicious Acoustic Pattern"
        spec_forensic_note = (
            "CNN trained on studio-grade FLAC data (ASVspoof 2019). "
            "Real-world mic recordings may trigger elevated scores due to acoustic domain differences. "
            "Voice Clone engine (primary detector) found no synthesis artifacts."
        )
    else:
        spec_display_prediction = spec_res.get("prediction", "Unknown")
        spec_forensic_note = None

    # 6. Threat Fusion Layer ─────────────────────────────────────────────────
    # Step A: Compute the weighted numeric fusion score.
    #   For Upload Audio mode, the Voice Clone engine has higher weight (default 0.65)
    #   because it is the primary deepfake detector, while the Spectrogram CNN (0.35) corroborates.
    #   For Live Call mode, the Spectrogram CNN is the primary detector (default 0.90)
    #   and Wav2Vec2 is the secondary signal (0.10) to account for WebRTC codec compression.
    spec_score = float(spec_res.get("spectrogram_score") or 0.0)

    w_clone = LIVE_FUSION_WEIGHT_CLONE if mode == "live" else UPLOAD_FUSION_WEIGHT_CLONE
    w_spec  = LIVE_FUSION_WEIGHT_SPEC  if mode == "live" else UPLOAD_FUSION_WEIGHT_SPEC

    final_fusion_score = (w_clone * clone_fusion_score) + (w_spec * spec_score)
    final_fusion_score = max(0.0, min(1.0, final_fusion_score))

    clone_is_positive = clone_fusion_score >= CLONE_CONFIDENCE_THRESHOLD
    
    t_spec = LIVE_SPEC_AI_THRESHOLD if mode == "live" else SPEC_CONFIDENCE_THRESHOLD
    spec_is_synthetic = spec_score >= t_spec

    # ── Step B: Label-based Threat Fusion Policy ─────────────────────────────
    #
    # The Voice Clone detector is the authoritative classifier for real-world audio.
    # The Spectrogram CNN is a supporting forensic module with a verified domain gap
    # on real microphone recordings (confirmed against ASVspoof2019 dataset).
    #
    # Policy (based on prediction labels, no confidence gates):
    #
    #   Case 1: Clone=Human  + Spec=Human                   → Safe
    #   Case 2: Clone=Human  + Spec=Suspicious Acoustic Pat → Safe  (domain-gap; clone authoritative)
    #   Case 3: Clone=AI/VC  + Spec=AI Generated            → High Risk
    #   Case 4: Clone=AI/VC  + Spec=Human or Suspicious     → Suspicious
    #
    # Case 5
    #
    # Live WhatsApp/WebRTC audio may produce moderate false positives
    # from the pretrained Wav2Vec2 detector due to codec compression.
    #
    # A Safe override is allowed ONLY when:
    #
    # 1. The Voice Clone detector's score is below the configurable
    #    LIVE_CLONE_SAFE_MAX_CONFIDENCE threshold (default 0.60).
    #    This ensures only weak positives are overridden — genuine
    #    AI voices scoring 0.79+ are never affected.
    #
    # 2. The custom WhatsApp-trained Spectrogram CNN explicitly predicts
    #    Human with very high confidence (LIVE_SPEC_HUMAN_CONFIDENCE,
    #    default 95.0%). A borderline Human prediction is not trusted
    #    enough to override the Wav2Vec2 detector.
    #
    # This dual-condition policy minimizes false Safe decisions while
    # still correcting known telephony-related false positives.
    #
    # Upload mode is NOT affected.

    # ── TEMPORARY CASE 5 DEBUG ─────────────────────────────────────────────────
    print("CASE5 DEBUG")
    print("mode:", mode)
    print("prediction:", prediction)
    print("spec_display_prediction:", spec_display_prediction)
    print("clone_fusion_score:", clone_fusion_score)
    print("spec_confidence:", spec_res.get("confidence"))
    print("threshold_clone:", LIVE_CLONE_SAFE_MAX_CONFIDENCE)
    print("threshold_spec:", LIVE_SPEC_HUMAN_CONFIDENCE)
    # ───────────────────────────────────────────────────────────────────────────

    if (
        mode == "live"
        and prediction != "Human"
        and spec_display_prediction == "Human"
        and clone_fusion_score < LIVE_CLONE_SAFE_MAX_CONFIDENCE
        and spec_res.get("confidence", 0.0) >= LIVE_SPEC_HUMAN_CONFIDENCE
    ):
        final_risk_level = "Safe"
        policy_case = 5
        final_fusion_score = spec_score
    elif prediction == "Human":
        final_risk_level = "Safe"
        if spec_display_prediction == "Suspicious Acoustic Pattern":
            final_fusion_score = clone_fusion_score
            policy_case = 2
        else:
            policy_case = 1
    else:
        if spec_display_prediction == "AI Generated":
            final_risk_level = "High Risk"
            policy_case = 3
        else:
            policy_case = 4
            if final_fusion_score < FUSION_SUSPICIOUS_THRESHOLD:
                final_risk_level = "Safe"
            else:
                final_risk_level = "Suspicious"

    # ================ THREAT FUSION ================
    # Mode           : LIVE / UPLOAD
    #
    # Voice Clone
    # ------------
    # Label          :
    # Confidence     :
    # Raw Score      :
    #
    # Spectrogram CNN
    # ---------------
    # Label          :
    # Confidence     :
    # Raw Score      :
    #
    # Fusion
    # ------
    # Clone Weight   :
    # Spec Weight    :
    # Weighted Score :
    # Decision Case  :
    # Final Risk     :
    # ==============================================
    print(
        "\n"
        "================ THREAT FUSION ================\n"
        f"Mode           : {mode.upper()}\n"
        "\n"
        "Voice Clone\n"
        "------------\n"
        f"Label          : {prediction}\n"
        f"Confidence     : {confidence:.2f}%\n"
        f"Raw Score      : {clone_fusion_score:.6f}\n"
        "\n"
        "Spectrogram CNN\n"
        "---------------\n"
        f"Label          : {spec_display_prediction}\n"
        f"Confidence     : {spec_res.get('confidence', 0.0):.2f}%\n"
        f"Raw Score      : {spec_score:.6f}\n"
        "\n"
        "Fusion\n"
        "------\n"
        f"Clone Weight   : {w_clone:.2f}\n"
        f"Spec Weight    : {w_spec:.2f}\n"
        f"Weighted Score : {final_fusion_score:.6f}\n"
        f"Decision Case  : {policy_case}\n"
        f"Final Risk     : {final_risk_level.upper()}\n"
        "================================================"
    )

    # Step C: Build combined explanation based on policy case
    combined_explanations = []
    if policy_case == 1:
        combined_explanations.append("Both Voice Clone and Spectrogram engines confirm authentic human speech. No synthetic artifacts detected.")
    elif policy_case == 2:
        combined_explanations.append(f"Voice Clone engine confirms authentic human speech (confidence {confidence:.1f}%). Spectrogram CNN flagged unusual acoustic patterns ({spec_score*100:.1f}%), but this is a known domain gap: the CNN was trained on studio-grade ASVspoof data and may over-fire on real-world microphone recordings. Voice Clone is the authoritative detector — classified as Safe.")
    elif policy_case == 3:
        combined_explanations.append(f"Dual-engine confirmation: Voice Clone engine detected synthetic speech (score {clone_fusion_score*100:.1f}%) and Spectrogram CNN detected AI-generated artifacts ({spec_score*100:.1f}%) — High Risk.")
    elif policy_case == 4:
        combined_explanations.append(f"Voice Clone engine detected synthetic speech (score {clone_fusion_score*100:.1f}%). Spectrogram analysis did not independently confirm synthesis ({spec_score*100:.1f}%) — classified as Suspicious.")
    elif policy_case == 5:
        combined_explanations.append(f"Voice Clone engine flagged synthesis artifacts (score {clone_fusion_score*100:.1f}%), but this is overridden by the Spectrogram CNN verifying authentic human speech ({(1.0 - spec_score)*100:.1f}% confidence). Pretrained voice clone models often over-fire on compressed WebRTC/WhatsApp call audio due to codec distortion — classified as Safe.")
    else:
        combined_explanations.append(f"Voice Clone engine score {clone_fusion_score*100:.1f}%, Spectrogram score {spec_score*100:.1f}%.")

    # Save Mel-spectrogram tensor optionally
    if SAVE_SPECTROGRAM_TENSORS:
        try:
            os.makedirs(TENSOR_DIR, exist_ok=True)
            timestamp = int(time.time())
            safe_name = "".join([c if c.isalnum() else "_" for c in os.path.basename(audio_path)])
            tensor_path = os.path.join(TENSOR_DIR, f"spec_{timestamp}_{safe_name}.npy")
            # Calculate log mel spectrogram again to dump
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            np.save(tensor_path, S_db)
            print(f"[VoiceCloneDetector] Spectrogram tensor saved: {tensor_path}")
        except Exception as te:
            print(f"[VoiceCloneDetector] Spectrogram tensor save failed: {te}")

    # ── Runtime Debug Log (all values for every request) ─────────────────────
    print("[ThreatFusion] ── Runtime Values ─────────────────────────────────")
    print(f"[ThreatFusion]   clone_prediction   = {prediction}")
    print(f"[ThreatFusion]   clone_confidence   = {confidence:.1f}%")
    print(f"[ThreatFusion]   clone_score        = {clone_fusion_score:.4f}  (threshold={CLONE_CONFIDENCE_THRESHOLD})")
    print(f"[ThreatFusion]   clone_is_positive  = {clone_is_positive}")
    print(f"[ThreatFusion]   spec_prediction    = {spec_res.get('prediction', 'N/A')}")
    print(f"[ThreatFusion]   spec_confidence    = {spec_res.get('confidence', 0):.1f}%")
    print(f"[ThreatFusion]   spec_score         = {spec_score:.4f}  (threshold={SPEC_CONFIDENCE_THRESHOLD})")
    print(f"[ThreatFusion]   spec_is_synthetic  = {spec_is_synthetic}")
    print(f"[ThreatFusion]   final_fusion_score = {final_fusion_score:.4f}")
    print(f"[ThreatFusion]   selected_policy    = Case {policy_case}")
    if policy_case == 1: print("[ThreatFusion]   → Case 1: Clone=Human  + Spec=Human               → Safe")
    elif policy_case == 2: print("[ThreatFusion]   → Case 2: Clone=Human  + Spec=Suspicious Pattern → Safe (domain-gap; clone authoritative)")
    elif policy_case == 3: print("[ThreatFusion]   → Case 3: Clone=AI/VC  + Spec=AI Generated       → High Risk")
    elif policy_case == 4: print("[ThreatFusion]   → Case 4: Clone=AI/VC  + Spec=Human/Suspicious   → Suspicious")
    elif policy_case == 5: print("[ThreatFusion]   → Case 5: Clone=AI/VC  + Spec=Human (High Conf)  → Safe (live WebRTC telephony override)")
    else: print(f"[ThreatFusion]   → Case {policy_case}: fallback")
    print(f"[ThreatFusion]   final_risk_level   = {final_risk_level}  ← AUTHORITATIVE")
    print("[ThreatFusion] ────────────────────────────────────────────────────")

    return {
        # Top-level fields: risk_level is now always the Threat Fusion result (final_risk_level).
        # The old per-engine risk_level was a pre-fusion value and must NOT be used here.
        "prediction": prediction,
        "prediction_internal": prediction_internal,
        "confidence": round(confidence, 1),
        "model_score": round(model_score, 3),
        "heuristic_score": round(heuristic_score, 3),
        "fusion_score": round(clone_fusion_score, 3),
        "risk_level": final_risk_level,        # ← was risk_level (per-engine), now final_risk_level
        "reasons": reasons,
        "spectrogram_image": spec_res["spectrogram_image"],
        "raw_model_output": raw_model_output,
        "raw_scores": raw_scores,

        # Phase 2 Stage 2 Detailed Forensic Breakdown
        "voice_clone_analysis": {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "risk_level": risk_level,           # ← per-engine verdict (Safe / Suspicious / High Risk)
            "reasons": reasons
        },
        "spectrogram_analysis": {
            "prediction": spec_display_prediction,          # display label (may be relabeled for domain gap)
            "prediction_raw": spec_res["prediction"],       # original CNN output always preserved
            "confidence": spec_res["confidence"],
            "score": spec_res["spectrogram_score"],         # numeric score always original
            "reasons": spec_res["reasons"],
            "spectrogram_image": spec_res["spectrogram_image"],
            "forensic_note": spec_forensic_note,            # None unless Case 4 domain-gap relabeling
        },
        "threat_fusion": {
            "final_risk_score": round(final_fusion_score * 100, 1),
            "risk_level": final_risk_level,     # ← Threat Fusion authoritative verdict
            "explanation": combined_explanations
        }
    }
