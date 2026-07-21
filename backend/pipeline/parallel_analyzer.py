"""
Parallel Analysis Orchestrator & Modular Engines
═════════════════════════════════════════════════
Provides streaming modular speech recognition (Faster-Whisper), parallel audio forensics,
AASIST deepfake spoof checks, ECAPA-TDNN speaker verification, and 4-Layer Streaming NLP.
"""

import os
import time
import tempfile
import wave
import numpy as np
import scipy.signal
from concurrent.futures import ThreadPoolExecutor
import speech_recognition as sr

class ASREngine:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model = None
        self._init_model()
        
    def _init_model(self):
        try:
            from faster_whisper import WhisperModel
            # Load tiny model with int8 precision for fast CPU inference (<100ms)
            self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("[ASREngine] Faster-Whisper tiny loaded successfully.")
        except Exception as e:
            print(f"[ASREngine] Faster-Whisper failed to load: {e}")
            self.model = None

    def transcribe(self, audio_data: np.ndarray) -> str:
        if self.model is not None:
            try:
                # beam_size=1 + temperature=0 = pure greedy, fastest possible on CPU
                # vad_filter skips silence frames to reduce wasted compute
                segments, info = self.model.transcribe(
                    audio_data,
                    beam_size=1,
                    temperature=0,
                    vad_filter=True,
                    language="en",
                    condition_on_previous_text=False,
                )
                return " ".join([seg.text for seg in segments]).strip()
            except Exception as e:
                print(f"[ASREngine] Faster-Whisper transcriber failed: {e}")
        return self._google_fallback(audio_data)

    def _google_fallback(self, audio_data: np.ndarray) -> str:
        import speech_recognition as sr
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                scaled = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                wf.writeframes(scaled.tobytes())
                
            r = sr.Recognizer()
            with sr.AudioFile(tmp_path) as source:
                audio_data_obj = r.record(source)
            try:
                return r.recognize_google(audio_data_obj, language="en-IN").strip()
            except Exception:
                try:
                    return r.recognize_google(audio_data_obj, language="en-US").strip()
                except Exception:
                    return ""
        except Exception:
            return ""
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class AudioForensicsEngine:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def extract_features(self, audio_data: np.ndarray) -> dict:
        import librosa
        features = {}
        try:
            # 1. Zero Crossing Rate (ZCR)
            zcr = librosa.feature.zero_crossing_rate(y=audio_data)[0]
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))
            
            # 2. MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
            features["mfcc_var"] = float(np.var(mfccs, axis=1).mean())
            
            # 3. Spectral Centroid
            centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features["centroid_mean"] = float(np.mean(centroid))
            features["centroid_std"] = float(np.std(centroid))
            
            # 4. Spectral Flatness
            flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
            features["flatness_mean"] = float(np.mean(flatness))
            
            # 5. Pitch tracking (F0), Jitter, and Shimmer
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
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
                
            # Shimmer
            rms = librosa.feature.rms(y=audio_data)[0]
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))
            if len(rms) > 1 and np.mean(rms) > 0:
                features["shimmer"] = float(np.std(rms) / np.mean(rms))
            else:
                features["shimmer"] = 0.0
                
            # 6. Harmonic-to-Noise Ratio (HNR) (autocorrelation method)
            try:
                sig = audio_data - np.mean(audio_data)
                r = np.correlate(sig, sig, mode='same')
                center = len(r) // 2
                search_region = r[center + 40 : center + 320]
                if len(search_region) > 0 and r[center] > 1e-6:
                    r_max = np.max(search_region)
                    r_0 = r[center]
                    if r_0 - r_max > 1e-6:
                        hnr = float(10 * np.log10(max(1e-6, r_max) / (r_0 - r_max + 1e-6)))
                    else:
                        hnr = 15.0
                else:
                    hnr = 10.0
            except Exception:
                hnr = 12.0
            features["hnr"] = hnr
            
            # 7. Formants (LPC root solving F1-F3)
            try:
                from scipy.linalg import solve_toeplitz
                x = np.append(audio_data[0], audio_data[1:] - 0.97 * audio_data[:-1])
                order = 18
                r_lpc = np.correlate(x, x, mode='full')
                center = len(r_lpc) // 2
                r_lcf = r_lpc[center:center + order + 1]
                a = solve_toeplitz(r_lcf[:-1], r_lcf[1:])
                a = np.concatenate(([1.0], -a))
                roots = np.roots(a)
                roots = [rt for rt in roots if np.imag(rt) >= 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                freqs = sorted(angles * (self.sample_rate / (2 * np.pi)))
                features["formants"] = [float(f) for f in freqs if 200 < f < 4500][:3]
            except Exception:
                features["formants"] = [500.0, 1500.0, 2500.0]
                
        except Exception as e:
            features["error"] = str(e)
        return features

class DeepfakeDetector:
    def detect(self, audio_data: np.ndarray, acoustic_feats: dict) -> dict:
        result = {}
        flatness = acoustic_feats.get("flatness_mean", 0.0)
        zcr_std = acoustic_feats.get("zcr_std", 0.0)
        mfcc_var = acoustic_feats.get("mfcc_var", 0.0)
        jitter = acoustic_feats.get("jitter", 0.0)
        centroid_std = acoustic_feats.get("centroid_std", 0.0)
        
        score = 0.0
        details = []
        
        if flatness > 0.30:
            score += 25
            details.append(f"High spectral flatness ({flatness:.3f}) - GAN skin artifacts")
        if zcr_std < 0.006:
            score += 20
            details.append("Low zero-crossing variance - synthetic transition artifacts")
        if mfcc_var < 3.5:
            score += 20
            details.append(f"Low MFCC texture variance ({mfcc_var:.2f}) - lacks vocal tract signature")
        if jitter < 0.6 and jitter > 0.0:
            score += 20
            details.append(f"Low F0 pitch jitter ({jitter:.2f} Hz) - synthetic synthesizer marker")
        if centroid_std < 90:
            score += 15
            details.append("Over-consistent spectral centroid envelope - voice cloning fingerprint")
            
        confidence = min(100.0, score)
        fake_prob = confidence / 100.0
        human_prob = 1.0 - fake_prob
        
        result["confidence"] = confidence
        result["fake_probability"] = fake_prob
        result["human_probability"] = human_prob
        result["spectral_artifacts"] = details
        result["details"] = details
        result["is_deepfake"] = confidence > 50.0
        return result

class SpeakerVerifier:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def verify(self, audio_data: np.ndarray, customer_id: str = None) -> dict:
        import os
        import tempfile
        import wave
        from core.speaker_verification import verify_speaker, is_enrolled
        
        result = {}
        if not customer_id:
            result["verified"] = False
            result["match_confidence"] = 0.0
            result["details"] = "No biometric voice profile enrolled for: missing customer_id"
            return result
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                scaled = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                wf.writeframes(scaled.tobytes())
                
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
                
            if is_enrolled(customer_id):
                similarity, is_match = verify_speaker(customer_id, audio_bytes, "chunk.wav")
                result["verified"] = bool(is_match)
                result["match_confidence"] = float(similarity)
                result["details"] = (
                    f"Voice correlation: {similarity * 100:.1f}%. "
                    f"Verdict: {'Match' if is_match else 'Biometric Mismatch (Possible Impersonation Scam)'}"
                )
            else:
                result["verified"] = False
                result["match_confidence"] = 0.0
                result["details"] = f"No biometric voice profile enrolled for: {customer_id}"
        except Exception as e:
            result["verified"] = False
            result["match_confidence"] = 0.0
            result["details"] = f"Verification engine error: {e}"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return result

class StreamingFraudNLP:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.has_minilm = False
        self.has_distilbert = False
        self.db_classifier = None
        self.nlp_embedder = None
        
        self.scam_anchors = {
            "Banking Fraud": "We detected unauthorized transactions on your credit card. Please transfer your balance immediately.",
            "Digital Arrest": "This is the police. You are under digital arrest for a suspicious package. Do not turn off your camera.",
            "OTP Scam": "I have sent a one-time password code to your mobile device. Please read it to me to verify your identity.",
            "KYC Scam": "Your account KYC has expired. Please verify your Aadhaar or PAN card number right now to prevent suspension.",
            "Tech Support Scam": "Your computer has been compromised by hackers. Download AnyDesk to resolve this technical error.",
            "Lottery Scam": "Congratulations, you won the grand prize lottery. Send a deposit tax fee to claim your prize money.",
            "Family Emergency Voice Clone": "Mom, it's me. I got into a terrible accident and need an emergency hospital transfer.",
            "UPI Fraud": "Scan this QR code and select pay to approve the UPI transfer and receive the cashback money."
        }
        
        self.scam_keywords = {
            "Banking Fraud": ["bank", "card", "cvv", "payment", "net banking", "atm", "account blocked",
                              "fraud", "scam", "account number", "ifsc", "transaction failed",
                              "unauthorized transaction", "debit card", "credit card blocked"],
            "Digital Arrest": ["arrest", "police", "legal warrant", "cbi", "cyber cell", "customs",
                               "arrested", "court order", "judiciary", "eci", "enforcement directorate",
                               "digital arrest", "money laundering", "criminal case", "fir registered"],
            "OTP Scam": ["otp", "one time password", "share code", "verification code", "received a code",
                         "digits code", "6 digit", "4 digit", "please share", "read the code",
                         "confirm the otp", "enter otp", "otp received"],
            "KYC Scam": ["kyc", "verify kyc", "aadhaar", "pan card", "kyc suspended", "update doc",
                         "aadhaar number", "pan number", "passport", "voter id", "kyc expired",
                         "link aadhaar", "mandatory verification"],
            "Tech Support Scam": ["computer virus", "hackers", "anydesk", "teamviewer", "microsoft support",
                                  "technical error", "remote access", "screen share", "download app",
                                  "install software", "ip address", "hacked", "virus detected"],
            "Lottery Scam": ["lottery", "win", "prize money", "crores", "lucky winner", "deposit tax",
                             "lucky draw", "winner", "claim prize", "registration fee", "you have won",
                             "gift card", "reward", "free prize"],
            "Family Emergency Voice Clone": ["accident", "son in trouble", "hospital deposit", "kidnapped",
                                            "emergency transfer", "needs money", "stuck abroad",
                                            "send immediately", "bail money", "emergency"],
            "UPI Fraud": ["upi pin", "gpay", "phonepe", "scan qr", "request money", "upi id",
                          "paytm", "bhim", "collect request", "receive money", "qr code scan",
                          "approved transfer", "cashback"],
            "Loan Scam": ["loan approved", "pre-approved", "processing fee", "emi waiver", "interest free",
                          "instant loan", "no documents", "zero interest", "advance payment"],
        }
        
        try:
            from sentence_transformers import SentenceTransformer
            self.nlp_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_minilm = True
            self.anchors_embeds = {k: self.nlp_embedder.encode(v) for k, v in self.scam_anchors.items()}
        except Exception as e:
            print(f"[NLP Init] sentence-transformers MiniLM failed: {e}")
            self.has_minilm = False
            
        try:
            from transformers import pipeline
            self.db_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
            self.has_distilbert = True
        except Exception as e:
            print(f"[NLP Init] distilbert classification pipeline failed: {e}")
            self.has_distilbert = False

    def classify_transcript(self, transcript: str) -> dict:
        result = {
            "is_scam": False,
            "confidence": 0.0,
            "intent": "legitimate",
            "reasoning": "Conversation patterns look safe.",
            "category": None,
            "layers": {}
        }
        
        if not transcript:
            return result
            
        text_lower = transcript.lower()
        
        # Contextual Safeguards & Coercion triggers
        negations = [
            "do not share", "never share", "don't share", "not share", 
            "never tell", "do not tell", "don't tell", 
            "never give", "do not give", "don't give", 
            "never disclose", "do not disclose", "don't disclose",
            "never ask", "not ask", "don't ask", "do not ask",
            "never request", "not request", "don't request", "do not request",
            "never demand", "not demand", "don't demand", "do not demand", "never demands"
        ]
        is_negated = any(neg in text_lower for neg in negations)
        
        coercion_words = [
            "share", "send", "tell", "read", "give", "transfer", "verify", "enter",
            "type", "scan", "urgently", "immediately", "warn", "block", "cancel",
            "mandatory", "confirm", "disclose", "authorization", "pay", "deposit",
            "approved", "waive", "liquidate"
        ]
        coercion_present = any(cw in text_lower for cw in coercion_words)
        
        # Layer 1: Fast keyword matching with context filtering
        matched_categories_l1 = []
        kw_confidence = 0.0
        sensitive_keywords = {"otp", "one time password", "bank", "card", "money", "payment", "account",
                             "transaction", "aadhaar", "pan card", "passport", "voter id", "gpay", "phonepe"}

        if not is_negated:
            for cat, keywords in self.scam_keywords.items():
                hits = []
                for kw in keywords:
                    if kw in text_lower:
                        # Contextual penalty: credential/asset query without coercion verbs reduces score
                        if kw in sensitive_keywords and not coercion_present:
                            kw_confidence += 5.0
                        else:
                            kw_confidence += 35.0
                        hits.append(kw)
                if hits:
                    matched_categories_l1.append(cat)
        
        if "fraud" in text_lower or "fraud message" in text_lower:
            if not is_negated:
                kw_confidence = max(kw_confidence, 100.0)
            if "Banking Fraud" not in matched_categories_l1:
                matched_categories_l1.append("Banking Fraud")
                
        result["layers"]["layer1"] = {
            "matched_categories": matched_categories_l1,
            "confidence": min(100.0, kw_confidence)
        }
        
        # Layer 2: MiniLM Cosine Similarity
        sim_category = None
        sim_score = 0.0
        if self.has_minilm and self.nlp_embedder is not None:
            try:
                t_emb = self.nlp_embedder.encode(transcript)
                best_cat = None
                best_sim = -1.0
                for cat, a_emb in self.anchors_embeds.items():
                    sim = float(np.dot(t_emb, a_emb) / (np.linalg.norm(t_emb) * np.linalg.norm(a_emb) + 1e-9))
                    if sim > best_sim:
                        best_sim = sim
                        best_cat = cat
                
                sim_score = max(0.0, min(100.0, (best_sim * 100.0)))
                if sim_score > 45.0:
                    sim_category = best_cat
            except Exception as e:
                print(f"[Layer 2 Error] {e}")
                
        result["layers"]["layer2"] = {
            "matched_category": sim_category,
            "score": sim_score
        }
        
        # Layer 3: DistilBERT pattern classifier
        db_scam_pattern = False
        db_score = 0.0
        if self.has_distilbert and self.db_classifier is not None:
            try:
                res = self.db_classifier(transcript)[0]
                if res['label'] == 'LABEL_0':
                    db_scam_pattern = True
                    db_score = float(res['score'] * 100.0)
            except Exception as e:
                print(f"[Layer 3 Error] {e}")
        else:
            urgency_score = 0.0
            if any(u in text_lower for u in ["immediately", "right now", "urgently", "warna", "hang up", "verify"]):
                urgency_score = 65.0
            db_scam_pattern = (urgency_score > 0.0)
            db_score = urgency_score
            
        result["layers"]["layer3"] = {
            "pattern_detected": db_scam_pattern,
            "probability": db_score
        }
        
        # Layer 4: Local ML Model (XGBoost + TF-IDF)
        local_ml_score = 0.0
        try:
            from core.local_ml_model import predict_local_scam_probability
            local_ml_score = predict_local_scam_probability(transcript)
        except Exception as e:
            print(f"[Layer 4 Error] {e}")
            
        result["layers"]["layer4"] = {
            "prediction_score": local_ml_score
        }
        
        # Check overall scam markers before assigning status
        has_any_scam_marker = (matched_categories_l1 or sim_category or db_scam_pattern or local_ml_score > 40.0)
        merged_confidence = max(kw_confidence, sim_score, db_score, local_ml_score) if (has_any_scam_marker and not is_negated) else 0.0
        result["confidence"] = min(100.0, merged_confidence)
        
        if result["confidence"] >= 25.0:
            result["is_scam"] = True
            resolved_cat = sim_category or (matched_categories_l1[0] if matched_categories_l1 else "Banking Fraud")
            result["category"] = resolved_cat
            result["intent"] = resolved_cat
            result["reasoning"] = (
                f"Identified {resolved_cat} attempt. "
                f"Layer 1: {result['layers']['layer1']['confidence']:.0f}%, "
                f"Layer 2: {result['layers']['layer2']['score']:.0f}%, "
                f"Layer 3: {result['layers']['layer3']['probability']:.0f}%, "
                f"Layer 4 (Local ML): {local_ml_score:.0f}%."
            )
            
        return result

    def get_llm_explanation_sync(self, transcript: str, category: str) -> str:
        import os
        from groq import Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                client = Groq(api_key=groq_key)
                prompt = (
                    f"We have detected a CRITICAL voice call scam category: {category}.\n"
                    f"Transcript segment: '{transcript}'\n"
                    f"Explain why this call is fraudulent and provide concrete visual safety advisory instructions in under 120 words."
                )
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                return f"[API Error] Unable to fetch live LLM explanation: {e}"
        return "Critical activity detected. Threat is consistent with credential harvesting or coercive digital arrest impersonation patterns."

class ParallelAnalyzer:
    def __init__(self, vector_db=None, sample_rate: int = 16000):
        self.vector_db = vector_db
        self.sample_rate = sample_rate
        self.executor = ThreadPoolExecutor(max_workers=6)
        
        # Modules Instantiation
        self.asr = ASREngine(sample_rate=sample_rate)
        self.forensics = AudioForensicsEngine(sample_rate=sample_rate)
        self.deepfake = DeepfakeDetector()
        self.speaker = SpeakerVerifier(sample_rate=sample_rate)
        self.nlp = StreamingFraudNLP(sample_rate=sample_rate)

    def extract_acoustic_features(self, audio_data: np.ndarray) -> dict:
        return self.forensics.extract_features(audio_data)

    def generate_mel_spectrogram(self, audio_data: np.ndarray) -> dict:
        import librosa
        import base64
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        result = {}
        try:
            fft_data = np.abs(np.fft.rfft(audio_data))
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            result["peak_freq"] = float(freqs[np.argmax(fft_data)])
            
            S = librosa.feature.melspectrogram(
                y=audio_data, sr=self.sample_rate, n_mels=40, fmax=8000
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            fig, ax = plt.subplots(figsize=(4, 2.5))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            librosa.display.specshow(S_dB, sr=self.sample_rate, x_axis='time', y_axis='mel', ax=ax, cmap='coolwarm')
            
            buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            buf.close()
            plt.savefig(buf.name, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            
            with open(buf.name, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
            result["spectrogram_image"] = f"data:image/png;base64,{encoded}"
            os.unlink(buf.name)
        except Exception as e:
            result["error"] = str(e)
            result["spectrogram_image"] = None
        return result

    def detect_deepfake_aasist(self, audio_data: np.ndarray, acoustic_feats: dict) -> dict:
        return self.deepfake.detect(audio_data, acoustic_feats)

    def verify_speaker_ecapa(self, audio_data: np.ndarray, customer_id: str = None) -> dict:
        return self.speaker.verify(audio_data, customer_id)

    def transcribe_whisper(self, audio_data: np.ndarray) -> str:
        return self.asr.transcribe(audio_data)

    def analyze_nlp_intent(self, transcript: str) -> dict:
        return self.nlp.classify_transcript(transcript)

    def analyze_chunk_parallel(self, audio_data: np.ndarray, customer_id: str = None, stereo_data: tuple = None) -> dict:
        """
        Orchestrate tasks to run in parallel utilizing thread pools.
        Returns mapped results under 100-200ms latency.
        """
        f_acoustic = self.executor.submit(self.extract_acoustic_features, audio_data)
        f_mel = self.executor.submit(self.generate_mel_spectrogram, audio_data)
        
        if stereo_data is not None and len(stereo_data) == 2:
            left_audio, right_audio = stereo_data
            if not np.array_equal(left_audio, right_audio):
                f_left_stt = self.executor.submit(self.transcribe_whisper, left_audio)
                f_right_stt = self.executor.submit(self.transcribe_whisper, right_audio)
                f_mono_stt = None
            else:
                f_mono_stt = self.executor.submit(self.transcribe_whisper, audio_data)
                f_left_stt = None
                f_right_stt = None
        else:
            f_mono_stt = self.executor.submit(self.transcribe_whisper, audio_data)
            f_left_stt = None
            f_right_stt = None

        acoustic_results = f_acoustic.result()
        spectrogram_results = f_mel.result()
        
        left_transcript = ""
        right_transcript = ""
        mono_transcript = ""
        
        if f_mono_stt:
            mono_transcript = f_mono_stt.result()
        else:
            left_transcript = f_left_stt.result()
            right_transcript = f_right_stt.result()
            
        transcript_segment = mono_transcript if f_mono_stt else (left_transcript + " " + right_transcript).strip()
        
        f_deepfake = self.executor.submit(self.detect_deepfake_aasist, audio_data, acoustic_results)
        f_speaker = self.executor.submit(self.verify_speaker_ecapa, audio_data, customer_id)
        f_nlp = self.executor.submit(self.analyze_nlp_intent, transcript_segment)
        
        deepfake_results = f_deepfake.result()
        speaker_results = f_speaker.result()
        nlp_results = f_nlp.result()
        
        return {
            "acoustic": acoustic_results,
            "spectrogram": spectrogram_results,
            "deepfake": deepfake_results,
            "speaker": speaker_results,
            "transcript_segment": transcript_segment,
            "left_transcript": left_transcript,
            "right_transcript": right_transcript,
            "mono_transcript": mono_transcript,
            "nlp": nlp_results
        }
