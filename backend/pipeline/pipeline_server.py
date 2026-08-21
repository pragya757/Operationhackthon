"""
Production Live Call Pipeline Manager
═════════════════════════════════════
Manages active live call ingestion, preprocessing, rolling buffers,
parallel analytics engines, and dynamic threat fusion mapping.
"""

import numpy as np
from pipeline.audio_preprocessor import AudioPreprocessor
from pipeline.sliding_window import SlidingWindowBuffer
from pipeline.parallel_analyzer import ParallelAnalyzer
from pipeline.threat_fusion import ThreatFusionEngine

def clean_overlap(prev_text: str, new_text: str) -> str:
    """Removes overlapping words from the start of new_text that match the end of prev_text."""
    if not prev_text:
        return new_text
    import re
    def normalize(w):
        return re.sub(r'[^\w]', '', w.lower())
        
    prev_words = prev_text.split()
    new_words = new_text.split()
    if not prev_words or not new_words:
        return new_text
        
    prev_normalized = [normalize(w) for w in prev_words if normalize(w)]
    new_normalized = [normalize(w) for w in new_words if normalize(w)]
    
    if not prev_normalized or not new_normalized:
        return new_text
        
    max_overlap = min(len(prev_normalized), len(new_normalized))
    best_overlap = 0
    for i in range(1, max_overlap + 1):
        if prev_normalized[-i:] == new_normalized[:i]:
            best_overlap = i
            
    if best_overlap > 0:
        matched_words_count = 0
        original_idx = 0
        while original_idx < len(new_words) and matched_words_count < best_overlap:
            if normalize(new_words[original_idx]):
                matched_words_count += 1
            original_idx += 1
        return " ".join(new_words[original_idx:])
        
    return new_text

class PipelineSession:
    def __init__(self, session_id: str, customer_id: str = None, sample_rate: int = 16000):
        self.session_id = session_id
        self.customer_id = customer_id
        self.sliding_buffer = SlidingWindowBuffer(sample_rate=sample_rate)
        # Left and Right channel buffers for stereo tracking/transcribing
        self.left_buffer = SlidingWindowBuffer(sample_rate=sample_rate)
        self.right_buffer = SlidingWindowBuffer(sample_rate=sample_rate)
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        
        # Accumulate transcript timeline for explains UI
        self.full_transcript = []
        self.risk_timeline = []
        self.prev_my_transcript = ""
        self.prev_caller_transcript = ""
        self.prev_mono_transcript = ""

class ProductionPipelineManager:
    def __init__(self, vector_db=None, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.active_sessions: dict[str, PipelineSession] = {}
        self.analyzer = ParallelAnalyzer(vector_db=vector_db, sample_rate=sample_rate)
        self.fusion_engine = ThreatFusionEngine()

    def get_or_create_session(self, session_id: str, customer_id: str = None) -> PipelineSession:
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = PipelineSession(session_id, customer_id, self.sample_rate)
        return self.active_sessions[session_id]

    def remove_session(self, session_id: str):
        self.active_sessions.pop(session_id, None)

    def process_pcm_chunk(self, session_id: str, raw_pcm_bytes: bytes, customer_id: str = None) -> list[dict]:
        """
        Ingests a raw PCM sound chunk.
        Runs: Buffering → 3s Window Extraction → Preprocessing → Parallel ML → Threat Fusion.
        Returns a list of updates generated for this input.
        """
        session = self.get_or_create_session(session_id, customer_id)
        
        import io
        import wave
        
        left_samples_float32 = None
        right_samples_float32 = None
        channels = 1
        
        try:
            if raw_pcm_bytes.startswith(b"RIFF") and b"WAVE" in raw_pcm_bytes[:20]:
                with wave.open(io.BytesIO(raw_pcm_bytes), "rb") as wf:
                    channels = wf.getnchannels()
                    n_frames = wf.getnframes()
                    sample_rate = wf.getframerate()
                    content = wf.readframes(n_frames)
                    samples_int16 = np.frombuffer(content, dtype=np.int16)
            else:
                samples_int16 = np.frombuffer(raw_pcm_bytes, dtype=np.int16)
                
            if len(samples_int16) > 0:
                if channels == 2 or (channels == 1 and len(samples_int16) % 2 == 0 and len(raw_pcm_bytes) > 200000):
                    if channels == 1:
                        channels = 2
                    samples_int16 = samples_int16[:(len(samples_int16) // 2) * 2].reshape(-1, 2)
                    left_int16 = samples_int16[:, 0]
                    right_int16 = samples_int16[:, 1]
                    
                    left_samples_float32 = left_int16.astype(np.float32) / 32768.0
                    right_samples_float32 = right_int16.astype(np.float32) / 32768.0
                    samples_float32 = (left_samples_float32 + right_samples_float32) / 2.0
                else:
                    samples_float32 = samples_int16.astype(np.float32) / 32768.0
                    left_samples_float32 = samples_float32
                    right_samples_float32 = samples_float32
        except Exception as e:
            print(f"[PCM Parse Exception] {e}")
            samples_float32 = np.zeros(0, dtype=np.float32)
            left_samples_float32 = np.zeros(0, dtype=np.float32)
            right_samples_float32 = np.zeros(0, dtype=np.float32)
            
        if len(samples_float32) == 0:
            return []
            
        session.sliding_buffer.append_samples(samples_float32)
        if hasattr(session, 'left_buffer'):
            session.left_buffer.append_samples(left_samples_float32)
            session.right_buffer.append_samples(right_samples_float32)
        else:
            session.left_buffer = SlidingWindowBuffer(sample_rate=self.sample_rate)
            session.right_buffer = SlidingWindowBuffer(sample_rate=self.sample_rate)
            session.left_buffer.append_samples(left_samples_float32)
            session.right_buffer.append_samples(right_samples_float32)
            
        updates = []
        
        while session.sliding_buffer.has_next_window():
            window_raw, start_t, end_t = session.sliding_buffer.get_next_window()
            left_window_raw, _, _ = session.left_buffer.get_next_window()
            right_window_raw, _, _ = session.right_buffer.get_next_window()
            
            cleaned_audio, has_speech = session.preprocessor.process(window_raw)
            
            if not has_speech:
                null_score = 0.0
                if len(session.risk_timeline) > 0:
                    null_score = session.risk_timeline[-1]
                    
                updates.append({
                    "type": "chunk_result",
                    "session_id": session_id,
                    "timestamp": round(end_t, 1),
                    "threat_score": null_score,
                    "verdict": "SAFE" if null_score < 40 else "HIGH RISK",
                    "deepfake_confidence": 0.0,
                    "scam_intent_confidence": 0.0,
                    "transcript_segment": "",
                    "explainable_reasons": ["[System] Silence/No voice activity detected."],
                    "risk_timeline": session.risk_timeline,
                    "alerts_triggered": False
                })
                continue
                
            cleaned_left, _ = session.preprocessor.process(left_window_raw)
            cleaned_right, _ = session.preprocessor.process(right_window_raw)
            
            # ── Step 1: Run ASR (Whisper) + parallel ML analysis together ─────
            analysis_dict = self.analyzer.analyze_chunk_parallel(
                cleaned_audio,
                session.customer_id,
                stereo_data=(cleaned_left, cleaned_right)
            )
            
            # ── Step 2: Build transcript segment from ASR output ──────────────
            seg_text = ""
            if analysis_dict.get("mono_transcript"):
                mono_text = analysis_dict["mono_transcript"]
                new_mono = clean_overlap(session.prev_mono_transcript, mono_text)
                session.prev_mono_transcript = mono_text
                seg_text = new_mono
            else:
                my_text = analysis_dict.get("left_transcript", "")
                caller_text = analysis_dict.get("right_transcript", "")
                new_my = clean_overlap(session.prev_my_transcript, my_text)
                new_caller = clean_overlap(session.prev_caller_transcript, caller_text)
                session.prev_my_transcript = my_text
                session.prev_caller_transcript = caller_text
                
                parts = []
                if new_my.strip():
                    parts.append(f"[You]: {new_my.strip()}")
                if new_caller.strip():
                    parts.append(f"[Person 1]: {new_caller.strip()}")
                seg_text = " ".join(parts)
                
            if seg_text:
                session.full_transcript.append(seg_text)
                
            analysis_dict["transcript_segment"] = seg_text

            # ── Step 3: Emit fast transcript packet immediately (no ML wait) ──
            if seg_text:
                updates.append({
                    "type": "transcript_ready",
                    "session_id": session_id,
                    "timestamp": round(end_t, 1),
                    "transcript_segment": seg_text,
                })
            
            # ── Step 4: Threat fusion (ML scoring) ───────────────────────────
            fusion_result = self.fusion_engine.fuse(analysis_dict)
            threat_score = fusion_result.get("threat_score", 0.0)
            
            session.risk_timeline.append(threat_score)
            
            update_packet = {
                "type": "chunk_result",
                "session_id": session_id,
                "timestamp": round(end_t, 1),
                "threat_score": threat_score,
                "verdict": fusion_result.get("verdict", "SAFE"),
                "severity": fusion_result.get("severity", "LOW"),
                "deepfake_confidence": fusion_result.get("deepfake_confidence", 0.0),
                "scam_intent_confidence": fusion_result.get("scam_intent_confidence", 0.0),
                "transcript_segment": seg_text,
                "full_transcript": " ".join(session.full_transcript),
                "spectrogram_image": analysis_dict.get("spectrogram", {}).get("spectrogram_image"),
                "explainable_reasons": fusion_result.get("explainable_reasons", []),
                "risk_timeline": session.risk_timeline,
                "alerts_triggered": fusion_result.get("alerts_triggered", False)
            }
            
            updates.append(update_packet)
            
        return updates

