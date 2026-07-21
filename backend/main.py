"""
Fraud Shield AI – Unified Scam Detection API
═════════════════════════════════════════════
Best of Zora AI + Fraud Shield combined.
Multi-source input → Central Classifier → Risk Scoring → Human Feedback Loop
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uuid
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from middleware.shadow_guard import ShadowGuardMiddleware
from middleware.dlp_guard import DLPGuardMiddleware
from core.vector_db import VectorDB
from core.classifier import CentralClassifier
from core.feedback import FeedbackStore
from core.threat_score import ThreatScore
from detectors.text_detector import TextDetector
from detectors.credential_detector import CredentialDetector
from detectors.url_detector import URLDetector
from detectors.voice_detector import VoiceDetector
from detectors.file_detector import FileDetector
from detectors.email_detector import EmailDetector, IMAPFetcher
from detectors.video_detector import VideoDetector
from core.live_call import process_chunk, end_call, get_detection_stats
from core.twilio_stream import get_or_create_handler, get_handler, remove_handler
# Pre-import so matplotlib Agg backend + font cache load at startup, not on
# the first /analyze/voice request (eliminates ~1.7s first-call penalty).
from core import spectrogram_generator as _spec_warmup  # noqa: F401
from detectors import spectrogram_detector as _spec_detector_warmup  # noqa: F401
from pipeline.pipeline_server import ProductionPipelineManager


# ── Startup ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_db = VectorDB()
    app.state.vector_db.seed_known_scams()
    app.state.feedback = FeedbackStore()
    app.state.production_pipeline = ProductionPipelineManager(vector_db=app.state.vector_db)
    yield



# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Shield AI",
    description="Unified scam detection: SMS, Email, URL, Voice, Files — with deepfake detection, DLP, and human-in-the-loop feedback",
    version="2.0.0",
    lifespan=lifespan,
)

# Middleware stack (order matters - outermost first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ShadowGuardMiddleware)
app.add_middleware(DLPGuardMiddleware)

from routes.voice_clone import router as voice_clone_router
app.include_router(voice_clone_router, prefix="/api/voice-clone")


# ── Health ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "Fraud Shield AI",
        "version": "2.0.0",
        "status": "running",
        "detectors": ["text", "credential", "url", "voice", "file", "email"],
        "middleware": ["shadow_guard", "dlp_guard"],
        "features": ["deepfake_detection", "ai_generated_text_detection", "human_feedback", "imap_integration"],
    }



# ── Contacts Whitelist CRUD ─────────────────────────────────────────────────
import os as _os
import re as _re
from pathlib import Path as _Path

# Always resolves to backend/data/contacts.json regardless of working directory
CONTACTS_FILE = _Path(__file__).parent / "data" / "contacts.json"

def _load_contacts() -> list:
    try:
        if not CONTACTS_FILE.exists():
            CONTACTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            return []
        data = json.loads(CONTACTS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[CONTACTS] Load error: {e}")
        return []

def _save_contacts(contacts: list) -> None:
    try:
        CONTACTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONTACTS_FILE.write_text(json.dumps(contacts, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[CONTACTS] Saved {len(contacts)} contact(s) to {CONTACTS_FILE}")
    except Exception as e:
        print(f"[CONTACTS] Save error: {e}")

def _normalize_phone(phone: str) -> str:
    return _re.sub(r"[\s\-()]+", "", phone)

def _find_contact(contacts: list, phone: str):
    clean = _normalize_phone(phone)
    if not clean:
        return None
    for c in contacts:
        clean_c = _normalize_phone(c.get("phone", ""))
        if clean == clean_c or (len(clean) >= 7 and clean_c.endswith(clean)) or (len(clean_c) >= 7 and clean.endswith(clean_c)):
            return c
    return None


@app.get("/contacts")
async def get_contacts():
    """Return all saved contacts."""
    return {"contacts": _load_contacts()}


@app.post("/contacts")
async def add_contact(name: str = Form(...), phone: str = Form(...)):
    """Add or update a saved contact."""
    if not name.strip() or not phone.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="name and phone are required")
    contacts = _load_contacts()
    # Remove existing entry with same phone
    clean_new = _normalize_phone(phone)
    contacts = [c for c in contacts if _normalize_phone(c.get("phone", "")) != clean_new]
    contacts.append({"name": name.strip(), "phone": phone.strip()})
    _save_contacts(contacts)
    return {"ok": True, "contacts": contacts}


@app.delete("/contacts")
async def delete_contact(phone: str = Query(...)):
    """Remove a saved contact by phone number."""
    contacts = _load_contacts()
    clean = _normalize_phone(phone)
    contacts = [c for c in contacts if _normalize_phone(c.get("phone", "")) != clean]
    _save_contacts(contacts)
    return {"ok": True, "contacts": contacts}


# ── Individual Detectors ────────────────────────────────────────────────────

@app.post("/analyze/text")
async def analyze_text(
    message: str = Form(...),
    sender: str = Form(default="unknown"),
    channel: str = Form(default="sms"),
):
    """Analyze SMS or Email body for scam indicators."""
    detector = TextDetector(app.state.vector_db)
    text_result = detector.analyze(message, sender, channel)

    cred_detector = CredentialDetector()
    cred_result = cred_detector.analyze(message)

    combined = ThreatScore.combine({"text": text_result, "credential": cred_result})
    return {
        "analysis_id": str(uuid.uuid4())[:8],
        "components": {"text": text_result, "credential": cred_result},
        "combined": combined,
    }


@app.post("/analyze/url")
async def analyze_url(url: str = Form(...)):
    """Sandbox + SSL + WHOIS + heuristic URL analysis."""
    detector = URLDetector()
    result = await detector.analyze(url)
    result["analysis_id"] = str(uuid.uuid4())[:8]
    return result


@app.post("/analyze/voice")
async def analyze_voice(
    request: Request,
    audio: UploadFile = File(...),
    customer_id: Optional[str] = Form(default=None),
):
    """
    Acoustic + STT + deepfake voice analysis.

    Optional: pass `customer_id` (Form field) to also run speaker verification
    against the enrolled voice profile.  Response will include
    `speaker_match_score` (0–1) and `speaker_verified` (bool).
    """
    audio_bytes = await audio.read()
    detector = VoiceDetector(vector_db=request.app.state.vector_db)
    result = detector.analyze(audio_bytes, audio.filename, customer_id=customer_id)
    result["analysis_id"] = str(uuid.uuid4())[:8]

    # Promote speaker fields to the top-level response for convenience
    raw = result.get("raw", {})
    if customer_id is not None:
        result["speaker_match_score"] = raw.get("speaker_match_score")
        result["speaker_verified"] = raw.get("speaker_verified")

    return result


@app.post("/enroll-voice")
async def enroll_voice(
    customer_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    Enroll a customer's voice for future speaker verification.

    Extracts a 192-dim speaker embedding via SpeechBrain ECAPA-TDNN and
    stores it under backend/data/voice_enrollments/<customer_id>.npy.

    On first call the model (~50 MB) is auto-downloaded from HuggingFace.
    Subsequent calls load instantly from the local cache.
    """
    try:
        from core.speaker_verification import enroll_speaker
        audio_bytes = await audio.read()
        result = enroll_speaker(customer_id, audio_bytes, audio.filename)
        result["analysis_id"] = str(uuid.uuid4())[:8]
        return result
    except RuntimeError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/analyze/video")
async def analyze_video(video: UploadFile = File(...)):
    """Video deepfake detection — temporal consistency + facial artifacts + AV sync."""
    video_bytes = await video.read()
    detector = VideoDetector()
    result = detector.analyze(video_bytes, video.filename)
    result["analysis_id"] = str(uuid.uuid4())[:8]
    return result


@app.post("/analyze/file")
async def analyze_file(attachment: UploadFile = File(...)):
    """YARA + ClamAV/VirusTotal file scan."""
    file_bytes = await attachment.read()
    detector = FileDetector()
    result = detector.analyze(file_bytes, attachment.filename)
    result["analysis_id"] = str(uuid.uuid4())[:8]
    return result


@app.post("/analyze/email")
async def analyze_email(
    raw_email: str = Form(default=""),
    body: str = Form(default=""),
    sender: str = Form(default="unknown"),
):
    """Analyze email headers (SPF/DKIM/DMARC) + body."""
    detector = EmailDetector()
    if raw_email:
        result = detector.analyze_raw(raw_email)
    else:
        result = detector.analyze_body(body, sender)
    result["analysis_id"] = str(uuid.uuid4())[:8]
    return result


# ── Full Analysis (Central Classifier) ──────────────────────────────────────

@app.post("/analyze/full")
async def analyze_full(
    message: str = Form(default=""),
    sender: str = Form(default="unknown"),
    channel: str = Form(default="sms"),
    url: str = Form(default=""),
    audio: UploadFile = File(default=None),
    attachment: UploadFile = File(default=None),
):
    """Run ALL available detectors and return a combined threat assessment."""
    classifier = CentralClassifier(app.state.vector_db)
    result = await classifier.classify(
        message=message,
        sender=sender,
        channel=channel,
        url=url,
        audio_bytes=(await audio.read()) if audio else None,
        audio_filename=audio.filename if audio else "",
        file_bytes=(await attachment.read()) if attachment else None,
        file_filename=attachment.filename if attachment else "",
    )
    result["analysis_id"] = str(uuid.uuid4())[:8]
    return result


# ── IMAP Email Scanning ─────────────────────────────────────────────────────

@app.post("/email/scan-inbox")
async def scan_inbox(
    imap_host: str = Form(...),
    email_addr: str = Form(...),
    password: str = Form(...),
    count: int = Form(default=10),
):
    """Connect to IMAP inbox and scan recent emails."""
    fetcher = IMAPFetcher(imap_host, email_addr, password)
    emails = fetcher.fetch_recent(count=count)
    detector = EmailDetector()
    text_det = TextDetector(app.state.vector_db)

    results = []
    for em in emails:
        if "error" in em:
            results.append(em)
            continue

        # Header analysis
        email_result = detector.analyze_raw(em.get("raw", ""))

        # Body text analysis
        body = em.get("body", "")
        text_result = text_det.analyze(body, em.get("from", "unknown"), "email") if body else None

        # Apply whitelist and authentication adjustment to body text analysis
        is_legit = email_result.get("raw", {}).get("is_authenticated_legit", False)
        if is_legit and text_result:
            text_result["score"] = 0.0
            text_result["reasons"] = ["Whitelisted and SPF/DKIM authenticated domain verified - body text risk cleared"] + text_result["reasons"]
            text_result["verdict"] = "SAFE"
            text_result["severity"] = "LOW"
            text_result["confidence"] = "NONE"
            text_result["band"] = {"band": "LEGITIMATE", "color": "green", "range": "0–24"}

        combined = ThreatScore.combine(
            {"email": email_result, **({"text": text_result} if text_result else {})}
        )

        results.append({
            "email_id": em.get("id"),
            "from": em.get("from"),
            "subject": em.get("subject"),
            "date": em.get("date"),
            "analysis": combined,
        })

    return {"emails_scanned": len(results), "results": results}


# ── Human-in-the-Loop Feedback ──────────────────────────────────────────────

@app.post("/feedback")
async def submit_feedback(
    analysis_id: str = Form(...),
    user_verdict: str = Form(...),  # "scam" | "safe" | "unsure"
    original_score: float = Form(default=0),
    original_verdict: str = Form(default=""),
    source: str = Form(default=""),
    original_input: str = Form(default=""),
    comment: str = Form(default=""),
):
    """Submit human feedback on an analysis result. Improves future accuracy."""
    entry = app.state.feedback.add_feedback(
        analysis_id=analysis_id,
        user_verdict=user_verdict,
        original_score=original_score,
        original_verdict=original_verdict,
        source=source,
        original_input=original_input,
        comment=comment,
    )

    # If user confirms it's a scam, add to vector DB for future matching
    if user_verdict == "scam" and original_input:
        app.state.vector_db.add_scam(original_input, reported_by="user_feedback")

    return {"status": "feedback recorded", "entry": entry}


@app.get("/feedback/stats")
async def feedback_stats():
    """Get accuracy statistics from user feedback."""
    return app.state.feedback.get_accuracy_stats()


@app.get("/feedback/recent")
async def recent_feedback(limit: int = Query(default=20)):
    """Get recent feedback entries."""
    return app.state.feedback.get_recent(limit=limit)


# ── Detection Timing Stats ───────────────────────────────────────────────────────────

@app.get("/detection-stats")
async def detection_stats():
    """
    Return aggregated timing stats for all completed live calls.

    Demonstrates the "flag within 10 seconds" checkpoint:
      - total_calls           – completed calls recorded in this session
      - total_flagged         – calls that crossed HIGH_RISK_THRESHOLD (70)
      - avg_time_to_alert     – mean seconds from call start to first HIGH-RISK flag
      - calls_flagged_under_10s – calls flagged in ≤ 10 seconds (the demo goal)
    """
    return get_detection_stats()


# ── Live Call WebSocket ──────────────────────────────────────────────────────

@app.websocket("/ws/live-call/{call_id}")
async def live_call_ws(websocket: WebSocket, call_id: str):
    """
    Real-time call analysis via WebSocket.

    Protocol:
      Client sends: binary audio chunks (5s WAV blobs)
      Server sends: JSON RiskState updates after each chunk
      Client sends: text "END" to close call and get final summary
    """
    await websocket.accept()
    vector_db = websocket.app.state.vector_db

    try:
        while True:
            message = await websocket.receive()

            # Text message — only "END" is supported
            if "text" in message:
                if message["text"].strip().upper() == "END":
                    state = end_call(call_id)
                    await websocket.send_text(json.dumps({
                        "type": "call_ended",
                        "final_score": round(state.current_score, 1) if state else 0,
                        "verdict": state.to_dict()["verdict"] if state else "UNKNOWN",
                        "full_transcript": state.transcript_so_far if state else "",
                        "intent_progression": state.intent_progression if state else [],
                    }))
                    break

            # Binary message — audio chunk
            elif "bytes" in message:
                audio_bytes = message["bytes"]
                if not audio_bytes:
                    continue

                result = process_chunk(call_id, audio_bytes, vector_db)
                result["type"] = "chunk_result"
                await websocket.send_text(json.dumps(result))

                # If alert threshold crossed (score ≥ 80) OR high-risk first-crossing
                # (score ≥ 70, first_alert_time just set), send a separate alert event.
                if result.get("alert") or result.get("high_risk_triggered"):
                    await websocket.send_text(json.dumps({
                        "type": "alert",
                        "call_id": call_id,
                        "score": result["current_score"],
                        "message": "HIGH RISK SCAM DETECTED — Advise caller to hang up immediately",
                        "intent_progression": result["intent_progression"],
                        # ── Timing instrumentation fields ────────────────────────
                        "elapsed_seconds": result.get("elapsed_seconds"),
                        "high_risk_triggered": result.get("high_risk_triggered", False),
                        "time_to_alert_seconds": result.get("time_to_alert_seconds"),
                    }))

    except WebSocketDisconnect:
        end_call(call_id)
    except RuntimeError as e:
        if "disconnect message" not in str(e):
            print(f"[WS ERROR] {call_id}: {e}")
        end_call(call_id)

# ─────────────────────────────────────────────────────────────────────────────
# Voice Forensics — Live Stream Buffer
# Runs parallel to the Voice Lab scam pipeline on the same WebSocket connection.
# Accumulates 10 seconds of audio then fires analyze_audio() asynchronously.
# ─────────────────────────────────────────────────────────────────────────────
import os as _os, io as _io, wave as _wave, tempfile as _tempfile
import numpy as _np

FORENSICS_WINDOW_SEC  = int(_os.getenv("FORENSICS_STREAM_WINDOW_SEC", "10"))
FORENSICS_TARGET_SR   = 16_000
FORENSICS_WIN_SAMPLES = FORENSICS_WINDOW_SEC * FORENSICS_TARGET_SR  # 160,000 samples

# ── Channel routing for Live Call Voice Forensics ────────────────────────────
# Browser stereo WAV channel layout (set explicitly in page.tsx):
#   ch0 / LEFT  = local microphone  (always human, must NOT be analyzed)
#   ch1 / RIGHT = tab audio         (remote caller — the analysis target)
#
# This env-var selects which channel index is the CALLER channel.
# Default is 1 (right). Override to 0 if your capture pipeline is inverted.
# Any value other than 0 or 1 will fall back to safe right-channel default.
FORENSICS_CALLER_CHANNEL = int(_os.getenv("FORENSICS_CALLER_CHANNEL", "1"))

# ── Debug: save the exact WAV fed to analyze_audio() for listening tests ──────
# Set  DEBUG_FORENSICS=1  to enable.
# Files are saved to:  backend/debug_forensics/<session_id>_chunk<N>.wav
# Listen to these files to verify channel routing and audio content.
# Disable in production (default off).
DEBUG_FORENSICS     = _os.getenv("DEBUG_FORENSICS", "0").strip() == "1"
DEBUG_FORENSICS_DIR = _os.path.join(_os.path.dirname(__file__), "debug_forensics")


class ForensicsStreamBuffer:
    """
    Per-session accumulator that converts incoming WAV chunks from the
    production live call WebSocket into a unified forensic analysis window.

    - Ingests 1-second stereo 16 kHz WAV blobs from the browser.
    - Mixes to mono float32 internally by selecting the caller channel.
    - Exposes functions to generate live spectrogram previews and slice
      exactly the first 10 seconds of audio.
    """
    def __init__(self, session_id: str):
        import time as _time
        self.session_id = session_id
        self._buf: list = []          # float32 mono samples
        self.total_samples_ingested: int = 0
        self.is_completed: bool = False
        self.chunk_count: int = 0
        self.last_chunk_time: float = _time.time()

    @property
    def buffer_length(self) -> int:
        """Returns the current number of accumulated float32 mono samples in the buffer."""
        return len(self._buf)

    @property
    def idle_time(self) -> float:
        """Returns the time elapsed in seconds since the last chunk was ingested."""
        import time as _time
        return _time.time() - self.last_chunk_time

    # ------------------------------------------------------------------
    def ingest(self, raw: bytes) -> None:
        """Append raw bytes (WAV or raw PCM int16)."""
        import time as _time
        self.chunk_count += 1
        samples = self._parse(raw)
        added = len(samples)
        if added:
            self._buf.extend(samples.tolist())
            self.total_samples_ingested += added
        self.last_chunk_time = _time.time()

    def get_current_wav_bytes(self) -> bytes:
        """Return all currently accumulated samples in the buffer as a 16 kHz mono WAV bytes blob."""
        chunk = _np.array(self._buf, dtype=_np.float32)
        pcm = (chunk * 32767.0).clip(-32768, 32767).astype(_np.int16)
        buf = _io.BytesIO()
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(FORENSICS_TARGET_SR)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def get_first_10s_wav_bytes(self) -> bytes:
        """Extract at most the first 10 seconds (up to 160,000 samples) as mono 16kHz WAV."""
        REQUIRED_SAMPLES = 160_000
        available = len(self._buf)
        if available < 80_000:
            raise RuntimeError(
                f"Need at least 80,000 samples, got {available}"
            )
        use_samples = min(available, REQUIRED_SAMPLES)
        chunk = _np.array(self._buf[:use_samples], dtype=_np.float32)
        pcm = (chunk * 32767.0).clip(-32768, 32767).astype(_np.int16)
        buf = _io.BytesIO()
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(FORENSICS_TARGET_SR)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
    # ------------------------------------------------------------------
    def _parse(self, raw: bytes) -> _np.ndarray:
        """
        Convert an incoming WAV (or raw PCM int16) blob to a mono float32 array
        containing ONLY the remote caller's audio.

        Stereo channel layout (guaranteed by page.tsx ChannelMergerNode wiring):
          ch0 / LEFT  (index 0) — local microphone  → DISCARDED
          ch1 / RIGHT (index 1) — tab / caller audio → KEPT

        We intentionally discard the local microphone channel because:
          • The Spectrogram CNN was fine-tuned on single-speaker recordings.
          • Blending two speakers corrupts MFCCs, harmonics, and spectrogram
            texture, causing the model to misclassify AI-generated voices.
          • The microphone captures the *analyst's* voice, which is always
            human and must never influence the Voice Forensics verdict.

        The selected channel index is configurable via the FORENSICS_CALLER_CHANNEL
        environment variable (default 1 = RIGHT) to handle edge cases where the
        capture pipeline maps channels differently across browsers or OS versions.

        Mono input is passed through unchanged for backward compatibility with the
        Upload Audio pipeline and any mono-only call sources.
        """
        try:
            if raw[:4] == b"RIFF" and b"WAVE" in raw[:20]:
                with _wave.open(_io.BytesIO(raw), "rb") as wf:
                    channels    = wf.getnchannels()
                    sample_rate = wf.getframerate()
                    n_frames    = wf.getnframes()
                    content     = wf.readframes(n_frames)
                    s = _np.frombuffer(content, dtype=_np.int16).astype(_np.float32) / 32768.0
            else:
                # Raw PCM fallback (mono assumed — Upload Audio or non-browser source)
                channels    = 1
                sample_rate = FORENSICS_TARGET_SR
                n_frames    = len(raw) // 2
                s = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0

            if channels == 2 and len(s) % 2 == 0:
                # ── Stereo: extract CALLER channel only ──────────────────────
                #
                # Safety: clamp FORENSICS_CALLER_CHANNEL to valid range [0, channels-1].
                # An invalid index (e.g. env-var set to 2 on a stereo stream, or -1)
                # would cause an IndexError, so we clamp and log a WARNING instead of
                # crashing, then fall back to the default right channel (index 1).
                raw_cfg   = FORENSICS_CALLER_CHANNEL
                caller_ch = raw_cfg if 0 <= raw_cfg < channels else channels - 1
                if caller_ch != raw_cfg:
                    print(
                        f"[ForensicsBuffer] {self.session_id}: ⚠️  WARNING — "
                        f"FORENSICS_CALLER_CHANNEL={raw_cfg} is out of range for a "
                        f"{channels}-channel stream. Clamped to ch{caller_ch} (last channel). "
                        f"Set FORENSICS_CALLER_CHANNEL=0 or =1 to silence this warning."
                    )
                local_ch = 1 - caller_ch  # the other channel (microphone side)

                stereo        = s.reshape(-1, 2)
                local_mic     = stereo[:, local_ch]   # analyst's voice — discarded
                remote_caller = stereo[:, caller_ch]  # caller's voice  — analyzed

                # ── Per-channel diagnostics (supports the silence/speech test) ─
                #
                # HOW TO VERIFY CHANNEL ROUTING:
                #   Test A — you speak, caller is silent:
                #     local_mic_rms  should be HIGH   (your voice)
                #     remote_caller_rms should be LOW (silent caller)
                #   Test B — caller speaks, you are silent:
                #     local_mic_rms  should be LOW    (you silent)
                #     remote_caller_rms should be HIGH (caller speaking)
                rms_caller = float(_np.sqrt(_np.mean(remote_caller ** 2)))
                rms_local  = float(_np.sqrt(_np.mean(local_mic    ** 2)))
                peak       = float(_np.max(_np.abs(remote_caller)))
                duration_s = len(remote_caller) / FORENSICS_TARGET_SR
                first_vals = remote_caller[:10].tolist()

                # ── Smart Fallback: If remote_caller is silent (< -46dBFS / 0.005) but local_mic has audio, use local_mic ──
                selected_ch_name = f"ch{caller_ch} (remote caller)"
                if rms_caller < 0.005 and rms_local >= 0.005:
                    s = local_mic
                    selected_ch_name = f"ch{local_ch} (local mic fallback)"
                    peak = float(_np.max(_np.abs(local_mic)))
                else:
                    s = remote_caller

                print(
                    f"[ForensicsBuffer] {self.session_id}: stereo WAV received\n"
                    f"  Detected channels   : {channels}  (ch{local_ch}=local_mic, ch{caller_ch}=remote_caller)\n"
                    f"  Configured ch index : FORENSICS_CALLER_CHANNEL={raw_cfg}"
                    + (f" → clamped to ch{caller_ch}" if caller_ch != raw_cfg else "") + "\n"
                    f"  Selected channel    : {selected_ch_name}\n"
                    f"  Sample rate         : {sample_rate} Hz\n"
                    f"  Duration            : {duration_s:.3f}s  ({len(s)} samples)\n"
                    f"  First 10 samples    : {[f'{v:.4f}' for v in first_vals]}\n"
                    f"  RMS remote_caller   : {rms_caller:.6f}  ({20 * _np.log10(rms_caller + 1e-9):.1f} dBFS)\n"
                    f"  RMS local_mic       : {rms_local:.6f}  ({20 * _np.log10(rms_local  + 1e-9):.1f} dBFS)\n"
                    f"  Peak (selected)     : {peak:.6f}"
                )

            else:
                # ── Mono: pass through unchanged ─────────────────────────────
                # Handles Upload Audio, Twilio, and any mono-only call source.
                duration_s = len(s) / FORENSICS_TARGET_SR
                rms        = float(_np.sqrt(_np.mean(s ** 2)))
                peak       = float(_np.max(_np.abs(s)))
                print(
                    f"[ForensicsBuffer] {self.session_id}: mono WAV received\n"
                    f"  Detected channels : {channels} (mono — no channel split needed)\n"
                    f"  Sample rate       : {sample_rate} Hz\n"
                    f"  Duration          : {duration_s:.3f}s  ({len(s)} samples)\n"
                    f"  RMS               : {rms:.6f}  ({20 * _np.log10(rms + 1e-9):.1f} dBFS)\n"
                    f"  Peak              : {peak:.6f}"
                )

            return s

        except Exception as exc:
            print(f"[ForensicsBuffer] {self.session_id}: _parse error — {exc}")
            return _np.zeros(0, dtype=_np.float32)


async def _live_call_timer_worker(
    ws: WebSocket,
    session_id: str,
    forensics_buf: ForensicsStreamBuffer,
    start_time: float
):
    """
    Timer-based worker (Thread B equivalent) that sleeps/waits until at least 10.0 seconds
    have elapsed and at least 160,000 samples of caller audio have been ingested,
    then freezes the buffer, exports to WAV, runs single analysis, and returns the final report.
    If the call is hung up early or times out, it aborts analysis with an Insufficient Audio error.
    """
    try:
        import time as _time
        import json as _json

        last_logged_samples = -1
        while True:
            elapsed = _time.time() - start_time
            samples = forensics_buf.total_samples_ingested

            if samples != last_logged_samples:
                print(f"Worker waiting... samples={samples}")
                last_logged_samples = samples
            if (elapsed >= 10.0 and samples >= 160_000) or getattr(forensics_buf, 'force_finish', False):
                print("Requirements satisfied or force finished.")
                print("Freezing buffer.")
                break
            # 1. Startup timeout: if no chunks arrived after 10 seconds of connection
            if forensics_buf.chunk_count == 0 and elapsed >= 10.0:
                err_msg = (
                    "Insufficient audio captured.\n"
                    "\n"
                    "Required: 10.0 seconds\n"
                    "Captured: 0.00 seconds (No chunks received)\n"
                    "\n"
                    "Analysis aborted."
                )
                print(err_msg)
                try:
                    await ws.send_text(_json.dumps({
                        "type": "error",
                        "message": err_msg
                    }))
                    await ws.send_text(_json.dumps({
                        "type": "call_ended",
                        "call_id": session_id
                    }))
                except Exception:
                    pass
                return

            # 2. Idle timeout: if chunks started arriving but stopped for more than 15.0 seconds (stream hung up/ended)
            if forensics_buf.chunk_count > 0 and forensics_buf.idle_time >= 15.0:
                captured_sec = samples / 16000.0
                err_msg = (
                    f"Insufficient audio captured.\n"
                    f"\n"
                    f"Required: 10.0 seconds\n"
                    f"Captured: {captured_sec:.2f} seconds (Stream went silent)\n"
                    f"\n"
                    f"Analysis aborted."
                )
                print(err_msg)
                try:
                    await ws.send_text(_json.dumps({
                        "type": "error",
                        "message": err_msg
                    }))
                    await ws.send_text(_json.dumps({
                        "type": "call_ended",
                        "call_id": session_id
                    }))
                except Exception:
                    pass
                return

            await asyncio.sleep(0.05)

        # 5. Immediately before get_first_10s_wav_bytes(), print
        print(f"Elapsed = {_time.time() - start_time:.2f}")
        print(f"Samples = {forensics_buf.total_samples_ingested}")
        print(f"Buffer length = {forensics_buf.buffer_length}")
        print(f"Total chunks = {forensics_buf.chunk_count}")

        # 1. Immediately freeze the first 10 seconds of collected PCM audio (by making a copy)
        wav_bytes = forensics_buf.get_first_10s_wav_bytes()
        forensics_buf.is_completed = True
        print("Buffer frozen")

        # 6. Verify the buffer itself
        print(f"buffer_length = {forensics_buf.buffer_length}")
        print(f"total_samples_ingested = {forensics_buf.total_samples_ingested}")

        # Log total samples and duration before exporting
        samples_count = forensics_buf.buffer_length
        duration = samples_count / 16000.0
        print(f"Total PCM samples: {samples_count}")
        print(f"Duration: {duration:.3f} s")

        # 2. Save mono 16 kHz PCM WAV locally
        import tempfile as _tempfile
        import shutil as _shutil
        temp_path = None
        with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            temp_path = tmp.name
        print("WAV exported")

        if DEBUG_FORENSICS:
            try:
                _os.makedirs(DEBUG_FORENSICS_DIR, exist_ok=True)
                debug_name = f"{session_id}_chunk01_t10s.wav"
                debug_path = _os.path.join(DEBUG_FORENSICS_DIR, debug_name)
                _shutil.copy2(temp_path, debug_path)
                print(f"[ForensicsDebug] {session_id}: 🎧 debug WAV saved → {debug_path}")
            except Exception as _dbg_err:
                print(f"[ForensicsDebug] {session_id}: ⚠️ Could not save debug WAV — {_dbg_err}")

        # 4. Trigger Voice Clone started + Spectrogram started
        print("Voice Clone started")
        print("Spectrogram started")

        from detectors.voice_clone_detector import analyze_audio
        import functools as _functools

        loop = asyncio.get_event_loop()
        analysis_start = _time.time()
        # Run analyze_audio in background executor thread (non-blocking)
        result = await loop.run_in_executor(None, _functools.partial(analyze_audio, temp_path, mode="live"))
        analysis_latency = _time.time() - analysis_start

        # Print requested raw outputs on backend
        print("\n========================")
        print("VOICE CLONE RAW OUTPUT  ")
        print("========================")
        print(f"  Model             : garystafford/wav2vec2-deepfake-voice-detector")
        print(f"  Raw Model Output  : {result.get('raw_model_output')}")
        print(f"  Raw Score Map     : {result.get('raw_scores')}")
        clone_score = result.get("model_score")
        human_score = 1.0 - clone_score if clone_score is not None else 0.0
        print(f"  Human Probability : {human_score:.6f}  ({human_score*100:.2f}%)")
        print(f"  Clone Probability : {clone_score:.6f}  ({clone_score*100:.2f}%)" if clone_score is not None else "  Clone Probability : 0.0")
        print(f"  Mapped fake_prob  : {clone_score:.6f}  (synthetic score sent to fusion)" if clone_score is not None else "  Mapped fake_prob  : 0.0")
        print(f"  Label before Fusion: {result.get('voice_clone_analysis', {}).get('prediction')}")
        print("========================\n")

        print("========================")
        print("SPECTROGRAM CNN RAW     ")
        print("========================")
        spec_analysis = result.get("spectrogram_analysis", {})
        print(f"  Prediction (Display): {spec_analysis.get('prediction')}")
        print(f"  Prediction (Raw)    : {spec_analysis.get('prediction_raw')}")
        print(f"  Confidence          : {spec_analysis.get('confidence'):.2f}%" if spec_analysis.get('confidence') is not None else "  Confidence          : 0.0%")
        print(f"  Raw Score           : {spec_analysis.get('score'):.6f}" if spec_analysis.get('score') is not None else "  Raw Score           : 0.0")
        print(f"  Reasons             : {spec_analysis.get('reasons')}")
        print(f"  Forensic Note       : {spec_analysis.get('forensic_note')}")
        print("========================\n")

        print("========================")
        print("THREAT FUSION           ")
        print("========================")
        threat_fusion = result.get("threat_fusion", {})
        from detectors.voice_clone_detector import LIVE_FUSION_WEIGHT_CLONE, LIVE_FUSION_WEIGHT_SPEC
        print(f"  Final Risk Score    : {threat_fusion.get('final_risk_score')}%")
        print(f"  Final Risk Level    : {threat_fusion.get('risk_level')}")
        print(f"  Clone Weight        : {LIVE_FUSION_WEIGHT_CLONE:.2f}")
        print(f"  Spec Weight         : {LIVE_FUSION_WEIGHT_SPEC:.2f}")
        print(f"  Explanation         : {threat_fusion.get('explanation')}")
        print("========================\n")

        print("Threat Fusion complete")

        # Cleanup temporary audio WAV
        if temp_path and _os.path.exists(temp_path):
            try:
                _os.unlink(temp_path)
            except Exception:
                pass

        # 7. Generate final report payload
        payload = {
            "type": "forensics_final_report",
            "session_id": session_id,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "risk_level": result.get("risk_level"),
            "threat_score": result.get("fusion_score") * 100.0 if result.get("fusion_score") is not None else 0.0,
            "reasons": result.get("reasons"),
            "spectrogram_image": result.get("spectrogram_image"),
            "voice_clone_analysis": result.get("voice_clone_analysis"),
            "spectrogram_analysis": result.get("spectrogram_analysis"),
            "threat_fusion": result.get("threat_fusion")
        }

        try:
            # Send report to frontend immediately
            await ws.send_text(json.dumps(payload))
            print("Report sent")
            print(f"Total analysis latency: {analysis_latency:.1f} sec")

            # End call session
            await ws.send_text(json.dumps({
                "type": "call_ended",
                "call_id": session_id
            }))
        except Exception as send_err:
            print(f"[ForensicsStream] {session_id}: Error sending report/call_ended — {send_err}")

    except asyncio.CancelledError:
        # 9. Handle early hang up during timer worker wait/run
        if forensics_buf.total_samples_ingested < 80_000:
            captured_sec = forensics_buf.total_samples_ingested / 16000.0
            err_msg = (
                f"Insufficient audio captured.\n"
                f"Required: 10 seconds\n"
                f"Captured: {captured_sec:.2f} seconds\n"
                f"Analysis aborted."
            )
            print(err_msg)
            try:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": err_msg
                }))
            except Exception:
                pass
    except Exception as exc:
        import traceback
        print(f"[ForensicsStream] {session_id}: worker exception — {exc}")
        traceback.print_exc()


@app.websocket("/ws/production-live-call/{call_id}")
async def production_live_call_ws(websocket: WebSocket, call_id: str, customer_id: Optional[str] = Query(default=None), caller_number: Optional[str] = Query(default=None)):
    """
    Production Live Call WebSocket.
    PIPELINE: AudioPreprocessor → SlidingWindowBuffer → Thread-Parallel Analyzer → ThreatFusion
              + ForensicsStreamBuffer (10s capture timer) → VoiceCloneDetector → SpectrogramAnalysis
    """
    await websocket.accept()
    _loop = asyncio.get_event_loop()

    import time as _time
    start_time = _time.time()
    first_chunk_time = None
    print(f"[WS-DEBUG] {call_id}: WebSocket ACCEPTED from {websocket.client} at {_time.strftime('%H:%M:%S', _time.localtime(start_time))}")

    print("Call Started")
    print("Recording Started")

    # ── Voice Forensics stream buffer ─────────────────────────────────────────
    forensics_buf = ForensicsStreamBuffer(call_id)
    print(f"[WS-DEBUG] {call_id}: ForensicsStreamBuffer created")

    # Start the 10-second timer analysis background worker (Thread B equivalent)
    analysis_task = asyncio.create_task(_live_call_timer_worker(websocket, call_id, forensics_buf, start_time))

    # Check if caller matches a saved contact
    saved_contact = _find_contact(_load_contacts(), caller_number or "")
    if saved_contact:
        print(f"[WS-DEBUG] {call_id}: saved_contact matched — {saved_contact['name']} (analysis bypassed)")
        await websocket.send_text(json.dumps({
            "type": "contact_verified",
            "name": saved_contact["name"],
            "phone": saved_contact["phone"],
            "verdict": "SAFE",
            "threat_score": 0.0,
            "explainable_reasons": [f"Identity verified via saved contact: {saved_contact['name']}"]
        }))

    if not hasattr(websocket.app.state, 'production_pipeline'):
        websocket.app.state.production_pipeline = ProductionPipelineManager(vector_db=websocket.app.state.vector_db)
    pipeline_mngr = websocket.app.state.production_pipeline

    # ── Keepalive: send a ping every 15s so the browser doesn't timeout ───────
    _keepalive_active = True
    async def _keepalive():
        while _keepalive_active:
            await asyncio.sleep(15)
            if not _keepalive_active:
                break
            try:
                await websocket.send_text(json.dumps({"type": "ping", "call_id": call_id}))
                print(f"[WS-DEBUG] {call_id}: keepalive ping sent")
            except Exception:
                break
    keepalive_task = asyncio.create_task(_keepalive())

    chunk_count = 0
    try:
        while True:
            message = await websocket.receive()

            # ── TEXT messages (END command or pong) ───────────────────────
            if "text" in message:
                text_cmd = message["text"].strip().upper()
                print(f"[WS-DEBUG] {call_id}: text message received: '{message['text'][:40]}'")
                if text_cmd == "END":
                    print(f"[WS-DEBUG] {call_id}: END command received — {forensics_buf.total_samples_ingested} samples accumulated ({forensics_buf.total_samples_ingested/16000:.1f}s)")
                    if forensics_buf.total_samples_ingested < 80_000:
                        captured_sec = forensics_buf.total_samples_ingested / 16000.0
                        err_msg = (
                            f"Insufficient audio captured.\n"
                            f"Required: 10 seconds\n"
                            f"Captured: {captured_sec:.2f} seconds\n"
                            f"Analysis aborted."
                        )
                        print(err_msg)
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": err_msg
                            }))
                            await websocket.send_text(json.dumps({
                                "type": "call_ended",
                                "call_id": call_id
                            }))
                        except Exception:
                            pass
                    pipeline_mngr.remove_session(call_id)
                    break

            # ── BYTES messages (PCM/WAV audio chunks) ───────────────────
            elif "bytes" in message:
                pcm_bytes = message["bytes"]
                chunk_count += 1
                if chunk_count == 1:
                    first_chunk_time = _time.time()
                    delay = first_chunk_time - start_time
                    print(f"[WS-DEBUG] {call_id}: First audio chunk arrived at {_time.strftime('%H:%M:%S', _time.localtime(first_chunk_time))} (delay: {delay:.2f}s)")
                chunk_kb = len(pcm_bytes) / 1024
                print(f"[WS-DEBUG] {call_id}: audio chunk #{chunk_count} received — {chunk_kb:.1f} KB")

                if not pcm_bytes:
                    print(f"[WS-DEBUG] {call_id}: chunk #{chunk_count} EMPTY — skipping")
                    continue

                # ── Voice Forensics: accumulate raw samples IMMEDIATELY ───────
                if not forensics_buf.is_completed:
                    prev_samples = forensics_buf.total_samples_ingested
                    forensics_buf.ingest(pcm_bytes)
                    added_samples = forensics_buf.total_samples_ingested - prev_samples
                    print(f"Chunk {chunk_count}")
                    print(f"Added {added_samples}")
                    print(f"Running Total {forensics_buf.total_samples_ingested}")

                if saved_contact:
                    await websocket.send_text(json.dumps({
                        "type": "chunk_result",
                        "session_id": call_id,
                        "timestamp": 0.0,
                        "threat_score": 0.0,
                        "verdict": "SAFE",
                        "severity": "NONE",
                        "deepfake_confidence": 0.0,
                        "scam_intent_confidence": 0.0,
                        "transcript_segment": "",
                        "full_transcript": "",
                        "spectrogram_image": None,
                        "explainable_reasons": [f"Contact identity verified as '{saved_contact['name']}'. Forensic scan bypassed."],
                        "risk_timeline": [0.0],
                        "alerts_triggered": False,
                        "is_saved_contact": True,
                        "contact_name": saved_contact["name"]
                    }))
                    continue

                # ━━ Scam Detection Pipeline (unmodified) ━━━
                print(f"[WS-DEBUG] {call_id}: chunk #{chunk_count} — dispatching to thread executor")
                updates = await _loop.run_in_executor(
                    None,
                    pipeline_mngr.process_pcm_chunk,
                    call_id,
                    pcm_bytes,
                    customer_id
                )
                print(f"[WS-DEBUG] {call_id}: chunk #{chunk_count} — pipeline returned {len(updates)} update(s)")

                for upd in updates:
                    upd["type"] = "chunk_result"
                    await websocket.send_text(json.dumps(upd))

                    if upd.get("alerts_triggered"):
                        await websocket.send_text(json.dumps({
                            "type":      "alert",
                            "call_id":   call_id,
                            "score":     upd["threat_score"],
                            "message":   f"CRITICAL SCAM WARNING: {upd['explainable_reasons'][0] if upd['explainable_reasons'] else 'High risk activity detected'}",
                            "timestamp": upd["timestamp"]
                        }))

                # Send a visual spectrogram preview during active recording (without model inference)
                if not forensics_buf.is_completed:
                    try:
                        from core.spectrogram_generator import generate_spectrogram_image
                        current_wav = forensics_buf.get_current_wav_bytes()
                        if len(current_wav) > 44:
                            spec_img = generate_spectrogram_image(current_wav, "preview.wav")
                            if spec_img:
                                preview_payload = {
                                    "type": "forensics_chunk_preview",
                                    "session_id": call_id,
                                    "chunk_index": chunk_count,
                                    "elapsed_seconds": chunk_count,
                                    "spectrogram_image": spec_img
                                }
                                await websocket.send_text(json.dumps(preview_payload))
                    except Exception as preview_err:
                        print(f"[WS-DEBUG] {call_id}: preview generation failed: {preview_err}")
    except (WebSocketDisconnect, RuntimeError) as wsd:
        # Ignore disconnect-related RuntimeError
        wsd_str = str(wsd).lower()
        is_disconnect = (
            "disconnect" in wsd_str or 
            'cannot call "receive"' in wsd_str or 
            "handler is closed" in wsd_str or 
            "transport closed" in wsd_str or 
            "closed=true" in wsd_str
        )
        if isinstance(wsd, RuntimeError) and not is_disconnect:
            import traceback
            print(f"[WS-DEBUG] {call_id}: UNEXPECTED RUNTIME EXCEPTION after {chunk_count} chunks — {wsd}")
            traceback.print_exc()
        else:
            print(f"[WS-DEBUG] {call_id}: WebSocket closed cleanly/client disconnected after {chunk_count} chunks")
        pipeline_mngr.remove_session(call_id)
    except Exception as e:
        import traceback
        print(f"[WS-DEBUG] {call_id}: UNEXPECTED EXCEPTION after {chunk_count} chunks — {e}")
        traceback.print_exc()
        pipeline_mngr.remove_session(call_id)
    finally:
        _keepalive_active = False
        try:
            keepalive_task.cancel()
        except Exception:
            pass

        if not analysis_task.done() and forensics_buf.total_samples_ingested >= 80_000:
            forensics_buf.force_finish = True
            print(f"[WS-DEBUG] {call_id}: Awaiting analysis task completion on END/close...")
            try:
                await asyncio.wait_for(analysis_task, timeout=40.0)
            except Exception as e:
                print(f"[WS-DEBUG] {call_id}: Error awaiting analysis task: {e}")
                try:
                    analysis_task.cancel()
                except Exception:
                    pass
        else:
            try:
                analysis_task.cancel()
            except Exception:
                pass

        print(f"[WS-DEBUG] {call_id}: session closed — total chunks received: {chunk_count}")



# ── Twilio Media Stream WebSocket ────────────────────────────────────────────
# Frontend connects here to watch live scores
_frontend_sockets: dict[str, WebSocket] = {}

@app.websocket("/ws/dashboard/{call_id}")
async def dashboard_ws(websocket: WebSocket, call_id: str):
    """Frontend browser connects here to receive live score updates."""
    await websocket.accept()
    _frontend_sockets[call_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        _frontend_sockets.pop(call_id, None)


@app.websocket("/ws/forensics/{call_id}")
async def forensics_ws(websocket: WebSocket, call_id: str):
    """
    Voice Forensics frontend connects here to receive live forensic analysis
    results pushed from the WhatsApp/Twilio call stream.

    The frontend should connect with the same call_id used when the Twilio
    call started (available from the /twilio/voice-webhook response or the
    dashboard WebSocket).  Once connected, the handler will receive a single
    'forensics_result' message after the first 10 seconds of the call are
    captured and analyzed.

    URL: wss://your-host/ws/forensics/{call_id}
    """
    await websocket.accept()
    print(f"[VoiceForensics WS] {call_id}: Frontend connected")

    # Attach this WebSocket to the existing Twilio handler for this call.
    # Use get_handler (not get_or_create) — the Twilio stream must already
    # be active. If the call hasn't started yet, inform the client and wait.
    handler = get_handler(call_id)
    if handler is None:
        # Call not started yet — send a status message and keep the connection
        # open so the client can receive forensics_result when the call arrives.
        # We'll try attaching again after each receive_text ping.
        import json as _json
        await websocket.send_text(_json.dumps({
            "type": "status",
            "message": "Waiting for Twilio call to connect...",
            "call_id": call_id
        }))

    if handler:
        handler.set_forensics_ws(websocket)

    try:
        while True:
            await websocket.receive_text()  # keep the connection alive
            # Re-attach if the handler appeared after initial connect
            if handler is None:
                handler = get_handler(call_id)
                if handler:
                    handler.set_forensics_ws(websocket)
    except WebSocketDisconnect:
        print(f"[VoiceForensics WS] {call_id}: Frontend disconnected")
        h = get_handler(call_id)
        if h:
            h.set_forensics_ws(None)


@app.websocket("/ws/twilio-stream/{call_id}")
async def twilio_stream_ws(websocket: WebSocket, call_id: str):
    """
    Twilio Media Streams connects here.
    Configure this URL in Twilio console as the Stream URL.
    Format: wss://your-ngrok-url/ws/twilio-stream/{call_id}
    """
    await websocket.accept()
    vector_db = websocket.app.state.vector_db
    frontend_ws = _frontend_sockets.get(call_id)
    handler = get_or_create_handler(call_id, vector_db, frontend_ws)

    try:
        while True:
            message = await websocket.receive_text()
            result = await handler.handle_message(message)
            if result:
                print(f"[TWILIO STREAM] {call_id}: {result}")
    except WebSocketDisconnect:
        # Fix 5: Normal call end — not an error
        print(f"[TWILIO STREAM] {call_id}: call ended (WebSocket closed)")
        remove_handler(call_id)
        end_call(call_id)
    except Exception as e:
        err = str(e)
        # Fix 5: ABNORMAL_CLOSURE 1006 = Twilio hung up = normal
        if "1006" in err or "ABNORMAL_CLOSURE" in err or "going away" in err.lower():
            print(f"[TWILIO STREAM] {call_id}: call ended normally")
        else:
            print(f"[TWILIO STREAM ERROR] {call_id}: {e}")
        remove_handler(call_id)
        end_call(call_id)


@app.post("/twilio/voice-webhook")
async def twilio_voice_webhook(request: Request):
    """
    Twilio calls this HTTP endpoint when a call comes in.
    Returns TwiML that forks audio to our WebSocket stream.
    Set this as your Twilio phone number's Voice URL.
    """
    import os
    from urllib.parse import urlencode

    # Generate a unique call ID
    call_id = str(uuid.uuid4())[:8]
    ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")

    if not ngrok_url:
        return {"error": "NGROK_URL not set in .env"}

    stream_url = f"wss://{ngrok_url.replace('https://', '').replace('http://', '')}/ws/twilio-stream/{call_id}"

    # TwiML response — tells Twilio to stream audio to us
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{stream_url}" />
    </Start>
    <Say>This call is being monitored for fraud protection.</Say>
    <Pause length="60"/>
</Response>"""

    from fastapi.responses import Response
    return Response(content=twiml, media_type="application/xml")


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
