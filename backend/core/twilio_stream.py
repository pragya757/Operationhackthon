"""
Twilio Media Streams Handler
─────────────────────────────
Fixes applied:
  - Fix 2: Upsample mulaw 8kHz → 16kHz before Whisper (better accuracy)
  - Fix 5: ABNORMAL_CLOSURE 1006 handled as normal call end, not error
"""

import base64
import audioop
import io
import json
import os
import tempfile
import wave
from typing import Dict, Optional

from core.live_call import process_chunk, end_call, get_or_create_call

# Twilio sends mulaw 8kHz mono
TWILIO_SAMPLE_RATE = 8000
TARGET_SAMPLE_RATE = 16000  # Fix 3: Whisper trained on 16kHz
TWILIO_CHANNELS = 1
CHUNK_DURATION_SEC = 5
SAMPLES_PER_CHUNK = TWILIO_SAMPLE_RATE * CHUNK_DURATION_SEC  # 40000 samples

# Forensics: capture the first 10 seconds (2 chunks) then run Voice Forensics
FORENSICS_CAPTURE_CHUNKS = 2       # 2 × 5s = 10 seconds
FORENSICS_CAPTURE_DONE_FLAG = "forensics_done"


def mulaw_to_wav(mulaw_bytes: bytes) -> bytes:
    """Convert raw mulaw bytes to WAV format (PCM 16-bit 16kHz mono).
    Fix 3: Upsample from 8kHz → 16kHz for better Whisper accuracy.
    """
    # mulaw → linear PCM 16-bit at 8kHz
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)

    # Fix 3: Upsample 8kHz → 16kHz using audioop
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, TWILIO_SAMPLE_RATE, TARGET_SAMPLE_RATE, None)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(TWILIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm_16k)
    return buf.getvalue()


class TwilioStreamHandler:
    """
    One instance per active Twilio call.
    Accumulates mulaw chunks, fires analysis every 5 seconds.
    """

    def __init__(self, call_sid: str, vector_db=None, frontend_ws=None):
        self.call_sid = call_sid
        self.vector_db = vector_db
        self.frontend_ws = frontend_ws
        self._mulaw_buffer: bytes = b""
        self._chunk_count = 0

        # ─ Voice Forensics capture state ─────────────────────────────────
        # Accumulates the first FORENSICS_CAPTURE_CHUNKS (2 × 5s = 10s) of
        # mulaw audio, then passes the full WAV to analyze_audio().
        # This runs independently of and in addition to the Voice Lab pipeline.
        self._forensics_mulaw: bytes = b""        # raw mulaw bytes accumulator
        self._forensics_chunks_captured: int = 0  # how many 5s chunks collected
        self._forensics_done: bool = False        # fire only once per call
        self._forensics_ws = None                 # optional dedicated frontend WS

    async def handle_message(self, raw: str) -> Optional[Dict]:
        try:
            msg = json.loads(raw)
        except Exception:
            return None

        event = msg.get("event")

        if event == "start":
            call_sid = msg.get("start", {}).get("callSid", self.call_sid)
            self.call_sid = call_sid
            get_or_create_call(call_sid)
            return {"event": "started", "call_sid": call_sid}

        elif event == "media":
            payload = msg.get("media", {}).get("payload", "")
            if not payload:
                return None

            mulaw_chunk = base64.b64decode(payload)
            self._mulaw_buffer += mulaw_chunk

            chunk_bytes = SAMPLES_PER_CHUNK  # 1 byte per mulaw sample

            if len(self._mulaw_buffer) >= chunk_bytes:
                chunk = self._mulaw_buffer[:chunk_bytes]
                self._mulaw_buffer = self._mulaw_buffer[chunk_bytes:]
                self._chunk_count += 1

                # Fix 3: Convert with upsampling to 16kHz
                wav_bytes = mulaw_to_wav(chunk)

                # ── Voice Forensics: accumulate first 10 seconds ──────────────
                if not self._forensics_done:
                    self._forensics_mulaw += chunk    # accumulate raw mulaw
                    self._forensics_chunks_captured += 1
                    print(
                        f"[ForensicsCapture] {self.call_sid}: "
                        f"chunk {self._forensics_chunks_captured}/{FORENSICS_CAPTURE_CHUNKS} "
                        f"({self._forensics_chunks_captured * CHUNK_DURATION_SEC}s / "
                        f"{FORENSICS_CAPTURE_CHUNKS * CHUNK_DURATION_SEC}s captured)"
                    )

                    if self._forensics_chunks_captured >= FORENSICS_CAPTURE_CHUNKS:
                        self._forensics_done = True
                        print(f"[ForensicsCapture] {self.call_sid}: First 10 seconds captured — starting forensic analysis")
                        # Run forensics asynchronously so it doesn't block chunk processing
                        import asyncio
                        asyncio.create_task(self._run_forensics())

                # ── Voice Lab pipeline (unchanged) ───────────────────────
                result = process_chunk(self.call_sid, wav_bytes, self.vector_db)
                result["type"] = "chunk_result"
                result["chunk_number"] = self._chunk_count

                # ── Timing instrumentation pass-through ───────────────────────
                # elapsed_seconds and high_risk_triggered are always set by
                # process_chunk(); time_to_alert_seconds only appears on the
                # chunk where the threshold is first crossed.
                # All three keys are already present in `result` — we just
                # ensure they are forwarded verbatim to the frontend WebSocket.
                # (No transformation needed — keys surfaced as-is.)

                if self.frontend_ws:
                    try:
                        await self.frontend_ws.send_text(json.dumps(result))
                        if result.get("alert") or result.get("high_risk_triggered"):
                            alert_payload = {
                                "type": "alert",
                                "call_id": self.call_sid,
                                "score": result["current_score"],
                                "message": "HIGH RISK SCAM DETECTED — Hang up immediately!",
                                "intent_progression": result.get("intent_progression", []),
                                # ── Timing fields ─────────────────────────────
                                "elapsed_seconds": result.get("elapsed_seconds"),
                                "high_risk_triggered": result.get("high_risk_triggered", False),
                                "time_to_alert_seconds": result.get("time_to_alert_seconds"),
                            }
                            await self.frontend_ws.send_text(json.dumps(alert_payload))
                    except Exception:
                        pass

                return result

        elif event == "stop":
            state = end_call(self.call_sid)
            if self.frontend_ws and state:
                try:
                    await self.frontend_ws.send_text(json.dumps({
                        "type": "call_ended",
                        "final_score": round(state.current_score, 1),
                        "verdict": state.to_dict()["verdict"],
                        "full_transcript": state.transcript_so_far,
                        "intent_progression": state.intent_progression,
                    }))
                except Exception:
                    pass
            return {"event": "stopped", "call_sid": self.call_sid}

        return None

    # ── Voice Forensics Analysis (runs after first 10s captured) ─────────────
    async def _run_forensics(self):
        """
        Convert the accumulated 10-second mulaw buffer to a 16kHz WAV,
        save it to a temp file, and run the full Voice Forensics pipeline
        (Voice Clone Detection + Spectrogram Analysis + Threat Fusion).
        Broadcasts the result to any connected forensics WebSocket client.
        Voice Lab chunk processing continues uninterrupted.
        """
        temp_path = None
        try:
            print(f"[VoiceForensics] {self.call_sid}: WhatsApp call connected — audio stream received")
            print(
                f"[VoiceForensics] {self.call_sid}: "
                f"Audio stream received — {FORENSICS_CAPTURE_CHUNKS * CHUNK_DURATION_SEC}s "
                f"@ {TWILIO_SAMPLE_RATE} Hz mulaw ({len(self._forensics_mulaw)} bytes)"
            )

            # Convert accumulated mulaw → 16kHz PCM WAV
            wav_bytes = mulaw_to_wav(self._forensics_mulaw)
            print(
                f"[VoiceForensics] {self.call_sid}: "
                f"First 10 seconds captured — WAV size={len(wav_bytes)} bytes @ {TARGET_SAMPLE_RATE} Hz"
            )

            # Save to temp file (analyze_audio requires a file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                temp_path = tmp.name
            print(f"[VoiceForensics] {self.call_sid}: Temp WAV written: {temp_path}")

            # ── Run the existing Voice Forensics pipeline ─────────────────────
            from detectors.voice_clone_detector import analyze_audio
            print(f"[VoiceForensics] {self.call_sid}: Voice Clone Detection started")
            print(f"[VoiceForensics] {self.call_sid}: Spectrogram Analysis started")
            result = analyze_audio(temp_path)

            tf = result.get("threat_fusion", {})
            vca = result.get("voice_clone_analysis", {})
            print(
                f"[VoiceForensics] {self.call_sid}: Threat Fusion completed — "
                f"clone={vca.get('prediction')} ({vca.get('confidence')}%) | "
                f"fused={tf.get('final_risk_score')}% | risk={tf.get('risk_level')}"
            )
            print(
                f"[VoiceForensics] {self.call_sid}: Final Risk Score generated — "
                f"{tf.get('risk_level')} ({tf.get('final_risk_score')}%)"
            )

            # Build broadcast payload (type tag + full forensics result)
            payload = {
                "type": "forensics_result",
                "call_sid": self.call_sid,
                "source": "whatsapp_live_call",
                "captured_seconds": FORENSICS_CAPTURE_CHUNKS * CHUNK_DURATION_SEC,
                **{k: v for k, v in result.items()
                   if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
            }

            # Broadcast to: (1) dedicated forensics WS and (2) shared Voice Lab WS
            for ws in [ws for ws in [self._forensics_ws, self.frontend_ws] if ws is not None]:
                try:
                    await ws.send_text(json.dumps(payload))
                except Exception as ws_err:
                    print(f"[VoiceForensics] {self.call_sid}: WS broadcast error — {ws_err}")

        except Exception as e:
            print(f"[VoiceForensics] {self.call_sid}: Forensic analysis error — {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"[VoiceForensics] {self.call_sid}: Temp file deleted: {temp_path}")
                except Exception:
                    pass

    def set_forensics_ws(self, ws):
        """Attach a dedicated Voice Forensics frontend WebSocket to this call handler."""
        self._forensics_ws = ws
        print(f"[VoiceForensics] {self.call_sid}: Forensics WebSocket attached")


# ── Active handlers store ────────────────────────────────────────────────────
_handlers: Dict[str, TwilioStreamHandler] = {}


def get_or_create_handler(call_sid: str, vector_db=None, frontend_ws=None) -> TwilioStreamHandler:
    if call_sid not in _handlers:
        _handlers[call_sid] = TwilioStreamHandler(call_sid, vector_db, frontend_ws)
    return _handlers[call_sid]


def get_handler(call_sid: str) -> Optional[TwilioStreamHandler]:
    """Return an existing handler without creating a new one. Returns None if not found."""
    return _handlers.get(call_sid)


def remove_handler(call_sid: str):
    _handlers.pop(call_sid, None)
