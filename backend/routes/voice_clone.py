"""
Voice Clone Router – API Endpoint for Voice Forensics
────────────────────────────────────────────────────
Exposes POST /api/voice-clone/analyze.
Validates file extension and size (max 50 MB), handles temporary file creation,
and ensures automatic cleanup.
"""

import os
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from detectors.voice_clone_detector import analyze_audio

router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".webm", ".ogg", ".mp4"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

@router.post("/analyze")
async def analyze_voice_clone_route(file: UploadFile = File(...)):
    # 1. Validate file extension
    filename = file.filename or "audio.wav"
    _, ext = os.path.splitext(filename.lower())

    # Debug: Step 3 – log exactly what the frontend sent
    print(f"[VoiceCloneRoute] Uploaded File Name: {filename}")
    print(f"[VoiceCloneRoute] Uploaded File Type (MIME): {file.content_type}")
    print(f"[VoiceCloneRoute] Detected Extension: {ext}")

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    temp_path = None
    try:
        # 2. Save uploaded data into a temporary file
        # Use correct extension suffix so librosa handles formats correctly
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            temp_path = tmp.name
            
            total_bytes = 0
            while True:
                chunk = await file.read(1024 * 1024)  # Read 1 MB chunk
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Uploaded file size exceeds maximum limit of 50 MB."
                    )
                tmp.write(chunk)

        # 3. Call the exposed internal method directly
        result = analyze_audio(temp_path)
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[VoiceCloneRoute] Analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal audio forensics engine error: {str(e)}"
        )
    finally:
        # 4. Enforce strict temporary file cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"[VoiceCloneRoute] Temporary file deleted successfully: {temp_path}")
            except Exception as ce:
                print(f"[VoiceCloneRoute] Failed to delete temporary file {temp_path}: {ce}")
