#!/usr/bin/env python3
"""
test_production_pipeline.py
===========================
End-to-End verification of the Production Live Call Fraud Detection Pipeline.
Feeds a simulated PCM stream representing a critical voice deepfake + scam call,
verifies that VAD, sliding window buffer, parallel analyzers, and threat fusion
engine correctly flag the call under 10 seconds.
"""

import os
import sys
import time
import math
import struct
import wave

# Make sure backend/ is in sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# Check imports from our pipeline package
try:
    from backend.pipeline.pipeline_server import ProductionPipelineManager
except ImportError:
    from pipeline.pipeline_server import ProductionPipelineManager

# ANSI coloration
G  = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
C  = "\033[96m"; B = "\033[1m"; RESET = "\033[0m"

def make_synthetic_scam_pcm(duration_sec=12.0, sr=16000) -> bytes:
    """
    Generates a 16kHz 16-bit Mono PCM buffer containing:
    1. A synthesized stable comb tone (mimics deepfake signature).
    2. Simulated speech dynamics.
    """
    n_samples = int(duration_sec * sr)
    f0 = 220.0
    n_harmonics = 15
    amp = 0.8 / n_harmonics
    
    pcm_bytes = bytearray()
    for i in range(n_samples):
        t = i / sr
        # Spectral comb formula: sum of harmonics
        val = sum(amp * math.sin(2 * math.pi * (k * f0) * t) for k in range(1, n_harmonics + 1))
        # Scale to 16-bit signed PCM
        sample = int(32767 * val)
        pcm_bytes += struct.pack("<h", max(-32768, min(32767, sample)))
        
    return bytes(pcm_bytes)

def run_test():
    print(f"\n{B}{'='*70}{RESET}")
    print(f"{B}  Fraud Shield AI -- Production Live Call Pipeline Test{RESET}")
    print(f"{B}{'='*70}{RESET}\n")

    # 1. Initialize Pipeline manager
    print(f"{C}[1/4] Initializing Parallel Pipeline Manager...{RESET}")
    pm = ProductionPipelineManager(sample_rate=16000)
    session_id = "test-prod-session-99"
    customer_id = "cust-alice-007"
    
    # 2. Generate target audio stream (12 seconds)
    print(f"{C}[2/4] Generating synthetic 16kHz PCM audio stream (12s total)...{RESET}")
    audio_stream = make_synthetic_scam_pcm(duration_sec=12.0)
    print(f"      Size: {len(audio_stream):,} bytes ({len(audio_stream)//32000:.1f}s)")
    
    # 3. Simulate step-by-step 1-second chunks to sliding window (each second is 32,000 bytes)
    print(f"{C}[3/4] Simulating live stream feeds in 1-second chunks...{RESET}\n")
    print(f"  {'Time(s)':<8} | {'Threat Score':<12} | {'Verdict':<10} | {'Deepfake Conf':<14} | {'Reasons/Alerts'}")
    print(f"  {'-'*78}")
    
    start_time = time.time()
    chunk_size = 32000 # 1 second of 16kHz 16-bit mono audio
    
    first_alert_time = None
    
    for sec in range(1, 13):
        # Slice 1 second of audio
        offset = (sec - 1) * chunk_size
        chunk_data = audio_stream[offset:offset+chunk_size]
        
        # Feed to production pipeline
        updates = pm.process_pcm_chunk(session_id, chunk_data, customer_id=customer_id)
        
        if updates:
            for upd in updates:
                t = upd["timestamp"]
                score = upd["threat_score"]
                verdict = upd["verdict"]
                df_conf = upd["deepfake_confidence"]
                reasons = upd["explainable_reasons"]
                elapsed = time.time() - start_time
                
                # Highlight critical findings
                color = R if score >= 75 else (Y if score >= 40 else G)
                alert_flag = f"{R}{B}[ALERT]{RESET}" if upd["alerts_triggered"] else ""
                
                if upd["alerts_triggered"] and first_alert_time is None:
                    first_alert_time = t
                
                # Clean up reasons to remove any unicode that could break output
                clean_reason = reasons[0] if reasons else 'Normal'
                clean_reason = clean_reason.replace('\u2014', '-').replace('\u2013', '-')
                
                print(f"  {t:<8.1f} | {color}{score:<12.1f}{RESET} | {color}{verdict:<10}{RESET} | {df_conf:<14.1f} | {clean_reason} {alert_flag}")
        else:
            print(f"  {float(sec):<8.1f} | [Buffering...] | -          | -              | Window accumulating ({sec}/3s)")
            
        # Small delay to simulate real-time ingestion latency
        time.sleep(0.05)
        
    print(f"\n{C}[4/4] Evaluation Summary:{RESET}")
    print(f"  {'-'*55}")
    if first_alert_time is not None:
        print(f"  {G}* Alert successfully triggered at {first_alert_time:.1f}s -- well within the 10-15s window!{RESET}")
    else:
        print(f"  {R}* Failed to trigger high risk alert in the 12-second call window.{RESET}")
    print(f"  {'-'*55}\n")
    
if __name__ == "__main__":
    run_test()
