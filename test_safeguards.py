import sys
import os

# Make sure backend/ is in sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    # Inserts backend first so things like 'import pipeline' resolve to 'backend/pipeline'
    sys.path.insert(0, BACKEND)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from pipeline.parallel_analyzer import StreamingFraudNLP
    from pipeline.threat_fusion import ThreatFusionEngine
except ImportError:
    from backend.pipeline.parallel_analyzer import StreamingFraudNLP
    from backend.pipeline.threat_fusion import ThreatFusionEngine

nlp = StreamingFraudNLP()
fusion = ThreatFusionEngine()

test_cases = [
    # ── Legitimate/Banking Alert Scenarios (Should be flagged safe) ─────────────────
    "For your safety, the bank will never ask you to share your OTP or money. Keep your details safe.",
    "Do not disclose your transaction OTP, bank password, or credit card number to anyone. We will never ask for them.",
    "I am returning the cash that I borrowed yesterday for my grocery shopping.",
    "Our standard banking application security policy says you should never share your password.",
    
    # ── Fraudulent/Coercive Scenarios (Should be flagged scam) ──────────────────────
    "Please share the OTP you received on your phone to complete your verification immediately otherwise we block.",
    "Give me the OTP code now, otherwise your ATM card and your bank account will be blocked today.",
    "Download AnyDesk for remote access so I can fix your computer virus right now",
    "This is CBI officer you are under digital arrest for money laundering court order issued"
]

print("="*80)
print("TESTING CONTEXT-AWARE SAFEGUARDS & THREAT FUSION OVERRIDES")
print("="*80)
print(f"{'Transcript':<60} | {'Is Scam':<7} | {'NLP Conf':<8} | {'Fused Score':<11}")
print("-"*90)

for case in test_cases:
    print(f"\nDEBUG: Processing case: {case}", flush=True)
    # 1. NLP classier classification
    nlp_res = nlp.classify_transcript(case)
    print(f"DEBUG: classify_transcript completed. Result: {nlp_res}", flush=True)
    
    # Simulate threat fusion input
    analysis_results = {
        "nlp": nlp_res,
        "transcript_segment": case,
        "deepfake": {"confidence": 0.0, "is_deepfake": False},
        "speaker": {"verified": True, "match_confidence": 1.0, "details": "Enrolled match confirmed"},
        "acoustic": {"flatness_mean": 0.05, "zcr_std": 0.02, "mfcc_var": 12.0, "jitter": 5.0, "centroid_std": 300.0}
    }
    
    # Run threat fusion engine
    print("DEBUG: Calling fusion.fuse...", flush=True)
    fused_res = fusion.fuse(analysis_results)
    print(f"DEBUG: fusion.fuse completed. Result: {fused_res}", flush=True)
    
    is_scam = nlp_res.get("is_scam", False)
    nlp_conf = nlp_res.get("confidence", 0.0)
    fused_score = fused_res.get("threat_score", 0.0)
    
    # Shorten text for neat printing
    short_text = case[:57] + "..." if len(case) > 57 else case
    print(f"{short_text:<60} | {str(is_scam):<7} | {nlp_conf:<8.1f} | {fused_score:<11.1f}")
