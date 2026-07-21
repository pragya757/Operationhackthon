import sys
import os

# Limit CPU threads to 1 to avoid PyTorch multi-threaded unpickling/Extraction crashes (Access Violations) on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

# Make sure backend/ is in sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
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

# Test scenarios with expected labels: 0 for Safe, 1 for Scam/Fraud
test_dataset = [
    # ── Legitimate Banking Reminders & Common Transactions (Expected: 0) ──────────────────
    ("For your safety, the bank will never ask you to share your OTP or money. Keep your details safe.", 0),
    ("Do not disclose your transaction OTP, bank password, or credit card number to anyone. We will never ask for them.", 0),
    ("Dear customer, remember that your bank never demands your PIN, CVV, or card number online.", 0),
    ("Please do not share your OTP. We request you to ignore any calls asking for your security codes.", 0),
    ("Our standard banking application security policy says you should never share your password.", 0),
    ("I am returning the cash that I borrowed yesterday for my grocery shopping.", 0),
    ("Hello I am calling to confirm your appointment for tomorrow at 3pm.", 0),
    ("We would like to remind you to review your account statement today for any unexpected transactions.", 0),
    ("Please transfer the electricity bill payment to our society account.", 0),
    ("Can you send me the address of the restaurant where we are having dinner tonight?", 0),
    
    # ── Fraudulent Solicitation & Scams (Expected: 1) ──────────────────────────────────────
    ("Please share the OTP you received on your phone to complete your verification immediately otherwise we block.", 1),
    ("Give me the OTP code now, otherwise your ATM card and your bank account will be blocked today.", 1),
    ("Download AnyDesk for remote access so I can fix your computer virus right now", 1),
    ("This is CBI officer you are under digital arrest for money laundering court order issued", 1),
    ("Congratulations you have won lottery prize money pay processing fee to claim prize", 1),
    ("Your Aadhaar card has been linked to illegal activities, verify your identity immediately to avoid arrest.", 1),
    ("Your card is blocked due to suspicious activity, call us immediately to verify your CVV card number.", 1),
    ("KYC expired on your account, please send details to complete verification immediately or account will be suspended.", 1),
    ("Urgent: Scan this QR code now using GPay or PhonePe to receive your cash prize of ten thousand rupees.", 1),
    ("This is customs officer speaking, a package containing narcotics was found on your name, pay fine now.", 1)
]

print("="*100)
print("RUNNING PIPELINE ACCURACY BENCHMARK ON CONTEXT-AWARE SAFEGUARDS")
print("="*100)
print(f"{'Transcript Snippet':<50} | {'Expected':<8} | {'Predicted':<9} | {'NLP Conf':<8} | {'Fused':<6} | {'StatusValue':<8}")
print("-"*100)

correct_predictions = 0
false_positives = 0
false_negatives = 0
true_positives = 0
true_negatives = 0

for text, expected in test_dataset:
    # 1. Feed text into ASR output processor (NLP classifier)
    nlp_res = nlp.classify_transcript(text)
    
    # 2. Build mock output results for threat fusion engine
    analysis_results = {
        "nlp": nlp_res,
        "transcript_segment": text,
        "deepfake": {"confidence": 0.0, "is_deepfake": False},
        "speaker": {"verified": True, "match_confidence": 1.0, "details": "Enrolled match confirmed"},
        "acoustic": {"flatness_mean": 0.05, "zcr_std": 0.02, "mfcc_var": 12.0, "jitter": 5.0, "centroid_std": 300.0}
    }
    
    # 3. Process with Threat Fusion Engine
    fused_res = fusion.fuse(analysis_results)
    fused_score = fused_res.get("threat_score", 0.0)
    
    # Overall pipeline verdict: High / Critical risk (> 40.0) represents Scam (1)
    predicted = 1 if fused_score >= 40.0 else 0
    
    status_str = "SUCCESS"
    if predicted == expected:
        correct_predictions += 1
        if expected == 1:
            true_positives += 1
        else:
            true_negatives += 1
    else:
        status_str = "FAILED"
        if expected == 0 and predicted == 1:
            false_positives += 1
        else:
            false_negatives += 1
            
    snippet = text[:47] + "..." if len(text) > 47 else text
    expected_str = "Scam (1)" if expected == 1 else "Safe (0)"
    predicted_str = "Scam (1)" if predicted == 1 else "Safe (0)"
    
    print(f"{snippet:<50} | {expected_str:<8} | {predicted_str:<9} | {nlp_res.get('confidence', 0.0):<8.1f} | {fused_score:<6.1f} | {status_str:<8}")

total_cases = len(test_dataset)
accuracy = (correct_predictions / total_cases) * 100.0
precision = (true_positives / (true_positives + false_positives)) * 100.0 if (true_positives + false_positives) > 0 else 0.0
recall = (true_positives / (true_positives + false_negatives)) * 100.0 if (true_positives + false_negatives) > 0 else 0.0

print("="*100)
print("BENCHMARK METRICS SUMMARY")
print("="*100)
print(f"Total Test Cases Evaluated : {total_cases}")
print(f"Correct Predictions        : {correct_predictions}")
print(f"True Negatives (Legit)     : {true_negatives}")
print(f"True Positives (Scam)      : {true_positives}")
print(f"False Positives (FA)       : {false_positives} (Target check: Should be 0!)")
print(f"False Negatives (Missed)   : {false_negatives}")
print(f"Pipeline Test Accuracy     : {accuracy:.2f}%")
print(f"Pipeline Test Precision    : {precision:.2f}%")
print(f"Pipeline Test Recall       : {recall:.2f}%")
print("="*100)
