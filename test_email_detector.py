import os
import sys

# Ensure backend folder is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from detectors.email_detector import EmailDetector

def test_email_scanning():
    print("=" * 70)
    print("   Fraud Shield AI -- Email Scanning and Whitelisting test")
    print("=" * 70)

    detector = EmailDetector()

    # 1. Legitimate Whitelisted Brand Email (contains potentially flagged warning words)
    legit_email = (
        "From: Google Accounts Team <no-reply@accounts.google.com>\n"
        "Subject: Critical Security alert - verify your profile\n"
        "Received-SPF: pass (google.com: Sender is authorized)\n"
        "Authentication-Results: mx.google.com; dkim=pass header.i=@google.com;\n"
        "DKIM-Signature: v=1; a=rsa-sha256; c=relaxed/relaxed; d=google.com;\n"
        "Content-Type: text/plain; charset=UTF-8\n\n"
        "Hello User, we detected a new login device on your Google account. "
        "Please open your account security console at https://google.com/security "
        "to confirm it was you. If this was not you, please verify your details immediately."
    )

    # 2. Phishing Email: Impersonation & domain link mismatch
    phish_email = (
        "From: Netflix Support <membership-renew@gmail.com>\n"
        "Subject: Your Netflix membership is suspended - verify payment details\n"
        "Content-Type: text/plain; charset=UTF-8\n\n"
        "Dear customer, your Netflix membership payment failed. "
        "To prevent service disruption, please verify your credit card details immediately "
        "by clicking this update link: http://netflix-billing-verify.top/login.html "
        "Please update now to resume streaming."
    )

    print("\n[1/2] Processing LEGITIMATE email (Google Security notice)...")
    legit_result = detector.analyze_raw(legit_email)
    print(f"  * Threat Score: {legit_result['score']}%")
    print(f"  * Verdict:      {legit_result['verdict']}")
    print("  * Reasons:")
    for r in legit_result["reasons"]:
        print(f"    - {r}")

    print("\n[2/2] Processing PHISHING email (Netflix Brand Impersonation + Mismatch URL)...")
    phish_result = detector.analyze_raw(phish_email)
    print(f"  * Threat Score: {phish_result['score']}%")
    print(f"  * Verdict:      {phish_result['verdict']}")
    print("  * Reasons:")
    for r in phish_result["reasons"]:
        print(f"    - {r}")

    # Assertions to ensure whitelist logic works correctly
    print("\n" + "-" * 70)
    assert legit_result["score"] < 45.0, f"Failed: Legit email score is too high: {legit_result['score']}%"
    assert phish_result["score"] >= 65.0, f"Failed: Phishing email not flagged: {phish_result['score']}%"
    print("SUCCESS: Whitelisting and Phishing/Link scanning is working perfectly!")
    print("-" * 70)

if __name__ == "__main__":
    test_email_scanning()
