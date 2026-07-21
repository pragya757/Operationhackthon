"""
Threat Fusion Engine
════════════════════
Performs confidence-weighted scoring to synthesize a single dynamic risk score
from 6 vital dimensions:
1. Deepfake confidence
2. Speaker authenticity
3. Acoustic anomaly score
4. Scam intent confidence
5. Conversation behavior
6. Historical reputation
"""

import os
import numpy as np

class ThreatFusionEngine:
    def __init__(self):
        # Boost NLP + behavioral since those are most reliable in real-time voice calls
        self.weights = {
            "deepfake":   0.15,
            "speaker":    0.10,
            "acoustic":   0.10,
            "nlp":        0.35,   # primary signal
            "behavioral": 0.25,   # keyword coercion patterns
            "reputation": 0.05
        }
        self.llm_explanation_cache = {}

        # ── Critical override keywords (instant HIGH RISK if ANY found) ───────
        # Score pushed to ≥88 regardless of other signals
        self.critical_keywords = [
            "otp", "one time password", "pin", "cvv", "card number",
            "account number", "net banking", "fraud", "fraud message",
            "anydesk", "teamviewer", "remote access", "screen share",
            "transfer money", "send money", "wire transfer",
            "arrest", "cbi", "eci", "judiciary", "customs officer",
            "digital arrest", "warrant", "court order",
            "aadhaar number", "pan number", "passport number",
            "loan approved", "emi waiver", "processing fee",
            "kyc expired", "kyc verification", "account blocked",
            "lottery winner", "prize money", "lucky draw", "claim prize",
            "verify your identity", "confirms your details",
            "do not tell anyone", "keep this call secret",
            "upi pin", "scan qr", "gpay request", "phonepe",
            "hospital deposit", "emergency transfer", "accident",
        ]

        # ── High-risk behavioral markers (moderate score boost) ───────────────
        self.high_risk_phrases = [
            "urgently", "immediately", "right now", "warna", "do not hang up",
            "stay on the line", "do not disconnect", "supervisor",
            "last warning", "legal action", "government officer",
            "just a small fee", "nominal charges", "refund process",
            "call from bank", "rbi officer", "sebi officer",
            "your package", "drug package", "suspicious parcel",
            "your son", "your daughter", "accident hospital",
            "kidnapped", "need money now", "send immediately",
        ]

    def fuse(self, analysis_results: dict) -> dict:
        """
        Synthesizes scores from parallel detectors using confidence-weighted scaling.
        Yields threat score between 0.0 and 100.0.
        """
        reasons = []
        scores = []
        confidences = []
        weights = []

        transcript_text = analysis_results.get("transcript_segment", "").lower()

        # ── 1. Deepfake spoofing (AASIST) ─────────────────────────────────────
        df_res = analysis_results.get("deepfake", {})
        df_conf = df_res.get("confidence", 0.0)
        c_df = df_conf / 100.0
        scores.append(df_conf)
        confidences.append(c_df)
        weights.append(self.weights["deepfake"])
        for det in df_res.get("details", []):
            reasons.append(f"[AASIST Deepfake] {det}")

        # ── 2. Speaker verification (ECAPA-TDNN) ──────────────────────────────
        sp_res = analysis_results.get("speaker", {})
        similarity = sp_res.get("match_confidence", 1.0)
        s_sp = (1.0 - similarity) * 100.0 if not sp_res.get("verified", True) else 0.0
        c_sp = 0.8
        scores.append(s_sp)
        confidences.append(c_sp)
        weights.append(self.weights["speaker"])
        if s_sp > 25:
            reasons.append(f"[ECAPA-TDNN Speaker] {sp_res.get('details')}")

        # ── 3. Acoustic anomaly score ──────────────────────────────────────────
        ac_res = analysis_results.get("acoustic", {})
        s_ac = 0.0
        c_ac = 0.5
        pitch_std = ac_res.get("pitch_std", 0.0)
        rms_std = ac_res.get("rms_std", 0.0)
        rms_mean = ac_res.get("rms_mean", 1e-5)
        hnr = ac_res.get("hnr", 15.0)

        if pitch_std > 150:
            s_ac += 35
            reasons.append(f"[Acoustic] High F0 pitch volatility ({pitch_std:.1f} Hz) — vocal agitation")
        if hnr < 8.0:
            s_ac += 35
            reasons.append(f"[Acoustic] Low HNR ({hnr:.2f} dB) — robotic or synthetic voice artifacts")
        if rms_std / max(1e-5, rms_mean) > 0.8:
            s_ac += 30
            reasons.append("[Acoustic] Extreme amplitude modulation — coercive pressure dynamics")

        s_ac = min(100.0, s_ac)
        scores.append(s_ac)
        confidences.append(c_ac)
        weights.append(self.weights["acoustic"])

        # ── 4. Semantic NLP intent (3-layer: keywords + MiniLM + DistilBERT) ──
        nlp_res = analysis_results.get("nlp", {})
        s_nlp = nlp_res.get("confidence", 0.0)
        c_nlp = 0.95 if nlp_res.get("is_scam", False) else 0.50
        scores.append(s_nlp)
        confidences.append(c_nlp)
        weights.append(self.weights["nlp"])
        if nlp_res.get("is_scam", False):
            cat = nlp_res.get("category", "Unknown")
            reasons.append(f"[NLP] {cat} detected — {nlp_res.get('reasoning', '')}")

        # ── 5. Behavioral / keyword coercion analysis ─────────────────────────
        s_beh = 0.0
        c_beh = 0.90
        hit_critical = []
        hit_high_risk = []

        for kw in self.critical_keywords:
            if kw in transcript_text:
                hit_critical.append(kw)

        for phrase in self.high_risk_phrases:
            if phrase in transcript_text:
                hit_high_risk.append(phrase)

        if hit_critical:
            s_beh = min(100.0, 60.0 + len(hit_critical) * 15.0)
            kw_str = ", ".join('"' + k + '"' for k in hit_critical[:5])
            reasons.append(f"[Behavior] Critical fraud vocabulary detected: {kw_str}")
        if hit_high_risk:
            s_beh = min(100.0, s_beh + len(hit_high_risk) * 10.0)
            ph_str = ", ".join('"' + p + '"' for p in hit_high_risk[:4])
            reasons.append(f"[Behavior] High-risk coercion phrases: {ph_str}")

        scores.append(s_beh)
        confidences.append(c_beh)
        weights.append(self.weights["behavioral"])

        # ── 6. Historical reputation ───────────────────────────────────────────
        reputation_risk = float(analysis_results.get("reputation_risk", 0.0))
        c_rep = 0.7
        scores.append(reputation_risk)
        confidences.append(c_rep)
        weights.append(self.weights["reputation"])
        if reputation_risk > 50.0:
            reasons.append(f"[Reputation] Caller in spam database: {reputation_risk:.0f}% risk score")

        # ── Confidence-Weighted Fusion ─────────────────────────────────────────
        numerator = sum(s * c * w for s, c, w in zip(scores, confidences, weights))
        denominator = sum(c * w for c, w in zip(confidences, weights))
        fused_score = (numerator / denominator) if denominator > 0 else 0.0

        # ── Critical Overrides ─────────────────────────────────────────────────
        if df_conf > 70.0:
            fused_score = max(fused_score, 88.0)
            reasons.insert(0, "[CRITICAL] AASIST model: synthetic voice clone pattern confirmed")

        if not sp_res.get("verified", True) and similarity < 0.4 and customer_id_present_check(analysis_results):
            fused_score = max(fused_score, 82.0)
            reasons.insert(0, "[CRITICAL] ECAPA-TDNN biometric mismatch — high-confidence impersonation")

        # Extract local ML probability and structured scam flags for security context verification
        nlp_layers = nlp_res.get("layers", {})
        layer4_probs = nlp_layers.get("layer4", {}) if isinstance(nlp_layers, dict) else {}
        local_ml_prob = layer4_probs.get("prediction_score", 0.0) if isinstance(layer4_probs, dict) else 0.0
        nlp_is_scam = nlp_res.get("is_scam", False)

        # Context safety negations check (prevents triggering on "Do not share OTP" alert notifications)
        negations = [
            "do not share", "never share", "don't share", "not share", 
            "never tell", "do not tell", "don't tell", 
            "never give", "do not give", "don't give", 
            "never disclose", "do not disclose", "don't disclose",
            "never ask", "not ask", "don't ask", "do not ask",
            "never request", "not request", "don't request", "do not request",
            "never demand", "not demand", "don't demand", "do not demand", "never demands"
        ]
        is_negated = any(neg in transcript_text for neg in negations)

        # Extraction/coercion cues checking
        coercion_words = [
            "share", "send", "tell", "read", "give", "transfer", "verify", "enter",
            "type", "scan", "urgently", "immediately", "warn", "block", "cancel",
            "mandatory", "confirm", "disclose", "authorization", "pay", "deposit",
            "approved", "waive", "liquidate"
        ]
        has_coercion_cue = any(cw in transcript_text for cw in coercion_words)

        # Allow critical keyword overrides only if there is a suspicious context backing it
        # (meaning the NLP is flagged as a scam, the local ML model indicates a threat probability of at least 45%,
        # or a physical coercion verb/pressure cue is detected alongside the sensitive keyword, and NO negation is present)
        is_context_suspicious = (not is_negated) and (nlp_is_scam or local_ml_prob >= 45.0 or has_coercion_cue)

        if is_context_suspicious:
            # Per-keyword critical overrides — each critical keyword alone pushes score significantly
            if "otp" in transcript_text or "one time password" in transcript_text:
                fused_score = max(fused_score, 88.0)
                reasons.insert(0, "[CRITICAL OVERRIDE] OTP solicitation detected — classic credential harvesting attack")

            if any(w in transcript_text for w in ["anydesk", "teamviewer", "remote access", "screen share"]):
                fused_score = max(fused_score, 92.0)
                reasons.insert(0, "[CRITICAL OVERRIDE] Remote access tool request — active device takeover attempt")

            if any(w in transcript_text for w in ["arrest", "digital arrest", "cbi", "customs officer", "court order", "warrant"]):
                fused_score = max(fused_score, 90.0)
                reasons.insert(0, "[CRITICAL OVERRIDE] Government/law enforcement impersonation — Digital Arrest scam pattern")

            if any(w in transcript_text for w in ["fraud", "fraud message", "cvv", "card number", "account number"]):
                fused_score = max(fused_score, 88.0)
                reasons.insert(0, "[CRITICAL OVERRIDE] Banking fraud credential request detected")

            if any(w in transcript_text for w in ["lottery", "prize money", "lucky draw", "winner"]):
                fused_score = max(fused_score, 85.0)
                reasons.insert(0, "[CRITICAL OVERRIDE] Lottery/prize scam pattern — advance fee fraud")

            if any(w in transcript_text for w in ["upi pin", "scan qr", "gpay", "phonepe"]):
                fused_score = max(fused_score, 87.0)
                reasons.insert(0, "[CRITICAL OVERRIDE] UPI/payment scam pattern — reverse payment trick")

            if any(w in transcript_text for w in ["aadhaar", "pan card", "kyc", "kyc expired"]):
                fused_score = max(fused_score, 85.0)
                reasons.insert(0, "[CRITICAL OVERRIDE] KYC/identity document extraction attempt")

        # Layer 4: LLM enrichment when risk > 60%
        if fused_score > 60.0 and nlp_res.get("category"):
            category = nlp_res["category"]
            explanation = ""
            if category in self.llm_explanation_cache:
                explanation = self.llm_explanation_cache[category]
            else:
                try:
                    from pipeline.parallel_analyzer import StreamingFraudNLP
                    nlp_helper = StreamingFraudNLP()
                    explanation = nlp_helper.get_llm_explanation_sync(transcript_text, category)
                    self.llm_explanation_cache[category] = explanation
                except Exception:
                    explanation = ""
            if explanation:
                reasons.append(f"[AI Advisory] {explanation}")

        fused_score = min(100.0, fused_score)

        # ── Classification Band ────────────────────────────────────────────────
        if fused_score >= 80.0:
            classification = "CRITICAL"
            severity = "HIGH"
        elif fused_score >= 60.0:
            classification = "HIGH RISK"
            severity = "MEDIUM_HIGH"
        elif fused_score >= 35.0:
            classification = "MEDIUM RISK"
            severity = "MEDIUM"
        else:
            classification = "SAFE"
            severity = "LOW"

        return {
            "threat_score": round(fused_score, 1),
            "verdict": classification,
            "severity": severity,
            "deepfake_confidence": round(df_conf, 1),
            "scam_intent_confidence": round(s_nlp, 1),
            "explainable_reasons": reasons,
            "alerts_triggered": fused_score >= 65.0
        }


def customer_id_present_check(analysis_results: dict) -> bool:
    sp_res = analysis_results.get("speaker", {})
    return "enrolled" in sp_res.get("details", "").lower() or "biometric" in sp_res.get("details", "").lower()
