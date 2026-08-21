"""
Email Detector – IMAP Integration + Email Header & Link Analysis
───────────────────────────────────────────────────────────────
Connects to real email inbox via IMAP and scans messages:
  1. Header analysis   (SPF, DKIM, DMARC, reply-to mismatch)
  2. Sender domain check (trust scaling, brand impersonation)
  3. Link extraction & heuristic checking
  4. Content routing   (passes body to text detector)
"""

import os
import re
import email
import email.utils
import imaplib
from typing import Dict, Any, List, Tuple
from email.header import decode_header

from core.threat_score import ThreatScore


# ── Email Header Analysis ───────────────────────────────────────────────────

FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "mail.com", "yandex.com", "zoho.com", "icloud.com",
    "tutanota.com", "gmx.com",
}


def analyze_headers(raw_email: str) -> Tuple[float, List[str]]:
    """Analyze email headers for spoofing indicators."""
    reasons = []
    score = 0.0

    try:
        msg = email.message_from_string(raw_email)

        # From / Reply-To mismatch
        from_addr = msg.get("From", "")
        reply_to = msg.get("Reply-To", "")
        if reply_to and from_addr:
            from_domain = from_addr.split("@")[-1].rstrip(">").lower()
            reply_domain = reply_to.split("@")[-1].rstrip(">").lower()
            if from_domain != reply_domain:
                score += 25
                reasons.append(f"Reply-To mismatch: From={from_domain}, Reply-To={reply_domain}")

        # SPF check
        received_spf = msg.get("Received-SPF", "").lower()
        if "fail" in received_spf:
            score += 30
            reasons.append("SPF check FAILED – sender domain spoofing likely")
        elif "softfail" in received_spf:
            score += 15
            reasons.append("SPF softfail – sender domain may be spoofed")

        # DKIM check
        dkim = msg.get("DKIM-Signature", "")
        auth_results = msg.get("Authentication-Results", "").lower()
        if "dkim=fail" in auth_results:
            score += 25
            reasons.append("DKIM signature FAILED – email may be forged")
        elif not dkim and not any(k in auth_results for k in ["dkim=pass", "dkim=none"]):
            score += 10
            reasons.append("No DKIM signature – cannot verify sender authenticity")

        # DMARC
        if "dmarc=fail" in auth_results:
            score += 20
            reasons.append("DMARC check FAILED")

        # X-Mailer / unusual sending tool
        x_mailer = msg.get("X-Mailer", "")
        if x_mailer and any(s in x_mailer.lower() for s in ["phpmailer", "swiftmailer", "mass", "bulk"]):
            score += 15
            reasons.append(f"Mass mailing tool detected: {x_mailer}")

    except Exception as e:
        reasons.append(f"Header analysis error: {str(e)[:60]}")

    return min(score, 100.0), reasons


def extract_email_body(raw_email: str) -> str:
    """Extract plain text body from raw email."""
    try:
        msg = email.message_from_string(raw_email)
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode("utf-8", errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode("utf-8", errors="replace")
    except Exception:
        pass
    return ""


def extract_urls_from_email(body: str) -> List[str]:
    """Extract all URLs from email body."""
    return re.findall(r"https?://[^\s<>\"']+", body)


# ── IMAP Fetcher ────────────────────────────────────────────────────────────

class IMAPFetcher:
    """Connect to IMAP inbox and fetch recent emails for scanning."""

    def __init__(self, host: str, email_addr: str, password: str, port: int = 993):
        self.host = host
        self.email_addr = email_addr
        self.password = password
        self.port = port

    def fetch_recent(self, folder: str = "INBOX", count: int = 10) -> List[Dict]:
        """Fetch the N most recent emails."""
        results = []
        try:
            mail = imaplib.IMAP4_SSL(self.host, self.port)
            mail.login(self.email_addr, self.password)
            mail.select(folder)

            _, data = mail.search(None, "ALL")
            email_ids = data[0].split()
            recent_ids = email_ids[-count:] if len(email_ids) > count else email_ids

            for eid in reversed(recent_ids):
                _, msg_data = mail.fetch(eid, "(RFC822)")
                raw = msg_data[0][1].decode("utf-8", errors="replace")
                msg = email.message_from_string(raw)

                subject = ""
                raw_subject = msg.get("Subject", "")
                if raw_subject:
                    decoded = decode_header(raw_subject)
                    subject = decoded[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(decoded[0][1] or "utf-8", errors="replace")

                results.append({
                    "id": eid.decode(),
                    "from": msg.get("From", ""),
                    "subject": subject,
                    "date": msg.get("Date", ""),
                    "raw": raw,
                    "body": extract_email_body(raw),
                })

            mail.logout()
        except Exception as e:
            results.append({"error": str(e)[:120]})

        return results


# ── Main Detector ───────────────────────────────────────────────────────────

class EmailDetector:
    def analyze_raw(self, raw_email: str) -> Dict[str, Any]:
        """Analyze a single raw email (RFC822 format)."""
        reasons = []

        # Parse sender details
        msg = email.message_from_string(raw_email)
        from_header = msg.get("From", "")
        display_name, sender_email = email.utils.parseaddr(from_header)
        sender_email = sender_email.strip().lower()
        sender_domain = sender_email.split("@")[-1] if "@" in sender_email else ""

        # Validate header security authentication states
        received_spf = msg.get("Received-SPF", "").lower()
        auth_results = msg.get("Authentication-Results", "").lower()
        has_spf_fail = "fail" in received_spf
        has_dkim_fail = "dkim=fail" in auth_results
        has_dmarc_fail = "dmarc=fail" in auth_results

        # Set of verified known high-reputation sender brand domains
        trusted_domains = {
            "google.com", "microsoft.com", "apple.com", "netflix.com",
            "paypal.com", "amazon.com", "github.com", "linkedin.com",
            "twitter.com", "googlemail.com", "zoom.us", "meet.google.com",
            "zoom.com", "netflix.net", "spotify.com"
        }

        # Authenticated brand trust check
        is_authenticated_legit = False
        domain_matched = False
        for td in trusted_domains:
            if sender_domain == td or sender_domain.endswith(f".{td}"):
                domain_matched = True
                break
        if domain_matched and not (has_spf_fail or has_dkim_fail or has_dmarc_fail):
            is_authenticated_legit = True

        # Phishing impersonation: Official display name on personal/free domain type
        brand_keywords = {"google", "microsoft", "paypal", "netflix", "amazon", "apple", "bank", "security", "support", "admin"}
        has_brand_impersonation = False
        if sender_domain in FREE_EMAIL_PROVIDERS:
            for kw in brand_keywords:
                if kw in display_name.lower() or kw in sender_email:
                    has_brand_impersonation = True
                    break

        # Header vulnerability assessment
        header_score, header_reasons = analyze_headers(raw_email)
        reasons.extend(header_reasons)

        body = extract_email_body(raw_email)
        urls = extract_urls_from_email(body)

        # Scans links heuristic parameters
        url_threat_score = 0.0
        url_reasons = []
        from detectors.url_detector import heuristic_checks
        for url in urls:
            u_score, u_reasons = heuristic_checks(url)
            if u_score > url_threat_score:
                url_threat_score = u_score
                url_reasons = u_reasons

            # Phishing check: url redirect destination mismatches authenticated sender
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(url if url.startswith("http") else f"http://{url}")
                url_domain = parsed_url.netloc.lower()
                if url_domain.startswith("www."):
                    url_domain = url_domain[4:]

                if sender_domain and sender_domain != "unknown" and sender_domain not in FREE_EMAIL_PROVIDERS:
                    if url_domain != sender_domain and not url_domain.endswith(f".{sender_domain}"):
                        common_hosts = {"google.com", "youtube.com", "twitter.com", "facebook.com", "github.com", "linkedin.com"}
                        if not any(h in url_domain for h in common_hosts):
                            url_threat_score = max(url_threat_score, 65.0)
                            reasons.append(
                                f"Phishing destination mismatch: Email claim: '{sender_domain}', "
                                f"but links point to unverified domain: '{url_domain}'"
                            )
            except Exception:
                pass

        # Run local XGBoost NLP model on body
        local_ml_score = 0.0
        if body:
            from core.local_ml_model import predict_local_scam_probability
            local_ml_score = predict_local_scam_probability(body)

            # Apply 40% reduction filter: safe state if no links or clickable attachments are in the email
            if not urls:
                local_ml_score *= 0.60
                reasons.append("Email contains no hyperlink attachments or targets – scaling down risk base")

            # Apply 90% trust discount if authenticated brand
            if is_authenticated_legit:
                local_ml_score *= 0.10
                reasons.append(f"Legitimate sender domain '{sender_domain}' matches authenticated whitelists – applying 90% trust offset")

            if local_ml_score > 30:
                reasons.append(f"Local ML (XGBoost) flags spam body pattern: {local_ml_score:.1f}% confidence")

        if urls:
            reasons.append(f"Scan parsed {len(urls)} link(s) in email contents")
            for ur in url_reasons:
                reasons.append(f"[Link Analysis] {ur}")

        # Impersonation scoring
        impersonation_score = 0.0
        if has_brand_impersonation:
            impersonation_score = 55.0
            reasons.append(f"Impression Flag: Official display name '{display_name}' sent from personal free account domain '{sender_domain}'")

        # Synthesize final scoring index
        final = max(header_score, local_ml_score, url_threat_score, impersonation_score)

        # Clear score to 0 if fully verified and flags remain minimal
        if is_authenticated_legit and final < 45.0:
            final = 0.0

        return ThreatScore.build(
            score=final,
            reasons=reasons,
            source="email",
            raw={
                "urls_found": urls[:5],
                "body_length": len(body),
                "header_score": round(header_score, 1),
                "local_ml_score": round(local_ml_score, 1),
                "link_threat_score": round(url_threat_score, 1),
                "sender_domain": sender_domain,
                "is_authenticated_legit": is_authenticated_legit,
            },
        )

    def analyze_body(self, body: str, sender: str = "unknown") -> Dict[str, Any]:
        """Analyze just the email body text (when raw headers unavailable)."""
        reasons = []

        # Parse sender domain
        _, sender_email = email.utils.parseaddr(sender)
        sender_email = sender_email.strip().lower()
        sender_domain = sender_email.split("@")[-1] if "@" in sender_email else ""

        is_free_provider = sender_domain in FREE_EMAIL_PROVIDERS
        trusted_domains = {
            "google.com", "microsoft.com", "apple.com", "netflix.com",
            "paypal.com", "amazon.com", "github.com", "linkedin.com", "spotify.com"
        }

        urls = extract_urls_from_email(body)

        from core.local_ml_model import predict_local_scam_probability
        local_ml_score = predict_local_scam_probability(body)

        # Reduce risk if no clickable links are present
        if not urls:
            local_ml_score *= 0.60
            reasons.append("Email contains no hyperlink attachments or targets – scaling down risk base")

        # Scaled trust adjustment
        domain_matched = False
        for td in trusted_domains:
            if sender_domain == td or sender_domain.endswith(f".{td}"):
                domain_matched = True
                break
        if domain_matched:
            local_ml_score *= 0.20
            reasons.append(f"Sender email domain '{sender_domain}' matches trusted brand template – applying safety discount")

        url_threat_score = 0.0
        from detectors.url_detector import heuristic_checks
        for url in urls:
            u_score, u_reasons = heuristic_checks(url)
            if u_score > url_threat_score:
                url_threat_score = u_score
            for ur in u_reasons:
                reasons.append(f"[Link Analysis] {ur}")

        # Phishing check
        if urls and sender_domain and not is_free_provider:
            for url in urls:
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url if url.startswith("http") else f"http://{url}")
                    url_domain = parsed_url.netloc.lower()
                    if url_domain.startswith("www."):
                        url_domain = url_domain[4:]
                    if url_domain != sender_domain and not url_domain.endswith(f".{sender_domain}"):
                        common_hosts = {"google.com", "youtube.com", "twitter.com", "facebook.com", "github.com", "linkedin.com"}
                        if not any(h in url_domain for h in common_hosts):
                            url_threat_score = max(url_threat_score, 65.0)
                            reasons.append(f"[Link Alert] Email claims sender '{sender_domain}', but links pointing to unverified domain: '{url_domain}'")
                except Exception:
                    pass

        final = max(local_ml_score, url_threat_score)

        return ThreatScore.build(
            score=final,
            reasons=reasons,
            source="email",
            raw={
                "sender": sender,
                "body_length": len(body),
                "local_ml_score": round(local_ml_score, 1),
                "link_threat_score": round(url_threat_score, 1),
            },
        )
