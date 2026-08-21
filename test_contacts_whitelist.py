#!/usr/bin/env python3
"""
test_contacts_whitelist.py
===========================
Verifies caller phone number detection and whitelist contact check.
Ensures whitelisted numbers immediately bypass analysis and return SAFE / score 0.
Ensures unverified numbers trigger live call analysis normally.
"""

import os
import sys
import json
import asyncio
from fastapi.testclient import TestClient

# Ensure root paths are in sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

try:
    from backend.main import app
except ImportError:
    from main import app

# Color definitions
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; C = "\033[96m"; B = "\033[1m"; RESET = "\033[0m"

def test_whitelist():
    print(f"\n{B}{'='*70}{RESET}")
    print(f"{B}  Testing Caller Phone Number Whitelisting & Bypass Pipeline{RESET}")
    print(f"{B}{'='*70}{RESET}\n")

    client = TestClient(app)

    # Test Case 1: Whitelisted number (+91 98765 43210 - "Dad")
    caller_whitelisted = "+91 98765 43210"
    print(f"{C}[1/2] Connecting to Production Live WS with Whitelisted Number: {caller_whitelisted}...{RESET}")
    
    with client.websocket_connect(f"/ws/production-live-call/test-call-123?caller_number={caller_whitelisted}") as ws:
        # 1. Verification frame should be received immediately
        data = ws.receive_json()
        print(f"      Initial packet: {G}{data}{RESET}")
        assert data["type"] == "contact_verified"
        assert data["name"] == "Dad"
        assert data["verdict"] == "SAFE"

        # 2. Sending bytes should immediately bypass scan returning score 0.0 without running model
        print("      Sending mock sound chunk...")
        ws.send_bytes(b"\x00" * 32000) # 1 second dummy PCM
        resp = ws.receive_json()
        print(f"      Response packet: {G}{resp}{RESET}")
        assert resp["type"] == "chunk_result"
        assert resp["threat_score"] == 0.0
        assert resp["is_saved_contact"] is True
        assert resp["contact_name"] == "Dad"
        assert "Forensic scan bypassed" in resp["explainable_reasons"][0]

    print(f"{G}      => Whitelisted call test passed!{RESET}\n")

    # Test Case 2: Unverified number (+1 999-999-9999)
    caller_unverified = "+1 999-999-9999"
    print(f"{C}[2/2] Connecting with Unverified Number: {caller_unverified}...{RESET}")
    
    with client.websocket_connect(f"/ws/production-live-call/test-call-456?caller_number={caller_unverified}") as ws:
        # Awaiting transcription or chunks. Standard numbers don't send verification packet first.
        print("      Sending mock sound chunk (unverified calls should evaluate speech normally)...")
        ws.send_bytes(b"\x00" * 32000) # 1 second dummy PCM
        
        # Receives standard chunk_result (might be quiet/empty but evaluated)
        resp = ws.receive_json()
        print(f"      Response packet: {Y}{resp}{RESET}")
        assert resp["type"] == "chunk_result"
        assert resp.get("is_saved_contact") is not True

    print(f"{G}      => Unverified call test passed!{RESET}\n")
    print(f"{B}{G}ALL TESTS PASSED SUCCESSFULLY! ✅{RESET}\n")

if __name__ == "__main__":
    test_whitelist()
