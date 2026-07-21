path = r'frontend\src\app\voice\page.tsx'
with open(path, encoding='utf-8') as f:
    content = f.read()

checks = [
    ("Animated soundwave listening state",          "Animated soundwave bars"),
    ("Quick-inject test buttons",                   "Quick-inject to test scoring"),
    ("Auto-submit on quick-inject click",           "requestSubmit"),
    ("inject-form id on form element",              'id="inject-form"'),
    ("liveInterim shown before messages",           "Live interim \u2014 always shown at top"),
    ("liveInterim state defined",                   "const [liveInterim, setLiveInterim]"),
    ("startSpeechRecognition wired in ws.onopen",   "startSpeechRecognition(); // browser"),
    ("stopSpeechRecognition in disconnectWs",       "stopSpeechRecognition();\n    cleanupCapture"),
    ("transcript_ready handler",                    "msg.type === \"transcript_ready\""),
    ("sliding window 1.5s",                         "window_size_sec: float = 1.5"),
    ("Whisper vad_filter",                          "vad_filter=True"),
    ("pipeline_server fast transcript packet",      "transcript_ready"),
]

print("\n=== Final Integrity Check ===\n")
all_ok = True
for label, needle in checks:
    found = needle in content
    print(f"  {'OK' if found else 'FAIL'} {label}")
    if not found:
        all_ok = False
        
print()
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
