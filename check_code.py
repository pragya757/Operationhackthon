path = r'frontend\src\app\voice\page.tsx'
with open(path, encoding='utf-8') as f:
    content = f.read()
    lines = content.splitlines()

checks = {
    "handleCaptureConnect defined": "const handleCaptureConnect",
    "handleCaptureConnect used in button": "onClick={handleCaptureConnect}",
    "getDisplayMedia in handleCaptureConnect": "getDisplayMedia",
    "getUserMedia in handleCaptureConnect": "getUserMedia",
    "startAudioCapture takes streams": "startAudioCapture = async (\n    ws: WebSocket,\n    micStream",
    "connectWs takes streams": "captureAudio: boolean = false,\n    micStream: MediaStream",
    "No stray connectWs(true) in button": None,
}

print("\n=== Code Integrity Check ===\n")
all_ok = True
for label, needle in checks.items():
    if needle is None:
        # Special check: count connectWs(true) occurrences — should be 0 in JSX
        count = content.count("connectWs(true)")
        ok = (count == 0)
        print(f"  {'OK' if ok else 'FAIL'} {label} (found {count}x connectWs(true))")
    else:
        found = needle in content
        print(f"  {'OK' if found else 'FAIL'} {label}")
        if not found:
            all_ok = False

print()
if all_ok:
    print("All checks passed.")
else:
    print("Some checks FAILED — review above.")
