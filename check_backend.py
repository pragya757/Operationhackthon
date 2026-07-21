import os
checks = [
    (r'backend\pipeline\sliding_window.py', 'window_size_sec: float = 1.5'),
    (r'backend\pipeline\parallel_analyzer.py', 'vad_filter=True'),
    (r'backend\pipeline\pipeline_server.py', '"type": "transcript_ready"'),
]
for fpath, needle in checks:
    with open(fpath, encoding='utf-8') as f:
        found = needle in f.read()
    print(f"  {'OK' if found else 'FAIL'} [{fpath}] {needle!r}")
