"""Patch page.tsx: wire primary capture button to handleCaptureConnect."""
path = r'frontend\src\app\voice\page.tsx'
with open(path, encoding='utf-8') as f:
    content = f.read()

# The old onClick line (line 1130)
old = "                          onClick={() => connectWs(true)}\n"
new = "                          onClick={handleCaptureConnect}\n"

if old not in content:
    print("ERROR: target text not found")
    # Show nearby content for debugging
    idx = content.find("connectWs(true)")
    print("Found 'connectWs(true)' at index:", idx)
    print("Context:", repr(content[max(0,idx-80):idx+80]))
else:
    content = content.replace(old, new, 1)

    # Also update button label text
    old_label = "                          <Mic className=\"w-4 h-4\" /> Connect & Capture Web Call\n"
    new_label  = "                          <Mic className=\"w-4 h-4\" /> Connect &amp; Capture Call Audio\n"
    if old_label in content:
        content = content.replace(old_label, new_label, 1)
        print("Label updated.")
    else:
        print("Label not found, skipping label update.")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Done. Button now calls handleCaptureConnect.")
