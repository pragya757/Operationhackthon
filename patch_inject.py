"""Patch quick-inject buttons to auto-submit when clicked."""
path = r'frontend\src\app\voice\page.tsx'
with open(path, encoding='utf-8') as f:
    content = f.read()

# The quick-inject buttons just set text — we need them to also trigger submit
old = """onClick={() => { setManualText(txt); setManualSpeaker("Caller"); }}"""
new = """onClick={async () => {
                                      setManualSpeaker("Caller");
                                      setManualText(txt);
                                      // Use a synthetic submit after state settles
                                      setTimeout(() => {
                                        const form = document.getElementById("inject-form") as HTMLFormElement | null;
                                        form?.requestSubmit();
                                      }, 50);
                                    }}"""

if old in content:
    content = content.replace(old, new, 1)
    print("Quick-inject onClick patched.")
else:
    print("Target not found, skipping.")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Done.")
