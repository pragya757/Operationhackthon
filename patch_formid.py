path = r'frontend\src\app\voice\page.tsx'
with open(path, encoding='utf-8') as f:
    content = f.read()

old = '<form onSubmit={injectManualTranscript} className="border-t border-outline/10 pt-3 mt-3 flex flex-col gap-2">'
new = '<form id="inject-form" onSubmit={injectManualTranscript} className="border-t border-outline/10 pt-3 mt-3 flex flex-col gap-2">'

if old in content:
    content = content.replace(old, new, 1)
    print("Form ID added.")
else:
    print("Form target not found.")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Done.")
