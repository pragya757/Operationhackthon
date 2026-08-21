path = r'frontend\src\app\voice\page.tsx'
with open(path, encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'latestChunk' in line or 'chunks[chunks.length' in line:
        print(i+1, repr(line[:120]))
