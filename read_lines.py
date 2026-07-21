f = open(r'frontend\src\app\voice\page.tsx', encoding='utf-8')
lines = f.readlines()
f.close()
for i in range(1126, 1137):
    print(i+1, repr(lines[i]))
