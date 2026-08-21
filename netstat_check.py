import subprocess
out = subprocess.check_output("netstat -ano", shell=True).decode('utf-8', errors='ignore')
for line in out.splitlines():
    if 'LISTENING' in line or 'listening' in line or '8000' in line:
        print(line)
