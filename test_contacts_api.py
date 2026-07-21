"""Test the contacts API endpoints directly against the running backend."""
import urllib.request, urllib.parse, urllib.error, json

BASE = "http://localhost:8000"

def call(method, path, data=None):
    url = BASE + path
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    if body:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode()}
    except Exception as e:
        return {"error": str(e)}

print("\n=== Contacts API Test ===\n")

# 1. GET /contacts (should be empty initially)
res = call("GET", "/contacts")
print(f"[GET /contacts]  => {res}")

# 2. POST add Mom
res = call("POST", "/contacts", {"name": "Mom", "phone": "+91 98765 00000"})
print(f"[POST add Mom]   => {res}")

# 3. POST add Dad
res = call("POST", "/contacts", {"name": "Dad", "phone": "+91 98765 43210"})
print(f"[POST add Dad]   => {res}")

# 4. GET again (should have 2)
res = call("GET", "/contacts")
print(f"[GET /contacts]  => {res}")

# 5. DELETE Mom
res = call("DELETE", "/contacts?phone=%2B91+98765+00000")
print(f"[DELETE Mom]     => {res}")

# 6. GET final
res = call("GET", "/contacts")
print(f"[GET /contacts]  => {res}")

print("\nDone. If you see contacts above, the backend API is working correctly.\n")
