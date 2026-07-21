import os
import json
import urllib.request
import urllib.error

def query_gemini(prompt: str, system_instruction: str = None) -> dict:
    """
    Query Gemini model using the REST API.
    Supports GEMINI_API_KEY or GOOGLE_API_KEY environment variables.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is configured.")

    # Using gemini-2.5-flash for speed and reliability
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [
                {"text": system_instruction}
            ]
        }
        
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        # 10 second timeout is plenty for a fast model like 1.5-flash
        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = response.read().decode("utf-8")
            data = json.loads(res_body)
            # Extract text from response candidates
            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text_response.strip())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e else ""
        raise RuntimeError(f"Gemini API Error: {e.code} - {error_body}")
    except Exception as e:
        raise RuntimeError(f"Gemini Request failed: {str(e)}")
