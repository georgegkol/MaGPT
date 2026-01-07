import os
import base64
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
import requests

# -----------------------------
# LOAD ENV
# -----------------------------
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# -----------------------------
# GEMINI REST CONFIG
# -----------------------------
# -----------------------------
# GEMINI REST CONFIG
# -----------------------------
MODEL = "models/gemini-1.5-flash"
ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent"
    f"?key={API_KEY}"
)


# -----------------------------
# OCR FUNCTION
# -----------------------------
def gemini_ocr(file_path: str) -> str:
    """
    OCR for JPG / PNG / PDF using Gemini REST API.
    Returns extracted text only.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    encoded = base64.b64encode(file_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Extract ALL text from this document exactly as written. "
                            "Preserve line breaks. Output text only."
                        )
                    },
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": encoded
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post(
        ENDPOINT,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text}"
        )

    data = response.json()

    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    test_file = "./scans/playjpgscans/Part1_page_17.jpg"
    text = gemini_ocr(test_file)

    print("\n--- EXTRACTED TEXT ---\n")
    print(text)
