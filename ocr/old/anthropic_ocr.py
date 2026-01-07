import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# -----------------------------
# LOAD ENV
# -----------------------------
env_path = Path(__file__).parent.parent / ".env"  # adjust if needed
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY)

# -----------------------------
# OCR FUNCTION
# -----------------------------
def claude_ocr(file_path, model="claude-2"):
    """
    Upload a PDF or image to Claude's API and get extracted text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1️⃣ Upload the file
    with open(file_path, "rb") as f:
        uploaded = client.beta.files.upload(
            file=(Path(file_path).name, f,
                  "application/pdf" if file_path.endswith(".pdf") else "image/jpeg")
        )

    file_id = uploaded.id
    print(f"Uploaded {file_path} with file_id={file_id}")

    # 2️⃣ Ask Claude to extract text via a completion
    prompt = f"""
    Please extract all text from the uploaded file with id {file_id}.
    Return the text only, no extra commentary.
    """

    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens_to_sample=2000
    )

    return response.completion.strip()


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    test_file = "./scans/playjpgscans/Part1_page_17.jpg"
    text = claude_ocr(test_file)
    print("\n--- EXTRACTED TEXT ---\n")
    print(text)
