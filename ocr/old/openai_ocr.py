from pathlib import Path
from dotenv import load_dotenv
import os
from PIL import Image
import base64
from io import BytesIO
import openai

# Load API key
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_DIR = Path("./scans/playjpgscans")
OUTPUT_DIR = Path("./ocr_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_WIDTH = 2000
MAX_HEIGHT = 3000

def ocr_page(image_path: Path):
    # Open and resize image
    img = Image.open(image_path)
    if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
        img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)

    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    # Construct input as required by Responses API
    payload = [
        {
            "type": "input_text",
            "text": "Extract all text from this handwritten German recipe image."
        },
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{img_b64}"
        }
    ]

    # Call OpenAI Responses API with vision/OCR
    response = openai.responses.create(
        model="gpt-4o-mini-ocr",
        input=payload
    )

    return response.output_text

# Process all pages
for page_file in sorted(INPUT_DIR.glob("*.jpg")):
    print(f"Processing {page_file.name}...")
    try:
        text = ocr_page(page_file)
        out_file = OUTPUT_DIR / f"{page_file.stem}.txt"
        out_file.write_text(text, encoding="utf-8")
        print(f"Saved OCR to {out_file}")
    except Exception as e:
        print(f"Failed on {page_file.name}: {e}")
