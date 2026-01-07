import os
from pathlib import Path
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# -----------------------------
# LOAD .ENV VARIABLES
# -----------------------------
load_dotenv()

GOOGLE_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
PROCESSOR_ID = os.getenv("PROCESSOR_ID")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_KEY_PATH

# -----------------------------
# INIT DOCUMENT AI CLIENT
# -----------------------------
client = documentai.DocumentProcessorServiceClient()
processor_name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

# -----------------------------
# IMAGE / ROW CONFIG
# -----------------------------
MIN_LINE_HEIGHT = 22
UPSCALE_FACTOR = 1.5

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(gray):
    """Denoise entire page (optional)."""
    return cv2.fastNlMeansDenoising(gray, h=10)

def detect_rows(gray):
    """Detect horizontal text rows using projection."""
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.dilate(bw, kernel, iterations=1)

    projection = bw.sum(axis=1).astype(np.float32)
    projection /= projection.max() + 1e-6
    projection = np.convolve(projection, np.ones(11)/11, mode="same")
    threshold = max(0.02, np.percentile(projection, 60))

    rows = []
    in_row = False
    start = 0
    for i, v in enumerate(projection):
        if v > threshold and not in_row:
            start = i
            in_row = True
        elif v <= threshold and in_row:
            end = i
            if end - start >= MIN_LINE_HEIGHT:
                rows.append((start, end))
            in_row = False
    if in_row:
        rows.append((start, len(projection)))

    return rows

def enhance_line(line_gray):
    """Enhance a cropped line for OCR."""
    line_denoised = cv2.fastNlMeansDenoising(line_gray, h=7)
    line_norm = cv2.normalize(line_denoised, None, 0, 255, cv2.NORM_MINMAX)
    line_upscaled = cv2.resize(line_norm, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                               interpolation=cv2.INTER_CUBIC)
    return line_upscaled

# -----------------------------
# DOCUMENT AI PROCESSING
# -----------------------------
def process_line_with_document_ai(line_array):
    """Send a single line (NumPy array) to Document AI and return extracted text."""
    pil_img = Image.fromarray(line_array).convert("RGB")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    line_bytes = buf.getvalue()

    document = {"content": line_bytes, "mime_type": "image/png"}
    request = {"name": processor_name, "raw_document": document}

    result = client.process_document(request=request)
    return result.document.text

# -----------------------------
# FULL PIPELINE
# -----------------------------
def process_image(image_path, debug_dir=None):
    debug_dir = Path(debug_dir or "./debug_lines")
    debug_dir.mkdir(exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = preprocess_image(gray)

    rows = detect_rows(gray)
    print(f"Detected {len(rows)} candidate rows")

    results = []
    for idx, (y1, y2) in enumerate(rows):
        row_height = y2 - y1
        pad = int(row_height * 0.8)
        y1p = max(0, y1 - pad)
        y2p = min(gray.shape[0], y2 + pad)
        line = gray[y1p:y2p, :]
        if line.shape[0] < MIN_LINE_HEIGHT:
            continue

        # Enhance
        line_enhanced = enhance_line(line)

        # Optional: save debug images
        debug_path = debug_dir / f"line_{idx:03d}.png"
        Image.fromarray(line_enhanced).convert("RGB").save(debug_path)

        # OCR via Document AI
        try:
            text = process_line_with_document_ai(line_enhanced)
        except Exception as e:
            text = f"[ERROR: {e}]"

        results.append((debug_path.name, text))

    return results

# -----------------------------
# RUN EXAMPLE
# -----------------------------
if __name__ == "__main__":
    IMAGE_PATH = "./scans/playjpgscans/Part1_page_17.jpg"  # replace with your image
    results = process_image(IMAGE_PATH, debug_dir="./scans/debug_lines")

    print("\n--- OCR RESULTS ---\n")
    for name, text in results:
        print(f"{name}: {text}")
