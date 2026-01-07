import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# -----------------------------
# CONFIG
# -----------------------------
MIN_LINE_HEIGHT = 22       # reject tiny rows
LINE_PAD_Y = 12            # vertical padding
LINE_PAD_X = 20            # horizontal padding
UPSCALE_FACTOR = 1.5       # upscale cropped lines for better OCR

# -----------------------------
# LOAD TR-OCR (German model)
# -----------------------------
processor = TrOCRProcessor.from_pretrained("fhswf/TrOCR_german_handwritten")
model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_german_handwritten")
model.eval()

"""FINETUNED_MODEL_DIR = "./models/trocr_finetuned/checkpoint-504"

processor = TrOCRProcessor.from_pretrained(FINETUNED_MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(FINETUNED_MODEL_DIR)
model.eval()"""

# -----------------------------
# FUNCTIONS
# -----------------------------
def preprocess_image(gray):
    """Denoise entire page (optional)."""
    gray_denoised = cv2.fastNlMeansDenoising(gray, h=10)
    return gray_denoised

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

def visualize_rows(img, rows, out_path):
    """Draw green boxes around detected rows."""
    vis = img.copy()
    for y1, y2 in rows:
        cv2.rectangle(vis, (0, y1), (vis.shape[1], y2), (0, 255, 0), 2)
    cv2.imwrite(str(out_path), vis)


def enhance_line_for_ocr(line_gray):
    # 1) Light denoising only
    line_denoised = cv2.fastNlMeansDenoising(line_gray, h=7)

    # 2) Contrast normalization (very important)
    line_norm = cv2.normalize(
        line_denoised, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    # 3) Upscale (TrOCR likes bigger text)
    line_upscaled = cv2.resize(
        line_norm, None,
        fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC
    )

    return line_upscaled


def crop_and_ocr(gray, rows, out_dir):
    """Crop detected rows, enhance them, and run TrOCR."""
    results = []
    for idx, (y1, y2) in enumerate(rows):
        # vertical padding
        row_height = y2 - y1
        pad = int(row_height * 0.8)
        y1p = max(0, y1 - pad)
        y2p = min(gray.shape[0], y2 + pad)

        line = gray[y1p:y2p, :]

        if line.shape[0] < MIN_LINE_HEIGHT:
            continue

        # Enhance the cropped line
        line_enhanced = enhance_line_for_ocr(line)

        line_img = Image.fromarray(line_enhanced).convert("RGB")
        line_path = out_dir / f"line_{idx:03d}.png"
        line_img.save(line_path)

        # TrOCR
        pixel_values = processor(line_img, return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=4,
                length_penalty=1.0
            )

        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        results.append((line_path.name, text))

    return results

def process_image(image_path, out_dir):
    """Full pipeline for one image."""
    out_dir.mkdir(exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image at {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: denoise entire page
    gray = preprocess_image(gray)

    # Detect rows + visualize
    rows = detect_rows(gray)
    print(f"Detected {len(rows)} candidate rows")
    visualize_rows(img, rows, out_dir / "rows_visualized.jpg")

    # Crop + enhance + OCR
    results = crop_and_ocr(gray, rows, out_dir)
    return results

# -----------------------------
# RUN (single image)
# -----------------------------
IMAGE_PATH = "./scans/Part7/Part7_page_3.jpg"
OUT_DIR = Path("./scans/debug_lines")

results = process_image(IMAGE_PATH, OUT_DIR)

print("\n--- OCR RESULTS ---\n")
for name, text in results:
    print(f"{name}: {text}")
