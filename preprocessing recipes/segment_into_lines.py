import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# -----------------------------
# CONFIG (FROM YOUR CODE)
# -----------------------------
MIN_LINE_HEIGHT = 22

PAGE_DIR = Path("scans/clean_handwriting")
OUT_DIR = Path("scans/lines_for_transcription")

# -----------------------------
# YOUR ROW DETECTION (UNCHANGED)
# -----------------------------
def detect_rows(gray):
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.dilate(bw, kernel, iterations=1)

    projection = bw.sum(axis=1).astype(np.float32)
    projection /= projection.max() + 1e-6
    projection = np.convolve(projection, np.ones(11) / 11, mode="same")
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

# -----------------------------
# PAGE → LINE EXTRACTION
# -----------------------------
def process_page(img_path: Path, page_idx: int):
    print(f"\n🖼 Processing {img_path.name}")

    page_dir = OUT_DIR / f"page_{page_idx:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️ Could not read {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows = detect_rows(gray)
    print(f"  Detected {len(rows)} rows")

    for line_idx, (y1, y2) in enumerate(rows):
        row_height = y2 - y1
        pad = int(row_height * 0.8)  # ← YOUR ORIGINAL LOGIC

        y1p = max(0, y1 - pad)
        y2p = min(gray.shape[0], y2 + pad)

        line = gray[y1p:y2p, :]

        if line.shape[0] < MIN_LINE_HEIGHT:
            continue

        out_path = page_dir / f"line_{line_idx:03d}.png"
        Image.fromarray(line).save(out_path)

# -----------------------------
# RUN
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pages = sorted(PAGE_DIR.glob("*.jpg"))
    if not pages:
        raise FileNotFoundError(f"No JPG pages found in {PAGE_DIR}")

    for idx, page in enumerate(pages, start=1):
        process_page(page, idx)

    print("\n✅ Line extraction finished.")

if __name__ == "__main__":
    main()
