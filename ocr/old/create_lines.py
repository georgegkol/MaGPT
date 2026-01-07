from pathlib import Path
from PIL import Image
import pytesseract

# Paths
input_pages = [
    "./scans/jpgscans/Part1_page_1.jpg",
    "./scans/jpgscans/Part2_page_2.jpg"
]
output_dir = Path("./ocr_output")
output_dir.mkdir(exist_ok=True)

# Set language to German ('deu') – make sure German language pack is installed
ocr_lang = 'deu'

for page_path in input_pages:
    img = Image.open(page_path).convert("L")  # convert to grayscale

    # OCR the entire page
    page_text = pytesseract.image_to_string(img, lang=ocr_lang)

    # Optional: save page text
    out_file = output_dir / f"{Path(page_path).stem}.txt"
    out_file.write_text(page_text, encoding="utf-8")

    print(f"Extracted text from {page_path} → {out_file}")

    # Optional: save line images and texts
    data = pytesseract.image_to_data(img, lang=ocr_lang, output_type=pytesseract.Output.DICT)
    for i, level in enumerate(data['level']):
        if level == 5:  # 5 corresponds to a line
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            line_img = img.crop((x, y, x + w, y + h))
            line_text = data['text'][i]
            line_path = output_dir / f"{Path(page_path).stem}_line_{i+1}.png"
            line_img.save(line_path)
            print(f"Saved line {i+1}: {line_text}")
