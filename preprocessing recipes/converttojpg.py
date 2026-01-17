import fitz  # PyMuPDF
import os

# Folders
pdf_folder = "data"
jpg_folder = "data/jpgscans"

# Ensure output folder exists
os.makedirs(jpg_folder, exist_ok=True)

# Loop over Part1.pdf to Part7.pdf
for i in range(5,8):
    pdf_path = os.path.join(pdf_folder, f"Part{i}.pdf")
    doc = fitz.open(pdf_path)
    
    for page_number in range(doc.page_count):  # all pages
        page = doc.load_page(page_number)  # 0-indexed
        pix = page.get_pixmap(dpi=300)
        output_path = os.path.join(jpg_folder, f"Part{i}_page_{page_number+1}.jpg")
        pix.save(output_path)
        print(f"Saved {output_path}")

    doc.close()
