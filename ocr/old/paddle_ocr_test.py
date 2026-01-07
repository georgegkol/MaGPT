from paddleocr import PaddleOCR
from PIL import Image

# Light German model
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='german',
    rec_model_dir=None,  # default model
    det_model_dir=None
)

# Load page
img_path = "./scans/jpgscans/Part7_page_4 copy.jpg"
img = Image.open(img_path)

# Downscale
max_side = 1500
ratio = max_side / max(img.size)
if ratio < 1:
    img = img.resize((int(img.width*ratio), int(img.height*ratio)))

# OCR
result = ocr.ocr(img_path)  # no segmentation
text = "\n".join([line[1][0] for line in result])

print(text[:500], "...")
