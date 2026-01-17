import os
from dotenv import load_dotenv
from google.cloud import vision

# Load .env variables
load_dotenv()

# GOOGLE_APPLICATION_CREDENTIALS is automatically used by Google SDK
print("Using key file:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

# Initialize Vision client
client = vision.ImageAnnotatorClient()

# Function to detect text in an image (NO preprocessing)
def detect_text(image_path):
    if not os.path.exists(image_path):
        return ""

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description
    else:
        return ""


if __name__ == "__main__":
    image_folder = "./data/Part7"
    output_folder = "./data/Part7_txt"

    os.makedirs(output_folder, exist_ok=True)

    for filename in sorted(os.listdir(image_folder)):
        if filename == "Part7_page_8.jpg":#.lower().endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)

            text = detect_text(image_path)

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}.txt")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"\n---- {filename} ----\n{text}")
