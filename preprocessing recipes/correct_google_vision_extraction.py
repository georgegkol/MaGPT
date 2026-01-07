import os
from dotenv import load_dotenv
import openai

# Load .env from parent folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

input_folder = "./preprocessing recipes/Part7_txt"
output_folder = "./preprocessing recipes/Part7_cleaned"

os.makedirs(output_folder, exist_ok=True)

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        prompt = f"""
        You are an expert German cookbook editor.

        Take the following raw OCR text from a handwritten recipe and rewrite it
        into a clean, human-readable recipe in proper German.
        Fix spelling mistakes, correct OCR errors, organize ingredients and steps logically.
        Output plain text, like a normal cookbook. Do NOT use JSON.

        Raw OCR text:
        \"\"\"{raw_text}\"\"\"
        """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        clean_text = response.choices[0].message.content

        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"Processed {filename}")