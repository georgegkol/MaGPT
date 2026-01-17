import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "indiv recipe txt"
OUTPUT_DIR = BASE_DIR / "structured recipe jsons"
OUTPUT_DIR.mkdir(exist_ok=True)

load_dotenv()
client = OpenAI()


# PROMPT
EXAMPLE_PROMPT = """
Convert the following text containing a recipe into a structured JSON with the keys: title, ingredients, instructions, total_time, servings, tags.
- Put the instructions into one text block (list with one string) without ever translating from German and just leave as null if there are no instructions.
- Standardise units to g, ml, teaspoons etc.
- Include total times and servings if directly mentioned in the text, otherwise use null.
- Come up with the appropriate tags based on the recipe content (avoid phrases with the word German).

For ingredients, output a list of objects with (if any information is missing, use null for that field):
- name (ingredient name only)
- quantity (numeric value)
- unit (unit of measurement)

Example structured JSON:
{
  "title": "Simple Cake",
  "ingredients": [
    {"name": "flour", "quantity": 200, "unit": "g"},
    {"name": "milk", "quantity": 100, "unit": "ml"}
  ],
  "instructions": [
    "Mix flour and sugar in a bowl. Add milk and stir until smooth. Bake at 180°C for 25 minutes."
  ],
  "total_time": "35 min",
  "servings": 4,
  "tags": ["Dessert", "Cake", "Baking"]
}

Return ONLY valid JSON. Do not include explanations or markdown.
"""


# FUNCTION
def structure_recipe_with_llm(recipe_title, raw_recipe_text):
    prompt = f"{EXAMPLE_PROMPT}\n\nRecipe title: {recipe_title}\n\nRaw recipe:\n---\n{raw_recipe_text}\n---\nStructured JSON:"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    json_text = response.choices[0].message.content

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse JSON",
            "raw_output": json_text
        }


# MAIN
if __name__ == "__main__":

    recipe_files = sorted(INPUT_DIR.glob("*.json"))

    for recipe_file in recipe_files:
        output_file = OUTPUT_DIR / recipe_file.name

        # Skip if already processed (VERY IMPORTANT for cost safety)
        if output_file.exists():
            continue

        with open(recipe_file, "r", encoding="utf-8") as f:
            raw_recipe = json.load(f)

        structured_recipe = structure_recipe_with_llm(
            raw_recipe["title"],
            raw_recipe["raw_text"]
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(structured_recipe, f, indent=2, ensure_ascii=False)

        print(f"Saved: {output_file.name}")