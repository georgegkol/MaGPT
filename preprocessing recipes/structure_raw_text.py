import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "text extraction" / "pages122-124_raw_description.txt"

load_dotenv()

client = OpenAI()

EXAMPLE_PROMPT = """
Convert the following text containing recipes into a structured JSON with the keys: title, ingredients, instructions, total_time, servings, tags.
- Detect and split multiple recipes, then structure each one separately.
- Standardise units to g, ml, teaspoons etc.
- For instructions, split into a list of steps.
- Come up with the appropriate tags based on the recipe content.

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
    "Mix flour and sugar in a bowl.",
    "Add milk and stir until smooth.",
    "Bake at 180°C for 25 minutes."
  ],
  "total_time": "35 min",
  "servings": 4,
  "tags": ["Dessert", "Cake", "Baking"]
}

Return ONLY valid JSON. Do not include explanations or markdown.
"""

def structure_recipe_with_llm(raw_recipe_text):
    """
    Takes raw recipe text and returns structured JSON with standardized fields.
    """
    # Combine the example schema with the actual recipe
    prompt = f"{EXAMPLE_PROMPT}\n\nRaw recipe:\n---\n{raw_recipe_text}\n---\nStructured JSON:"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    # Extract the assistant's message
    json_text = response.choices[0].message.content
    
    # Parse JSON safely
    try:
        recipe_json = json.loads(json_text)
    except json.JSONDecodeError:
        # If LLM output has minor formatting issues, we could clean it or return raw
        recipe_json = {"error": "Failed to parse JSON", "raw_output": json_text}
    
    return recipe_json


# Example usage
if __name__ == "__main__":
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    structured_recipe = structure_recipe_with_llm(raw_text)
    print(json.dumps(structured_recipe, indent=2))