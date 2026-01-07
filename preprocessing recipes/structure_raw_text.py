import json
from openai import OpenAI

client = OpenAI()  # Make sure your OPENAI_API_KEY is set

EXAMPLE_PROMPT = """
Convert the following recipe into a structured JSON with the keys: title, ingredients, instructions, prep_time, cook_time, total_time, servings, tags.

For ingredients, output a list of objects with:
- name (ingredient name only)
- quantity (numeric, use null if not specified)
- unit (use null if not specified)
- raw_text (the original text from the recipe)

Example:

Raw recipe:
---
Title: Simple Cake
Ingredients:
200g flour
50g sugar
100ml milk
Instructions:
Mix flour and sugar in a bowl.
Add milk and stir until smooth.
Bake at 180°C for 25 minutes.
Prep time: 10 min
Cook time: 25 min
Servings: 4
---

Structured JSON:
{
  "title": "Simple Cake",
  "ingredients": [
    {"name": "flour", "quantity": 200, "unit": "g", "raw_text": "200g flour"},
    {"name": "sugar", "quantity": 50, "unit": "g", "raw_text": "50g sugar"},
    {"name": "milk", "quantity": 100, "unit": "ml", "raw_text": "100ml milk"}
  ],
  "instructions": [
    "Mix flour and sugar in a bowl.",
    "Add milk and stir until smooth.",
    "Bake at 180°C for 25 minutes."
  ],
  "prep_time": "10 min",
  "cook_time": "25 min",
  "total_time": "35 min",
  "servings": 4,
  "tags": ["Dessert", "Cake", "Baking"]
}
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
    json_text = response.choices[0].message["content"]
    
    # Parse JSON safely
    try:
        recipe_json = json.loads(json_text)
    except json.JSONDecodeError:
        # If LLM output has minor formatting issues, we could clean it or return raw
        recipe_json = {"error": "Failed to parse JSON", "raw_output": json_text}
    
    return recipe_json

# Example usage
if __name__ == "__main__":
    with open("recipe1.txt") as f:
        raw_text = f.read()
    
    structured_recipe = structure_recipe_with_llm(raw_text)
    print(json.dumps(structured_recipe, indent=2))