import json
from typing import List, Dict, Tuple, Any
from openai import OpenAI

client = OpenAI()

# -----------------------------
# TOOL 1: PARSE USER INPUT
# -----------------------------
def parse_user_input(user_text: str) -> Dict[str, Any]:
    """
    Extract user ingredients and quantities from natural language.
    
    Returns:
    {
        "ingredients": ["chicken", "onion"],
        "quantities": {
            "chicken": [300, "g"]
        }
    }
    """

    prompt = f"""
    Extract structured ingredient information from the text below.

    Return JSON in this format:
    {{
        "ingredients": ["ingredient1", "ingredient2"],
        "quantities": {{
            "ingredient_name": [quantity, "unit"]
        }}
    }}

    If no quantity is specified, only include it in "ingredients".

    Text:
    "{user_text}"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"ingredients": [], "quantities": {}}


# -----------------------------
# TOOL 2: FIND MISSING INGREDIENTS
# -----------------------------
def normalize_ingredient(name: str) -> str:
    return name.lower().replace(",", "").replace(".", "").strip()


def find_missing_ingredients(input_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    input_data:
    {
        "recipe_ingredients": [...],
        "user_ingredients": [...]
    }
    """

    recipe_ingredients = input_data["recipe_ingredients"]
    user_ingredients = input_data["user_ingredients"]

    recipe_norm = [normalize_ingredient(i) for i in recipe_ingredients]
    user_norm = {normalize_ingredient(i) for i in user_ingredients}

    available = []
    missing = []

    for original, norm in zip(recipe_ingredients, recipe_norm):
        if norm in user_norm:
            available.append(original)
        else:
            missing.append(original)

    return {"available": available, "missing": missing}


# -----------------------------
# TOOL 3: SUGGEST SUBSTITUTIONS
# -----------------------------
def suggest_substitutions(input_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    input_data:
    {
        "missing_ingredients": [...],
        "recipe_title": "...",
        "recipe_tags": [...]
    }
    """

    missing_ingredients = input_data["missing_ingredients"]
    recipe_title = input_data["recipe_title"]
    recipe_tags = input_data.get("recipe_tags", [])

    tags_text = ", ".join(recipe_tags) if recipe_tags else "none"

    prompt = f"""
    You are a professional chef.

    Recipe: {recipe_title}
    Recipe tags: {tags_text}
    Missing ingredients: {missing_ingredients}

    Suggest 2–3 substitutions per missing ingredient.
    Return ONLY valid JSON:
    {{
        "ingredient_name": ["sub1", "sub2"]
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {}