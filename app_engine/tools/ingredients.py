from typing import List, Dict
from openai import OpenAI
import json

client = OpenAI()


def normalize_ingredient(name: str) -> str:
    """
    Normalize ingredient names for comparison.
    """
    return (
        name.lower()
        .replace(",", "")
        .replace(".", "")
        .strip()
    )


def find_missing_ingredients(recipe_ingredients: List[str],user_ingredients: List[str],) -> Dict[str, List[str]]:
    """
    Compare recipe ingredients against user ingredients.

    Returns:
        {
            "available": [...],
            "missing": [...]
        }
    """

    recipe_norm = [normalize_ingredient(i) for i in recipe_ingredients]
    user_norm = {normalize_ingredient(i) for i in user_ingredients}

    available = []
    missing = []

    for original, norm in zip(recipe_ingredients, recipe_norm):
        if norm in user_norm:
            available.append(original)
        else:
            missing.append(original)

    return {
        "available": available,
        "missing": missing,
    }



def suggest_substitutions(missing_ingredients: List[str], recipe_title: str, recipe_tags: List[str] | None = None,) -> Dict[str, List[str]]:
    """
    Use an LLM to suggest substitutions for missing ingredients.
    Returns a dict: {ingredient: [sub1, sub2]}
    """

    tags_text = ", ".join(recipe_tags) if recipe_tags else "none"

    prompt = f"""
        You are a professional chef.

        Recipe: {recipe_title}
        Recipe tags: {tags_text}

        Missing ingredients:
        {missing_ingredients}

        For EACH missing ingredient, suggest 2–3 reasonable substitutions.
        - Prefer common supermarket items
        - Keep the recipe style intact
        - If no good substitution exists, return an empty list for that ingredient

        Return ONLY valid JSON in this format:

        {{
        "ingredient_name": ["sub1", "sub2"]
        }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


from typing import Dict, Tuple


def scale_ingredients(recipe_ingredients: Dict[str, Tuple[float, str]],user_ingredients: Dict[str, Tuple[float, str]],) -> Dict[str, Tuple[float, str]]:
    """
    Scales recipe ingredients based on the limiting ingredient
    present in user_ingredients.

    recipe_ingredients:
        {"chicken": (500, "g"), "onion": (2, "count")}

    user_ingredients:
        {"chicken": (300, "g")}

    Returns scaled recipe ingredients.
    """

    scale_factors = []

    for name, (recipe_qty, unit) in recipe_ingredients.items():
        if name in user_ingredients:
            user_qty, user_unit = user_ingredients[name]

            if unit != user_unit or recipe_qty == 0:
                continue

            scale_factors.append(user_qty / recipe_qty)

    if not scale_factors:
        return recipe_ingredients

    scale_factor = min(scale_factors)

    scaled = {}
    for name, (qty, unit) in recipe_ingredients.items():
        scaled[name] = (round(qty * scale_factor, 2), unit)

    return scaled

