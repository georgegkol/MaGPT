# tools.py
from typing import List, Dict

# -----------------------------
# TOOL 1: NORMALIZE INGREDIENT NAMES
# -----------------------------
def normalize_ingredient(name: str) -> str:
    """
    Normalize ingredient names for comparison.
    """
    return name.lower().replace(",", "").replace(".", "").strip()


# -----------------------------
# TOOL 2: FIND MISSING INGREDIENTS
# -----------------------------
def find_missing_ingredients(recipe_ingredients: List[str], user_ingredients: List[str]) -> Dict[str, List[str]]:
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

    return {"available": available, "missing": missing}
