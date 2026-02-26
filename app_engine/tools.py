# tools.py
from typing import List, Dict
from openai import OpenAI
import json
from dotenv import load_dotenv
# -------------------------------------------------
# OpenAI client (set your API key via environment)
# -------------------------------------------------
load_dotenv()
client = OpenAI()


# -------------------------------------------------
# 1. Find missing ingredients (single LLM call)
# -------------------------------------------------
def find_missing_ingredients(
    recipe_ingredients: List[str],
    user_ingredients: List[str]
) -> Dict[str, List[str]]:
    """
    Compare recipe ingredients against user ingredients using an LLM.

    The LLM will handle language differences, plural/singular,
    synonyms, and minor phrasing variations.

    Returns a dict in the exact format:
        {
            "available": [...],
            "missing": [...]
        }
    """

    # Build prompt
    prompt = f"""
    You are a cooking assistant.

    Compare the following recipe ingredients (German) against the user-provided ingredients (any language).

    Recipe ingredients:
    {recipe_ingredients}

    User ingredients:
    {user_ingredients}

    For each recipe ingredient, determine if the user has it.

    Return ONLY a valid JSON object with two keys: "available" and "missing", each containing lists of the recipe ingredients. 
    Do not add any extra text or explanation.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content.strip()

    try:
        data = json.loads(content)
        # Ensure keys exist
        return {
            "available": data.get("available", []),
            "missing": data.get("missing", [])
        }
    except json.JSONDecodeError:
        # fallback: assume nothing available
        return {"available": [], "missing": recipe_ingredients}
