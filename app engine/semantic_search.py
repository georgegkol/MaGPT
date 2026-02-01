import json
import sqlite3
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# SETUP
# -----------------------------
load_dotenv()
client = OpenAI()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "recipes.db"

# -----------------------------
# EMBEDDING
# -----------------------------
def embed_text(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def normalize_query(user_query: str) -> str:
    """Convert user query to a short, neutral search description"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's request into a short, neutral recipe search description. "
                    "Be faithful to the original meaning. Output one sentence only."
                )
            },
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -----------------------------
# SEMANTIC SEARCH
# -----------------------------
def search_recipes(query: str, top_n: int = 20):
    """Return top-N candidate recipes based on embeddings"""
    normalized_query = normalize_query(query)
    query_emb = embed_text(normalized_query)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT recipe_id, title, ingredients, instructions, embedding FROM recipes WHERE embedding IS NOT NULL")
    results = []
    for recipe_id, title, ingredients_json, instructions_json, embedding_json in cursor.fetchall():
        recipe_emb = np.array(json.loads(embedding_json), dtype=np.float32)
        score = cosine_similarity(query_emb, recipe_emb)
        results.append({
            "recipe_id": recipe_id,
            "title": title,
            "ingredients": json.loads(ingredients_json),
            "instructions": json.loads(instructions_json),
            "score": score
        })

    conn.close()
    # Sort by similarity
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]

# -----------------------------
# LLM RERANKING
# -----------------------------
import json

def rerank_recipes(user_query: str, candidates: list):
    """
    Rerank candidate recipes using the LLM, producing real relevance scores (0–1).
    Only pass compact text to reduce token usage.
    """
    # Build compact recipe descriptions
    recipes_text = ""
    for r in candidates:
        recipes_text += f"Recipe ID: {r['recipe_id']}\nTitle: {r.get('title','')}\n"

        # Ingredients
        ingredients_list = r.get('ingredients') or []
        ingredients_text = ", ".join(
            f"{i.get('name','')} {i.get('quantity','') or ''} {i.get('unit','') or ''}".strip()
            for i in ingredients_list
        )
        recipes_text += "Ingredients: " + ingredients_text + "\n"

        # Instructions
        instructions_list = r.get('instructions') or []
        instructions_text = " ".join(map(str, filter(None, instructions_list)))
        recipes_text += "Instructions: " + instructions_text + "\n\n"

    prompt = (
        f"You are a recipe expert. A user asked: '{user_query}'.\n\n"
        f"Here are candidate recipes:\n{recipes_text}\n"
        "For each recipe, give a confidence score between 0 and 1 for how well it matches the user's request. "
        "Do not reorder recipes."
        "Return a JSON list of objects with keys: recipe_id, score. "
        "Do NOT normalize or force rankings. Do not include explanations."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        reranked = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # fallback: keep embedding order with embedding scores
        reranked = [{"recipe_id": r["recipe_id"], "score": r.get("score", 0)} for r in candidates]

    # Map back to recipe objects and attach final_score
    id_to_recipe = {r["recipe_id"]: r for r in candidates}
    final_results = []
    for r in reranked:
        recipe = id_to_recipe.get(r["recipe_id"])
        if recipe:
            recipe["final_score"] = r.get("score", 0)
            final_results.append(recipe)

    # Optionally, filter out low-confidence recipes (e.g., <0.3)
    final_results = [r for r in final_results if r["final_score"] >= 0.3]

    return final_results

def show_recipe(recipe_id: int):
    """Fetch and display a recipe with all non-null fields (no LLM, no JSON)."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT recipe_id, title, ingredients, instructions, total_time, servings, tags
        FROM recipes
        WHERE recipe_id = ?
    """, (recipe_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"\n❌ No recipe found with id={recipe_id}")
        return

    (recipe_id, title, ingredients_json, instructions_json, total_time, servings, tags_json) = row

    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50)

    if servings:
        print(f"🍽 Servings: {servings}")

    if total_time:
        print(f"⏱ Total time: {total_time}")

    # -------- INGREDIENTS --------
    if ingredients_json:
        ingredients = json.loads(ingredients_json)
        print("\n🧂 Ingredients:")
        for ing in ingredients:
            name = ing.get("name", "")
            qty = ing.get("quantity", "")
            unit = ing.get("unit", "")
            line = " ".join(str(x) for x in [qty, unit, name] if x)
            print(f"  • {line}")

    # -------- INSTRUCTIONS --------
    if instructions_json:
        instructions = json.loads(instructions_json)
        print("\n📋 Instructions:")
        for i, step in enumerate(instructions, 1):
            print(step)

    # -------- TAGS --------
    """if tags_json:
        tags = json.loads(tags_json)
        if tags:
            print("\n🏷 Tags:", ", ".join(tags))"""

    print("\n" + "=" * 50)


# -----------------------------
# CLI TEST
# -----------------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk for a recipe (or 'exit'): ")
        if query.lower() == "exit":
            break

        candidates = search_recipes(query, top_n=20)
        final_matches = rerank_recipes(query, candidates)

        print("\nTop matches:")
        for m in final_matches[:5]:
            print(f"{m['final_score']:.3f} | {m['title']} (id={m['recipe_id']})")

        choice = input("\nEnter recipe ID to view full recipe (or press enter): ").strip()
        if choice.isdigit():
            show_recipe(int(choice))
