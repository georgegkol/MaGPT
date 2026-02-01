import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import supabase
from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# SETUP
# -----------------------------
load_dotenv()
client = OpenAI()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "recipes-index"

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(INDEX_NAME)

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
# SEMANTIC SEARCH
# -----------------------------
def search_recipes(query: str, top_n: int = 20):
    """Return top-N candidate recipes based on embeddings (IDs handled as strings)."""
    normalized_query = normalize_query(query)
    query_emb = embed_text(normalized_query)

    # Query Pinecone
    results = pinecone_index.query(vector=query_emb.tolist(), top_k=top_n, include_metadata=True)
    if not results['matches']:
        print("⚠️ Pinecone returned no matches for this query.")
        return []

    # Pinecone IDs are strings; Supabase id column is text
    recipe_ids = [match['id'] for match in results['matches']]

    # Fetch recipes from Supabase
    supabase_resp = supabase_client.table("recipes").select("*").in_("id", recipe_ids).execute()
    recipes_data = {r["id"]: r for r in supabase_resp.data}

    # Build results list in Pinecone order
    recipes = []
    for match in results['matches']:
        rid = match['id']
        recipe = recipes_data.get(rid)
        if recipe:
            recipe['score'] = match['score']
            recipes.append(recipe)

    # Sort by Pinecone similarity (already roughly in order)
    recipes.sort(key=lambda x: x["score"], reverse=True)

    if not recipes:
        print("⚠️ No recipes found in Supabase matching Pinecone IDs.")
    return recipes

# -----------------------------
# LLM RERANKING
# ----------------------
def rerank_recipes(user_query: str, candidates: list):
    """
    Rerank candidate recipes using the LLM.
    Fallback to Pinecone similarity if LLM fails.
    Ensures final_results always contains recipes with 'final_score'.
    """
    if not candidates:
        return []

    # Build compact recipe text for LLM
    recipes_text = ""
    for r in candidates:
        recipes_text += f"Recipe ID: {r['id']}\nTitle: {r.get('title','')}\n"
        ingredients_list = r.get('ingredients') or []
        ingredients_text = ", ".join(
            f"{i.get('name','')} {i.get('quantity','') or ''} {i.get('unit','') or ''}".strip()
            for i in ingredients_list
        )
        recipes_text += "Ingredients: " + ingredients_text + "\n"
        instructions_list = r.get('instructions') or []
        instructions_text = " ".join(map(str, filter(None, instructions_list)))
        recipes_text += "Instructions: " + instructions_text + "\n\n"

    prompt = (
        f"You are a recipe expert. A user asked: '{user_query}'.\n\n"
        f"Here are candidate recipes:\n{recipes_text}\n"
        "For each recipe, give a confidence score between 0 and 1 for how well it matches the user's request. "
        "Do not reorder recipes. Return a JSON list of objects with keys: id, score. "
        "Do NOT normalize or force rankings. Do not include explanations."
    )

    # Attempt LLM rerank
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        reranked_raw = response.choices[0].message.content
        reranked = json.loads(reranked_raw)

    except Exception as e:
        print("⚠️ LLM rerank failed, using Pinecone fallback:", e)
        # Ensure IDs are strings for safe mapping
        reranked = [{"id": str(r["id"]), "score": r.get("score", 1.0)} for r in candidates]

    # Map rerank scores back to candidate recipes
    id_to_recipe = {str(r["id"]): r for r in candidates}
    final_results = []
    for r in reranked:
        rid = str(r["id"])
        recipe = id_to_recipe.get(rid)
        if recipe:
            # final_score defaults to rerank score, then Pinecone score, then 1.0
            recipe["final_score"] = r.get("score", recipe.get("score", 1.0))
            final_results.append(recipe)

    # Return all candidates — no filtering
    return final_results


# -----------------------------
# SHOW RECIPE
# -----------------------------
def show_recipe(recipe_id):
    """Fetch and display a recipe (IDs as strings)."""
    resp = supabase_client.table("recipes").select("*").eq("id", recipe_id).single().execute()
    recipe = resp.data
    if not recipe:
        print(f"\n❌ No recipe found with id={recipe_id}")
        return

    print("\n" + "=" * 50)
    print(f"{recipe['title']}")
    print("=" * 50)

    if recipe.get("servings"):
        print(f"🍽 Servings: {recipe['servings']}")
    if recipe.get("total_time"):
        print(f"⏱ Total time: {recipe['total_time']}")

    if recipe.get("ingredients"):
        print("\n🧂 Ingredients:")
        for ing in recipe['ingredients']:
            name = ing.get("name", "")
            qty = ing.get("quantity", "")
            unit = ing.get("unit", "")
            line = " ".join(str(x) for x in [qty, unit, name] if x)
            print(f"  • {line}")

    if recipe.get("instructions"):
        print("\n📋 Instructions:")
        for step in recipe['instructions']:
            print(step)

    print("\n" + "=" * 50)


# -----------------------------
# CLI LOOP
# -----------------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk for a recipe (or 'exit'): ")
        if query.lower() == "exit":
            break

        # 1️⃣ Search
        candidates = search_recipes(query, top_n=20)

        if not candidates:
            print("No candidate recipes found.")
            continue

        # 2️⃣ Rerank
        final_matches = rerank_recipes(query, candidates)

        if not final_matches:
            print("No final matches after rerank.")
            continue

        # 3️⃣ Display top matches
        print("\nTop matches:")
        for m in final_matches[:5]:
            print(f"{m['final_score']:.3f} | {m['title']} (id={m['id']})")

        # 4️⃣ Allow user to view a recipe
        choice = input("\nEnter recipe ID to view full recipe (or press enter): ").strip()
        if choice:
            show_recipe(choice)
