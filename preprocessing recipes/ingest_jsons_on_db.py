import json
import sqlite3
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIR = SCRIPT_DIR / "structured recipe jsons"
DB_PATH = DATA_DIR / "recipes.db"


# DB SETUP
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            recipe_id INTEGER PRIMARY KEY,
            title TEXT,
            ingredients TEXT,
            instructions TEXT,
            total_time TEXT,
            servings INTEGER,
            tags TEXT,
            embedding TEXT
        )
    """)

    conn.commit()
    return conn


# EMBEDDING BUILDER
def get_recipe_embedding(recipe):
    # ---- INGREDIENTS ----
    ingredients = recipe.get("ingredients")
    if isinstance(ingredients, list):
        ingredients_text = ", ".join(
            f"{i.get('name','')} {i.get('quantity','') or ''} {i.get('unit','') or ''}".strip()
            for i in ingredients if isinstance(i, dict)
        )
    else:
        ingredients_text = ""

    # ---- INSTRUCTIONS ----
    instructions = recipe.get("instructions")
    if isinstance(instructions, list):
        instructions_text = " ".join(
            map(str, filter(None, instructions))
        )
    else:
        instructions_text = ""

    # ---- TAGS ----
    tags = recipe.get("tags")
    tags_text = ", ".join(tags) if isinstance(tags, list) else ""

    recipe_text = (
        f"Title: {recipe.get('title','')}\n"
        f"Ingredients: {ingredients_text}\n"
        f"Instructions: {instructions_text}\n"
        f"Tags: {tags_text}"
    )

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=recipe_text
    )

    return response.data[0].embedding


# INGEST JSONs
def ingest_recipes():
    conn = init_db()
    cursor = conn.cursor()

    recipe_files = sorted(INPUT_DIR.glob("*.json"))
    if not recipe_files:
        raise RuntimeError(f"No JSON files found in {INPUT_DIR}")

    for recipe_file in recipe_files:
        try:
            # ---- EXTRACT ID FROM FILENAME ----
            match = re.search(r"(\d+)", recipe_file.stem)
            if not match:
                print(f"Skipping file without numeric ID: {recipe_file.name}")
                continue

            recipe_id = int(match.group(1))

            # ---- SKIP IF ALREADY EXISTS ----
            cursor.execute(
                "SELECT recipe_id FROM recipes WHERE recipe_id = ?",
                (recipe_id,)
            )
            if cursor.fetchone():
                print(f"Skipping already ingested recipe_id={recipe_id}")
                continue

            # ---- LOAD JSON ----
            with open(recipe_file, "r", encoding="utf-8") as f:
                recipe = json.load(f)

            # ---- EMBEDDING ----
            embedding_vector = get_recipe_embedding(recipe)

            # ---- INSERT ----
            cursor.execute("""
                INSERT INTO recipes (
                    recipe_id,
                    title,
                    ingredients,
                    instructions,
                    total_time,
                    servings,
                    tags,
                    embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recipe_id,
                recipe.get("title"),
                json.dumps(recipe.get("ingredients"), ensure_ascii=False),
                json.dumps(recipe.get("instructions"), ensure_ascii=False),
                recipe.get("total_time"),
                recipe.get("servings"),
                json.dumps(recipe.get("tags"), ensure_ascii=False),
                json.dumps(embedding_vector)
            ))

            print(f"Inserted recipe_id={recipe_id}: {recipe.get('title')}")

        except Exception as e:
            print(
                f"\n❌ ERROR ingesting file: {recipe_file.name}\n"
                f"   recipe_id={recipe_id if 'recipe_id' in locals() else 'UNKNOWN'}\n"
                f"   Error: {e}\n"
            )

    conn.commit()
    conn.close()

# MAIN
if __name__ == "__main__":
    ingest_recipes()
