from pathlib import Path
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "text extraction"
OUTPUT_DIR = BASE_DIR / "segmented recipes"
OUTPUT_DIR.mkdir(exist_ok=True)

# LLM prompt for page segmentation
SEGMENT_PROMPT = """
You are reading cookbook OCR-extracted pages.

Current recipe title (may be null): {current_title}
Next page text:
---
{page_text}
---

Decide whether this page:
1) continues the current recipe, or
2) starts one or more new recipes.

Completely ignore the segmentation ===PAGE X=== markers, pretend they dont exist.

Return JSON actions ONLY, in this format:
[
  {{
    "type": "continue" | "start_new",
    "recipe_title": "...",
    "text": "..."
  }}
]

Do not include any explanations or markdown.
"""

# -----------------------------
# FUNCTIONS
# -----------------------------
def segment_page(current_title, page_text):
    """Ask the LLM how this page fits into recipes."""
    prompt = SEGMENT_PROMPT.format(current_title=current_title, page_text=page_text)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    json_text = response.choices[0].message.content
    # Strip backticks if LLM includes them
    json_text = json_text.strip().strip("```json").strip("```")
    try:
        actions = json.loads(json_text)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from LLM:\n{json_text}")
    return actions

def segment_recipes_from_pages():
    """Read all txt pages and segment recipes, saving one JSON per recipe."""
    pages = sorted(TEXT_DIR.glob("*.txt"))  # sorted alphabetically
    current_recipe = None
    recipe_counter = 0

    for page_file in pages:
        print( f"{recipe_counter} processed so far")
        with open(page_file, "r", encoding="utf-8") as f:
            page_text = f.read()

        actions = segment_page(
            current_title=current_recipe["title"] if current_recipe else None,
            page_text=page_text
        )

        for action in actions:
            if action["type"] == "start_new":
                # Save previous recipe
                if current_recipe:
                    recipe_counter += 1
                    out_path = OUTPUT_DIR / f"recipe_{recipe_counter}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(current_recipe, f, indent=2, ensure_ascii=False)
                # Start new recipe
                current_recipe = {
                    "title": action["recipe_title"],
                    "raw_text": action["text"]
                }
            elif action["type"] == "continue":
                if current_recipe:
                    current_recipe["raw_text"] += "\n" + action["text"]
                else:
                    # Edge case: continue but no current recipe
                    current_recipe = {
                        "title": action.get("recipe_title") or "Unknown",
                        "raw_text": action["text"]
                    }
            else:
                raise ValueError(f"Unknown action type: {action['type']}")
            
    # Save the last recipe
    if current_recipe:
        recipe_counter += 1
        out_path = OUTPUT_DIR / f"recipe_{recipe_counter}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(current_recipe, f, indent=2, ensure_ascii=False)

    print(f"Segmentation complete: {recipe_counter} recipes saved.")
    return recipe_counter


# MAIN
if __name__ == "__main__":
    segment_recipes_from_pages()
