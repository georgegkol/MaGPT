from dotenv import load_dotenv
import os
from pathlib import Path

# Dynamically find the project root and load .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # adjust if needed
load_dotenv(PROJECT_ROOT / ".env")

from agent import run_recipe_agent

# -----------------------------
# Test the recipe agent
# -----------------------------
recipe_title = "Simple Chicken Stew"
recipe_ingredients = ["chicken", "onion", "garlic", "tomato"]
user_text = "I have onion"

result = run_recipe_agent(recipe_title, recipe_ingredients, user_text)

print("Available:", result["available"])
print("Missing:", result["missing"])
print("Substitutions:", result["substitutions"])

