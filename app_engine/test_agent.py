from agent import run_recipe_agent

recipe_title = "Simple Chicken Stew"
recipe_ingredients = ["chicken", "onion", "garlic", "tomato"]
user_ingredients = ["onion", "tomato"]

result = run_recipe_agent(recipe_title, recipe_ingredients, user_ingredients)
print("Available:", result["available"])
print("Missing:", result["missing"])
print("Substitutions:", result["substitutions"])
