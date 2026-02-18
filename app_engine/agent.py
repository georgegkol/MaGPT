from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from tools import find_missing_ingredients
import json
import os
from dotenv import load_dotenv

# -----------------------------
# LLM TOOL: SUGGEST SUBSTITUTIONS
# -----------------------------
def suggest_substitutions_llm(missing_ingredients, recipe_title, recipe_tags=None):
    """
    Ask GPT to suggest substitutions for missing ingredients.
    """
    tags_text = ", ".join(recipe_tags) if recipe_tags else "none"

    prompt_template = """
You are a professional chef.

Recipe: {recipe_title}
Recipe tags: {tags_text}
Missing ingredients: {missing_ingredients}

Suggest 2–3 substitutions per missing ingredient.
Return ONLY valid JSON:
{{ "ingredient_name": ["sub1", "sub2"] }}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    chain = LLMChain(llm=chat, prompt=prompt)

    response = chain.run(
        recipe_title=recipe_title,
        tags_text=tags_text,
        missing_ingredients=json.dumps(missing_ingredients)
    )

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {i: [] for i in missing_ingredients}


# -----------------------------
# LANGCHAIN TOOLS
# -----------------------------
tools = [
    Tool(
        name="Find Missing Ingredients",
        func=find_missing_ingredients,
        description="Determine which ingredients the user has and which are missing."
    ),
    Tool(
        name="Suggest Substitutions",
        func=suggest_substitutions_llm,
        description="Provide substitutions for missing ingredients using GPT."
    ),
]

# -----------------------------
# RECIPE AGENT
# -----------------------------
chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
agent = initialize_agent(
    tools,
    llm=chat,
    agent="zero-shot-react-description",
    verbose=True
)

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def run_recipe_agent(recipe_title, recipe_ingredients, user_ingredients, recipe_tags=None):
    """
    Orchestrates tools:
    1. Find missing ingredients
    2. Suggest substitutions using LLM
    """
    # Step 1: missing ingredients
    missing_data = find_missing_ingredients(recipe_ingredients, user_ingredients)
    available = missing_data["available"]
    missing = missing_data["missing"]

    # Step 2: substitutions
    substitutions = {}
    if missing:
        substitutions = suggest_substitutions_llm(missing, recipe_title, recipe_tags)

    return {
        "available": available,
        "missing": missing,
        "substitutions": substitutions
    }
