import json
from typing import List, Dict, Any
from tools import find_missing_ingredients
from openai import OpenAI

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, create_openai_functions_agent

client = OpenAI()

# -----------------------------
# LLM TOOL 1: PARSE USER INPUT
# -----------------------------
def parse_user_input(user_text: str) -> Dict[str, Any]:
    """
    Takes natural language input and returns a structured list of ingredients.
    """
    prompt = f"""
Extract structured ingredient info from this text:

"{user_text}"

Return JSON:
{{
  "ingredients": ["ingredient1", "ingredient2"],
  "quantities": {{
    "ingredient_name": [quantity, "unit"]
  }}
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"ingredients": [], "quantities": {}}


# -----------------------------
# LLM TOOL 2: SUGGEST SUBSTITUTIONS
# -----------------------------
def suggest_substitutions(data: Dict[str, Any]) -> Dict[str, List[str]]:
    missing_ingredients = data["missing_ingredients"]
    recipe_title = data["recipe_title"]
    recipe_tags = data.get("recipe_tags", [])
    tags_text = ", ".join(recipe_tags) if recipe_tags else "none"

    prompt = f"""
You are a professional chef.

Recipe: {recipe_title}
Recipe tags: {tags_text}
Missing ingredients: {missing_ingredients}

Suggest 2–3 substitutions per missing ingredient.
Return only valid JSON:
{{ "ingredient_name": ["sub1", "sub2"] }}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {i: [] for i in missing_ingredients}


# -----------------------------
# REGISTER TOOLS FOR AGENT
# -----------------------------
tools = [
    Tool(
        name="Parse User Input",
        func=parse_user_input,
        description="Parse natural language input into structured ingredients."
    ),
    Tool(
        name="Find Missing Ingredients",
        func=find_missing_ingredients,
        description="Compare recipe ingredients with user ingredients."
    ),
    Tool(
        name="Suggest Substitutions",
        func=suggest_substitutions,
        description="Suggest substitutions for missing ingredients."
    ),
]

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

agent_executor = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    verbose=True
)


# -----------------------------
# HELPER FUNCTION TO RUN AGENT
# -----------------------------
def run_recipe_agent(recipe_title: str, recipe_ingredients: List[str], user_text: str, recipe_tags: List[str] | None = None):
    user_data = parse_user_input(user_text)
    user_ingredients = user_data.get("ingredients", [])

    missing_data = find_missing_ingredients({
        "recipe_ingredients": recipe_ingredients,
        "user_ingredients": user_ingredients
    })

    missing = missing_data["missing"]
    available = missing_data["available"]

    substitutions = {}
    if missing:
        substitutions = suggest_substitutions({
            "missing_ingredients": missing,
            "recipe_title": recipe_title,
            "recipe_tags": recipe_tags or []
        })

    return {
        "available": available,
        "missing": missing,
        "substitutions": substitutions
    }
