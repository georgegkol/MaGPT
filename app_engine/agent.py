# agent.py
import json
from typing import Dict, List, Any
from tools import parse_user_input, find_missing_ingredients
from openai import OpenAI

# --- LangChain imports ---
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent

# -----------------------------
# LLM TOOL: SUBSTITUTIONS
# -----------------------------
client = OpenAI()

def suggest_substitutions(input_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    LLM-based tool to suggest ingredient substitutions
    input_data:
        {
            "missing_ingredients": [...],
            "recipe_title": "...",
            "recipe_tags": [...]
        }
    """
    missing_ingredients = input_data["missing_ingredients"]
    recipe_title = input_data["recipe_title"]
    recipe_tags = input_data.get("recipe_tags", [])
    tags_text = ", ".join(recipe_tags) if recipe_tags else "none"

    prompt = f"""
You are a professional chef.

Recipe: {recipe_title}
Recipe tags: {tags_text}

Missing ingredients:
{missing_ingredients}

Suggest 2–3 substitutions per missing ingredient.
Return ONLY valid JSON:
{{ "ingredient_name": ["sub1", "sub2"] }}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {i: [] for i in missing_ingredients}

# -----------------------------
# REGISTER TOOLS
# -----------------------------
tools = [
    Tool(
        name="Parse User Input",
        func=parse_user_input,
        description="Parse natural language input from the user to extract ingredients and quantities."
    ),
    Tool(
        name="Find Missing Ingredients",
        func=find_missing_ingredients,
        description="Compare recipe ingredients with user ingredients and return which are missing."
    ),
    Tool(
        name="Suggest Substitutions",
        func=suggest_substitutions,
        description="Use LLM to suggest alternative ingredients for missing items."
    ),
]

# -----------------------------
# CREATE AGENT
# -----------------------------
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=None)

agent_executor = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    verbose=True
)

# -----------------------------
# HELPER FUNCTION TO RUN AGENT
# -----------------------------
def run_recipe_agent(recipe_title: str, recipe_ingredients: List[str], user_input: str, recipe_tags: List[str] | None = None) -> Dict[str, Any]:
    """
    Run the agent on a recipe + user-provided ingredient list.
    
    Steps:
    1. Parse user input
    2. Find missing ingredients
    3. Suggest substitutions via LLM
    """
    # Step 1: Parse user's ingredients
    user_data = parse_user_input(user_input)
    user_ingredients = user_data.get("ingredients", [])

    # Step 2: Find missing ingredients
    missing_data = find_missing_ingredients({
        "recipe_ingredients": recipe_ingredients,
        "user_ingredients": user_ingredients
    })

    missing = missing_data["missing"]
    available = missing_data["available"]

    # Step 3: Suggest substitutions
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
