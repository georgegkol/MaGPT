import json
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tools import find_missing_ingredients

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents import create_openai_functions_agent

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# TOOL 1: Parse User Input
# -----------------------------
def parse_user_input(user_text: str) -> Dict[str, Any]:
    """
    Extracts ingredients and quantities from natural language input.
    """
    prompt = f"""
Extract the ingredients from the following text.

User input: "{user_text}"

Return only JSON in this exact format:
{{
  "ingredients": ["ingredient1", "ingredient2"],
  "quantities": {{
    "ingredient_name": [quantity, "unit"]
  }}
}}
Do NOT add any explanation or text outside the JSON.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("JSON parse error. Output was:", content)
        return {"ingredients": [], "quantities": {}}


# -----------------------------
# TOOL 2: Suggest Substitutions
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

Suggest 2–3 substitutions per missing ingredient only.
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
        return {ingredient: [] for ingredient in missing_ingredients}


# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# -----------------------------
# Define Tools for Agent
# -----------------------------
tools = [
    {
        "name": "Parse User Input",
        "func": parse_user_input,
        "description": "Parse natural language text into structured ingredient list."
    },
    {
        "name": "Find Missing Ingredients",
        "func": find_missing_ingredients,
        "description": "Compare recipe ingredients with user ingredients."
    },
    {
        "name": "Suggest Substitutions",
        "func": suggest_substitutions,
        "description": "Suggest ingredient substitutions for missing items."
    }
]

# -----------------------------
# Prompt Template for Agent
# -----------------------------
template_str = """
You are a helpful assistant chef.
Use the available tools to parse user ingredients and suggest substitutions.

Tools:
- Parse User Input
- Find Missing Ingredients
- Suggest Substitutions

{agent_scratchpad}
"""

human_msg = HumanMessagePromptTemplate.from_template(template_str)
prompt = ChatPromptTemplate.from_messages([human_msg])

# -----------------------------
# Create Agent Executor
# -----------------------------
agent_executor = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# -----------------------------
# Helper to Run Agent
# -----------------------------
def run_recipe_agent(
    recipe_title: str,
    recipe_ingredients: List[str],
    user_text: str,
    recipe_tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Runs the full agent workflow:
    1. Parse user input
    2. Identify missing ingredients
    3. Suggest substitutions
    """
    user_data = parse_user_input(user_text)
    user_ingredients = user_data.get("ingredients", [])

    missing_data = find_missing_ingredients(recipe_ingredients, user_ingredients)
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
