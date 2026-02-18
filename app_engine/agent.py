import json
from typing import List, Dict, Any, Optional
from tools import find_missing_ingredients
from openai import OpenAI
from dotenv import load_dotenv
import os

# LangChain imports
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, create_openai_functions_agent


load_dotenv()
client = OpenAI()


# -----------------------------
# LLM TOOL 1: PARSE USER INPUT
# -----------------------------
def parse_user_input(user_text: str) -> Dict[str, Any]:
    """
    Takes natural language input and returns a structured list of ingredients.
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

    # Strip any extra whitespace or text outside JSON
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("JSON parse error:", e)
        print("Model output was:", content)
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

Suggest 2–3 substitutions per missing ingredient only, not for all ingredients.
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


# LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Tools
tools = [
    {"name": "Parse User Input", "func": parse_user_input, "description": "Parse natural language text into structured ingredient list."},
    {"name": "Find Missing Ingredients", "func": find_missing_ingredients, "description": "Compare recipe ingredients with user ingredients."},
    {"name": "Suggest Substitutions", "func": suggest_substitutions, "description": "Suggest ingredient substitutions for missing items."},
]

# Prompt template with agent_scratchpad
template_str = """
You are a helpful assistant chef. 
Use the available tools to parse user ingredients and suggest substitutions.

Tools:
- Parse User Input
- Find Missing Ingredients
- Suggest Substitutions

{agent_scratchpad}
"""

# Wrap your existing template
human_msg = HumanMessagePromptTemplate.from_template(template_str)

# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([human_msg])

# Create agent executor
agent_executor = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# -----------------------------
# HELPER FUNCTION TO RUN AGENT
# -----------------------------
def run_recipe_agent(
    recipe_title: str,
    recipe_ingredients: List[str],
    user_text: str,
    recipe_tags: Optional[List[str]] = None
):
    user_data = parse_user_input(user_text)  # extracts {"ingredients": ["onion", "tomato"]}
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
