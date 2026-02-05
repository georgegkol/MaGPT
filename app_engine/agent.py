from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from tools import find_missing_ingredients, scale_ingredients
import json

# -----------------------------
# LLM WRAPPER FOR SUBSTITUTIONS
# -----------------------------
def suggest_substitutions_lc(missing_ingredients, recipe_title, recipe_tags=None):
    tags_text = ", ".join(recipe_tags) if recipe_tags else "none"

    prompt_template = """
    You are a professional chef.
    Recipe: {recipe_title}
    Recipe tags: {tags_text}
    Missing ingredients: {missing_ingredients}

    Suggest 2-3 substitutions per missing ingredient.
    Return ONLY valid JSON: {{ "ingredient_name": ["sub1","sub2"] }}
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
        return {}

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
        func=suggest_substitutions_lc,
        description="Provide substitutions for missing ingredients."
    ),
    Tool(
        name="Scale Ingredients",
        func=scale_ingredients,
        description="Scale recipe ingredients based on user ingredients."
    )
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
# HELPER FUNCTION TO RUN AGENT
# -----------------------------
def run_recipe_agent(recipe_name, recipe_ingredients, user_ingredients, recipe_tags=None):
    """
    recipe_name: str
    recipe_ingredients: List[str] or Dict[str,(qty,unit)]
    user_ingredients: List[str] or Dict[str,(qty,unit)]
    recipe_tags: Optional list of tags
    """
    # Step 1: find missing ingredients
    missing_data = find_missing_ingredients(recipe_ingredients, user_ingredients)
    available = missing_data["available"]
    missing = missing_data["missing"]

    # Step 2: suggest substitutions
    substitutions = {}
    if missing:
        substitutions = suggest_substitutions_lc(missing, recipe_name, recipe_tags)

    # Step 3: scale ingredients (if using quantities)
    scaled = {}
    if isinstance(recipe_ingredients, dict) and isinstance(user_ingredients, dict):
        scaled = scale_ingredients(recipe_ingredients, user_ingredients)

    return {
        "available": available,
        "missing": missing,
        "substitutions": substitutions,
        "scaled": scaled
    }