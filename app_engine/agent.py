from langchain.tools import StructuredTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage

from tools import (
    parse_user_input,
    find_missing_ingredients,
    suggest_substitutions,
)

# -----------------------------
# WRAP TOOLS
# -----------------------------

parse_tool = StructuredTool.from_function(
    func=parse_user_input,
    name="parse_user_input",
    description="Extract ingredients and optional quantities from a user's natural language input."
)

missing_tool = StructuredTool.from_function(
    func=find_missing_ingredients,
    name="find_missing_ingredients",
    description="Compare recipe ingredients with user ingredients to determine which are missing."
)

substitution_tool = StructuredTool.from_function(
    func=suggest_substitutions,
    name="suggest_substitutions",
    description="Suggest ingredient substitutions for missing recipe ingredients."
)

tools = [parse_tool, missing_tool, substitution_tool]

# -----------------------------
# LLM
# -----------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -----------------------------
# PROMPT
# -----------------------------

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are an intelligent cooking assistant.

When a user provides their available ingredients:

1. Parse the user input.
2. Compare it with the recipe ingredients.
3. Identify missing ingredients.
4. If ingredients are missing, suggest substitutions.
5. Respond clearly and helpfully.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# -----------------------------
# CREATE AGENT
# -----------------------------

agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# -----------------------------
# RUN FUNCTION
# -----------------------------

def run_agent(user_message: str, recipe_ingredients: list, recipe_title: str, recipe_tags=None):
    """
    Runs the autonomous recipe agent.
    """

    input_data = {
        "input": user_message,
        "recipe_ingredients": recipe_ingredients,
        "recipe_title": recipe_title,
        "recipe_tags": recipe_tags or []
    }

    return agent_executor.invoke(input_data)
