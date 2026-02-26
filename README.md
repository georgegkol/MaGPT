# MaGPT

This project is an interactive recipe assistant that allows users to search, explore, and get cooking recommendations from a collection of handwritten recipes. Users can ask natural language questions about recipes, including ingredient substitutions or missing ingredients, and the agent provides context-aware answers.

The recipes are digitized using a fine-tuned Transformer OCR model, ensuring that the original handwriting is accurately captured and stored. Behind the scenes, an LLM-powered agent built with LangChain and OpenAI APIs handles user queries, retrieves relevant recipes, and generates intelligent recommendations.

The application is deployed as an interactive Streamlit web app, showcasing the full pipeline from handwritten recipe scans to AI-powered recipe guidance.
