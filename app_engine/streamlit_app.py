import streamlit as st
from semantic_search import search_recipes

st.set_page_config(
    page_title="Grandma's Recipes",
    layout="wide"
)

st.title("Grandma's Recipes Finder 🍲")

query = st.text_input("Search for a recipe (any language):")

if query:
    with st.spinner("Finding recipes..."):
        results = search_recipes(query, top_n=20)

    if not results:
        st.warning("No recipes found for your query.")
    else:
        st.subheader("Top recipes")

        for recipe in results:
            # --- SAFE SCORE HANDLING ---
            score = (
                recipe.get("final_score")
                or recipe.get("score")
                or recipe.get("pinecone_score")
            )

            title = recipe.get("title", "Untitled recipe")
            if score is not None:
                title = f"{title} (Score: {score:.2f})"

            with st.expander(title):
                # --- INGREDIENTS ---
                st.markdown("### 🧂 Ingredients")
                ingredients = recipe.get("ingredients") or []

                if ingredients:
                    for ing in ingredients:
                        name = ing.get("name", "")
                        qty = ing.get("quantity") or ""
                        unit = ing.get("unit") or ""
                        line = " ".join(str(x) for x in [qty, unit, name] if x)
                        st.text(f"• {line}")
                else:
                    st.text("No ingredients listed.")

                # --- INSTRUCTIONS ---
                st.markdown("### 📋 Instructions")
                instructions = recipe.get("instructions") or []

                if instructions:
                    for i, step in enumerate(instructions, 1):
                        st.text(step)
                else:
                    st.text("No instructions available.")

                # --- META ---
                st.markdown("### ℹ️ Details")
                if recipe.get("tags"):
                    st.text(f"Tags: {', '.join(recipe['tags'])}")
                if recipe.get("total_time"):
                    st.text(f"Total time: {recipe['total_time']}")
                if recipe.get("servings"):
                    st.text(f"Servings: {recipe['servings']}")
