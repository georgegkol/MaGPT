import streamlit as st
from semantic_search import search_recipes, rerank_recipes

st.set_page_config(page_title="Grandma's Recipes", layout="wide")

st.title("Grandma's Recipes Finder 🍰")

query = st.text_input("Search for a recipe (any language):")

if query:
    with st.spinner("Finding recipes..."):
        candidates = search_recipes(query, top_n=20)
        final_matches = rerank_recipes(query, candidates)
    
    if final_matches:
        st.subheader("Top recipes:")
        for recipe in final_matches:
            with st.expander(f"{recipe['title']} (Score: {recipe['final_score']:.2f})"):
                st.markdown(f"**Ingredients:**")
                for ing in recipe['ingredients'] or []:
                    name = ing.get("name", "")
                    qty = ing.get("quantity") or ""
                    unit = ing.get("unit") or ""
                    st.text(f"- {name} {qty} {unit}".strip())
                
                st.markdown("**Instructions:**")
                for i, step in enumerate(recipe['instructions'] or []):
                    st.text(f"{i+1}. {step}")
                
                st.markdown(f"**Tags:** {', '.join(recipe.get('tags') or [])}")
                if recipe.get("total_time"):
                    st.text(f"Total time: {recipe['total_time']}")
                if recipe.get("servings"):
                    st.text(f"Servings: {recipe['servings']}")
    else:
        st.warning("No recipes found for your query.")
