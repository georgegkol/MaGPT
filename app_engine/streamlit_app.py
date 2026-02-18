import streamlit as st
from semantic_search import search_recipes
from agent import run_recipe_agent, parse_user_input

st.set_page_config(page_title="Grandma's Recipes", layout="wide")
st.title("Grandma's Recipes Finder 🍲")

# --- Step 1: Recipe Search ---
query = st.text_input("Search for a recipe (any language):")

if query:
    with st.spinner("Finding recipes..."):
        results = search_recipes(query, top_n=20)

    if not results:
        st.warning("No recipes found for your query.")
    else:
        st.subheader("Top recipes")
        
        for idx, recipe in enumerate(results):
            # Safe score handling
            score = recipe.get("final_score") or recipe.get("score") or recipe.get("pinecone_score")
            title = recipe.get("title", f"Untitled recipe {idx}")
            if score is not None:
                title = f"{title} (Score: {score:.2f})"

            with st.expander(title):
                # --- Recipe Details ---
                st.markdown("### 🧂 Ingredients")
                ingredients = recipe.get("ingredients") or []
                recipe_ingredient_names = [ing.get("name") for ing in ingredients if ing.get("name")]

                if ingredients:
                    for ing in ingredients:
                        name = ing.get("name", "")
                        qty = ing.get("quantity") or ""
                        unit = ing.get("unit") or ""
                        line = " ".join(str(x) for x in [qty, unit, name] if x)
                        st.text(f"• {line}")
                else:
                    st.text("No ingredients listed.")

                st.markdown("### 📋 Instructions")
                instructions = recipe.get("instructions") or []
                if instructions:
                    for i, step in enumerate(instructions, 1):
                        st.text(f"{i}. {step}")
                else:
                    st.text("No instructions available.")

                # --- User Ingredients Input ---
                st.markdown("### 🥄 Enter Ingredients You Have")
                user_input = st.text_area(
                    f"Enter your ingredients for '{recipe.get('title', 'this recipe')}' (one per line or free text):",
                    key=f"user_input_{idx}"
                )

                if st.button("Check Ingredients & Get Suggestions", key=f"check_{idx}") and user_input.strip():
                    with st.spinner("Analyzing your ingredients..."):
                        # Parse user input via LLM
                        user_data = parse_user_input(user_input)
                        user_ingredient_names = user_data.get("ingredients", [])

                        agent_result = run_recipe_agent(
                            recipe_title=recipe.get("title", ""),
                            recipe_ingredients=recipe_ingredient_names,
                            user_text=user_input_text,  # the raw string the user typed in Streamlit
                            recipe_tags=recipe.get("tags", [])
                        )

                    # --- Display Results ---
                    st.markdown("### ✅ Ingredients You Have")
                    st.text(", ".join(agent_result["available"]) or "None")

                    st.markdown("### ⚠️ Missing Ingredients")
                    st.text(", ".join(agent_result["missing"]) or "None")

                    st.markdown("### 🔄 Suggested Substitutions")
                    substitutions = agent_result.get("substitutions", {})
                    if substitutions:
                        for miss, subs in substitutions.items():
                            st.text(f"{miss} → {', '.join(subs) if subs else 'No good substitutes'}")
                    else:
                        st.text("No substitutions suggested.")

                # Optional meta
                st.markdown("### ℹ️ Details")
                if recipe.get("tags"):
                    st.text(f"Tags: {', '.join(recipe['tags'])}")
                if recipe.get("total_time"):
                    st.text(f"Total time: {recipe['total_time']}")
