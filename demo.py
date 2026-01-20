import streamlit as st
import operator
import json
import re
import docx
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# --- 1. State Schema ---
class RecipeAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ingredients: list[str]
    dietary_restrictions: list[str]
    max_cooking_time: int
    cuisine_preference: str
    matched_recipes: list[dict]

# --- 2. UPDATED Word File Loader for JSON ---
def load_recipes_from_word(file):
    file.seek(0)
    doc = docx.Document(file)
    # Combine all paragraphs into one string to get the full JSON
    full_text = "".join([para.text for para in doc.paragraphs])
    
    try:
        # Parse the string as JSON
        recipes = json.loads(full_text)
        return recipes
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON in Word doc: {e}")
        return []

# --- 3. Agent Nodes ---
import re

def extract_ingredients(state: RecipeAgentState):
    user_input = state["messages"][-1].content.lower()
    
    # 1. Remove common conversational phrases to clean the input
    clean_input = re.sub(r'(i have|i want|using|with|and|some|a |the |please|find|recipe)', ',', user_input)
    
    # 2. Split by commas or multiple spaces
    parts = re.split(r'[,]+', clean_input)
    
    # 3. Filter out "stop words" and empty strings
    # We ignore units, common verbs, and time indicators
    stop_words = {"i", "want", "for", "min", "minutes", "vegetarian", "vegan", "gluten-free", "any", "me", "a", "the", "recipe"}
    
    found = [word.strip() for word in parts if word.strip() and word.strip() not in stop_words and not word.isdigit()]
    
    # Optional: Deduplicate
    found = list(dict.fromkeys(found))
    
    return {"ingredients": found}

def extract_preferences(state: RecipeAgentState):
    user_input = state["messages"][-1].content.lower()
    time_match = re.search(r'(\d+)\s*min', user_input)
    time_val = int(time_match.group(1)) if time_match else 60 
    diet = [d for d in ["vegan", "vegetarian", "gluten-free"] if d in user_input]
    return {"dietary_restrictions": diet, "max_cooking_time": time_val}

def search_recipes(state: RecipeAgentState):
    all_recipes = load_recipes_from_word(st.session_state.file_obj)
    matches = []
    
    for r in all_recipes:
        # Check Time (Ensure key matches your JSON: "cooking_time")
        if r.get("cooking_time", 999) > state["max_cooking_time"]:
            continue
        
        # Check Dietary
        if state["dietary_restrictions"]:
            recipe_diet = [d.lower() for d in r.get("dietary", [])]
            if not any(d in recipe_diet for d in state["dietary_restrictions"]):
                continue
        
        matches.append(r)
    return {"matched_recipes": matches}

def rank_recipes(state: RecipeAgentState):
    user_available = [i.lower() for i in state["ingredients"]]
    scored_matches = []
    
    for recipe in state["matched_recipes"]:
        # Get all ingredients for this recipe in lowercase
        recipe_reqs = [i.lower() for i in recipe.get("ingredients", [])]
        
        matched_items_list = []
        score = 0
        
        # KEYWORD MATCHING LOGIC
        for user_ing in user_available:
            for recipe_ing in recipe_reqs:
                # Check if "lentils" is inside "red lentils"
                if user_ing in recipe_ing:
                    score += 1
                    matched_items_list.append(recipe_ing)
                    break # Move to next user ingredient to avoid double-counting
        
        recipe["score"] = score
        recipe["matched_items_list"] = list(set(matched_items_list)) # Unique matches
        scored_matches.append(recipe)
    
    # Sort by score descending
    ranked = sorted(scored_matches, key=lambda x: x["score"], reverse=True)
    
    return {"matched_recipes": ranked}

def generate_recommendation(state: RecipeAgentState):
    """
    Only recommends recipes that have at least one matching ingredient.
    """
    # Filter the list to only include recipes where score > 0
    top_recipes = [r for r in state["matched_recipes"] if r.get("score", 0) > 0]
    
    # Take only the top 3 of the filtered list
    top_recipes = top_recipes[:3]
    
    if not top_recipes:
        # Custom message if everything was filtered out or score was 0
        msg = ("🔍 I found recipes that fit your time and diet, but **none of them use the ingredients** you mentioned "
               f"({', '.join(state['ingredients'])}). \n\n"
               "Try adding different ingredients or check your recipe file!")
    else:
        msg = f"### 🧑‍🍳 Best Matches for your Ingredients:\n"
        for r in top_recipes:
            matched_str = ", ".join(r.get("matched_items_list", []))
            msg += f"- **{r['name']}** | Score: {r['score']} (Uses: {matched_str})\n"
            msg += f"  *Time: {r['cooking_time']} min | Cuisine: {r['cuisine']}*\n\n"
        
    return {"messages": [AIMessage(content=msg)]}

# --- 4. Streamlit UI ---
st.set_page_config(page_title="JSON Recipe Agent", page_icon="📝")
st.title("📝 JSON-Doc Recipe Agent")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Word Doc (JSON format)", type="docx")
    if uploaded_file:
        st.session_state.file_obj = uploaded_file
        test_data = load_recipes_from_word(uploaded_file)
        if test_data:
            st.success(f"✅ Successfully parsed {len(test_data)} recipes!")

if prompt := st.chat_input("Ex: I want pasta and eggs, 30 min"):
    if not uploaded_file:
        st.error("Please upload the Word file first.")
    else:
        st.chat_message("user").write(prompt)
        
        # Graph Setup
        workflow = StateGraph(RecipeAgentState)
        workflow.add_node("extract_ingredients", extract_ingredients)
        workflow.add_node("extract_preferences", extract_preferences)
        workflow.add_node("search_recipes", search_recipes)
        workflow.add_node("rank_recipes", rank_recipes)
        workflow.add_node("generate_recommendation", generate_recommendation)
        
        workflow.set_entry_point("extract_ingredients")
        workflow.add_edge("extract_ingredients", "extract_preferences")
        workflow.add_edge("extract_preferences", "search_recipes")
        workflow.add_edge("search_recipes", "rank_recipes")
        workflow.add_edge("rank_recipes", "generate_recommendation")
        workflow.add_edge("generate_recommendation", END)
        
        app = workflow.compile()
        
        # Initial State
        result = app.invoke({
            "messages": [HumanMessage(content=prompt)],
            "ingredients": [],
            "dietary_restrictions": [],
            "max_cooking_time": 60,
            "cuisine_preference": "Any",
            "matched_recipes": []
        })
        # --- DEBUG VIEW ---
        with st.expander("🔍 Agent Logic (Debug)"):
            st.write(f"**Ingredients Found:** {result['ingredients']}")
            st.write(f"**Max Time:** {result['max_cooking_time']} min")
            st.write(f"**Dietary:** {result['dietary_restrictions']}")
            st.write(f"**Total Matches Found:** {len(result['matched_recipes'])}")
        
        st.chat_message("assistant").markdown(result["messages"][-1].content)