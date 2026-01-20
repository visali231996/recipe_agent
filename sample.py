import streamlit as st
import operator
import json
import re
import docx
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq # Or use ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import logging

# Configure the logging settings
logging.basicConfig(filename='logfile.log', level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Initialize LLM ---
# Replace with your API key and preferred provider
def get_llm():
    """Returns the LLM instance. Easy to mock in pytest."""
    return ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY")
    )

# For Streamlit use:
llm = get_llm()

# --- 1. State Schema ---
class RecipeAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ingredients: list[str]
    dietary_restrictions: list[str]
    max_cooking_time: int
    cuisine_preference: str
    matched_recipes: list[dict]
    is_valid: bool  # New field to track validity
    intent: str

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
        logging.error("File uploaded is not in json format")
        return []

# --- 3. New Validation Node ---
def validate_intent(state: RecipeAgentState):
    """Uses LLM to check if the user is asking about food/cooking."""
    user_input = state["messages"][-1].content
    
    prompt = f"""
    You are a kitchen assistant. Determine if the following user input is related to 
    food, cooking, recipes, or dietary preferences. 
    
    If it is about trading, stocks, politics, or any non-cooking topic, respond with 'INVALID'.
    If it is related to food/cooking, respond with 'VALID'.
    If it is NOT about food, respond: INVALID
    If the user is asking HOW to cook a specific dish (e.g., "give me the recipe for..."), respond: INSTRUCTION
    If the user is listing ingredients or asking for suggestions, respond: SEARCH
    
    Respond with ONLY one word.
    
    User Input: "{user_input}"
    Response:"""
    
    response = llm.invoke(prompt).content.strip().upper()
    
    if "INVALID" in response:
        logging.warning("INVALID inputs given to the agent")
        return {
            "is_valid": False,"intent" :"none", 
            "messages": [AIMessage(content="I'm sorry, I'm a specialized cooking assistant. I can't help with trading or other non-food topics.")]
        }
    if "INSTRUCTION" in response:
        return {"is_valid": True, "intent": "instruction"}
    return {"is_valid": True, "intent": "search"}

# --- Routing Logic ---
def route_after_validation(state: RecipeAgentState):
    if not state["is_valid"]:
        return END
    if state["intent"] == "instruction":
        return "get_recipe"  # Jump straight to instructions
    return "extract_ingredients" # Proceed to search

# --- Existing Extraction Nodes (Slightly Updated) ---
def extract_ingredients(state: RecipeAgentState):
    user_input = state["messages"][-1].content
    # Tell the LLM to be very strict with the format
    prompt = f"""Extract the food ingredients from this text: "{user_input}". 
    Return ONLY a comma-separated list of ingredients. 
    No periods, no introductory text. If none, return 'None'."""
    
    res = llm.invoke(prompt).content
    
    if "None" in res:
        logging.error("LLM did'nt extract ingredients properly")
        return {"ingredients": []}
    
    
    # CLEANING LOGIC: Remove punctuation like periods or extra dashes
    raw_list = res.split(",")
    found = []
    for item in raw_list:
        clean_item = re.sub(r'[^\w\s]', '', item).strip().lower()
        if clean_item:
            found.append(clean_item)
            
    return {"ingredients": found}

def extract_preferences(state: RecipeAgentState):
    user_input = state["messages"][-1].content.lower()
    time_match = re.search(r'(\d+)\s*min', user_input)
    time_val = int(time_match.group(1)) if time_match else 60 
    diet = [d for d in ["vegan", "vegetarian", "gluten-free"] if d in user_input]
    return {"dietary_restrictions": diet, "max_cooking_time": time_val}

# [Remaining nodes: search_recipes, rank_recipes, generate_recommendation stay the same as your previous code]

# --- 4. Streamlit UI & Graph Construction ---
# (Standard UI code here...)
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
def get_recipe(state: RecipeAgentState):
    """Generates the full recipe for the SPECIFIC item the user asked for."""
    user_msg = state["messages"][-1].content.lower()
    
    # Search for which of our matched recipes the user is talking about
    target_recipe = None
    for r in state["matched_recipes"]:
        if r["name"].lower() in user_msg:
            target_recipe = r
            break
            
    if not target_recipe:
        return {"messages": [AIMessage(content="I'm not sure which recipe you'd like. Could you please type the exact name?")]}

    prompt = f"""
    You are an expert chef. The user wants the full recipe for: {target_recipe['name']}.
    Ingredients available: {', '.join(target_recipe['ingredients'])}.
    Please provide the full instructions, portions, and cooking steps.
    """
    
    detailed_recipe = llm.invoke(prompt).content
    return {"messages": [AIMessage(content=detailed_recipe)]}

def route_after_recommendation(state: RecipeAgentState) -> Literal["get_recipe", END]:
    user_msg = state["messages"][-1].content.lower()
    
    # 1. Define explicit "Recipe Request" keywords
    instruction_keywords = ["how to make", "give me the recipe", "instructions", "steps", "how do i cook"]
    
    # 2. Check if the user is explicitly asking "how" to do something
    is_asking_how = any(k in user_msg for k in instruction_keywords)
    
    # 3. Check if they mentioned a recipe name AND asked "how"
    # This prevents the initial search (e.g., "i want pasta") from triggering get_recipe
    any_recipe_mentioned = any(r["name"].lower() in user_msg for r in state["matched_recipes"])
    
    if is_asking_how and any_recipe_mentioned:
        return "get_recipe"
    
    # Otherwise, just end (stop at the recommendation list)
    return END


st.set_page_config(page_title="JSON Recipe Agent", page_icon="📝")
st.title("📝 JSON-Doc Recipe Agent")
# Initialize history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize this at the top of your app
if "matched_recipes" not in st.session_state:
    st.session_state.matched_recipes = []

# Display entire chat history on every rerun
for msg in st.session_state.messages:
    # msg is a LangChain object, so we check .type or isinstance
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Word Doc (JSON format)", type="docx")
    if uploaded_file:
        st.session_state.file_obj = uploaded_file
        test_data = load_recipes_from_word(uploaded_file)
        if test_data:
            st.success(f"✅ Successfully parsed {len(test_data)} recipes!")
if prompt := st.chat_input("Ex: I want pasta and eggs, 30 min"):
    # ... inside the prompt block:
    # 1. Immediately show user message and save to state
    st.chat_message("user").markdown(prompt)
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    
    workflow = StateGraph(RecipeAgentState)
    
    # Add Nodes
    workflow.add_node("validate_intent", validate_intent)
    workflow.add_node("extract_ingredients", extract_ingredients)
    workflow.add_node("extract_preferences", extract_preferences)
    workflow.add_node("search_recipes", search_recipes)
    workflow.add_node("rank_recipes", rank_recipes)
    workflow.add_node("generate_recommendation", generate_recommendation)
    workflow.add_node("get_recipe", get_recipe)
    
    # Set Entry and Conditional Edges
    workflow.set_entry_point("validate_intent")
    workflow.add_conditional_edges(
    "validate_intent",
    route_after_validation,
    {
        "extract_ingredients": "extract_ingredients",
        "get_recipe": "get_recipe",
        END: END
    }
)
    
    # Standard Edges
    workflow.add_edge("extract_ingredients", "extract_preferences")
    workflow.add_edge("extract_preferences", "search_recipes")
    workflow.add_edge("search_recipes", "rank_recipes")
    workflow.add_edge("rank_recipes", "generate_recommendation")
    workflow.add_conditional_edges(
        "generate_recommendation",
        route_after_recommendation
    )
    workflow.add_edge("get_recipe",END )
    
    app = workflow.compile()
    
    # 2. Generate the diagram as bytes (PNG)
    # Ensure you have 'pygraphviz' or 'graphviz' installed, or use 'mermaid'
    try:
        graph_png = app.get_graph().draw_mermaid_png()
        
        # 3. Display it in the Streamlit Sidebar or Main page
        with st.sidebar:
            st.subheader("Agent Workflow")
            st.image(graph_png)
    except Exception as e:
        # If graphviz isn't installed, this might fail silently
        print(f"Could not generate graph: {e}")


    # Initial State
    initial_state = {
        "messages": st.session_state.messages,
        "ingredients": [],
        "dietary_restrictions": [],
        "max_cooking_time": 60,
        "cuisine_preference": "Any",
        "matched_recipes": st.session_state.matched_recipes,
        "is_valid": True
    }
    
    

    with st.spinner("Chef is thinking..."):
        result = app.invoke(initial_state)
    
    # 3. Extract the LAST message from the result (the Assistant's response)
    response_msg = result["messages"][-1]

    st.session_state.matched_recipes = result.get("matched_recipes", [])
    
    # 4. Show assistant message and save to state
    with st.chat_message("assistant"):
        st.markdown(response_msg.content)
    
    st.session_state.messages.append(response_msg)