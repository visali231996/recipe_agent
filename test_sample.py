import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
# Replace 'sample' with your actual filename if different
from sample import extract_ingredients, rank_recipes, search_recipes

# --- 1. Fixtures ---

@pytest.fixture
def mock_recipe_data():
    return [
        {"name": "Carbonara", "ingredients": ["spaghetti", "eggs", "pancetta"], "cooking_time": 20, "dietary": []},
        {"name": "Beef Tacos", "ingredients": ["beef", "tortilla"], "cooking_time": 15, "dietary": []},
        {"name": "Lentil Soup", "ingredients": ["lentils", "water"], "cooking_time": 30, "dietary": ["vegan", "vegetarian"]}
    ]

# --- 2. Testing Extraction (Mocking LLM) ---

@patch("sample.llm") # We patch the global llm object in your sample.py
def test_extract_ingredients(mock_llm,):
    # Setup the mock to return a specific string
    mock_llm.invoke.return_value = MagicMock(content="olive oil, palak.")
    
    state = {"messages": [HumanMessage(content="I want olive oil and palak")]}
    result = extract_ingredients(state)
    
    # Check if cleaning logic works (no period, lowercase)
    assert "olive oil" in result["ingredients"]
    assert "eggs" in result["ingredients"]
    assert "palak" not in result["ingredients"]

def test_rank_recipes_scoring():
    state = {
        "ingredients": ["eggs", "spaghetti"],
        "matched_recipes": [
            {"name": "Carbonara", "ingredients": ["spaghetti", "eggs", "bacon"]},
            {"name": "Beef Tacos", "ingredients": ["beef", "shells"]}
        ]
    }
    result = rank_recipes(state)
    
    # Carbonara should be first with score 2
    assert result["matched_recipes"][0]["name"] == "Carbonara"
    assert result["matched_recipes"][0]["score"] == 2
    # Tacos should have score 0
    assert result["matched_recipes"][1]["score"] == 0

# --- 4. Testing Search (Mocking Streamlit State) ---

@patch("sample.load_recipes_from_word")
@patch("sample.st.session_state")
def test_search_recipes_time_filter(mock_st_state, mock_loader, mock_recipe_data):
    # Mock Streamlit session state and the Word loader
    mock_st_state.file_obj = "fake_file"
    mock_loader.return_value = mock_recipe_data
    
    # Test for 20 mins max
    state = {
        "max_cooking_time": 20,
        "dietary_restrictions": [],
    }
    
    result = search_recipes(state)
    
    # Carbonara (20) and Tacos (15) should stay, Lentil Soup (30) should be gone
    names = [r["name"] for r in result["matched_recipes"]]
    assert "Carbonara" in names
    assert "Beef Tacos" in names
    assert "Lentil Soup" not in names



