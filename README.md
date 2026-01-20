#  AI Recipe & Kitchen Assistant

A stateful, graph-based agent that parses recipes from Word documents, validates user intent, and provides intelligent cooking recommendations using **LangGraph** and **Groq**.

---

## <u>SYSTEM ARCHITECTURE</u>

The agent uses a Directed Acyclic Graph (DAG) to manage the conversation flow. This ensures that the agent doesn't just "guess," but follows a logical path from validation to instruction. can also capture logging errors,warnings in logfile.log and has pytest in test_sample.py



---

## <u>CORE FEATURES</u>

* **INTENT VALIDATION**: Uses Llama 3.1 to ensure the agent only discusses food. It will politely decline requests about stocks, trading, or politics.
* **DOCX DATA EXTRACTION**: Custom logic to read and parse JSON-formatted recipe databases stored inside `.docx` files.
* **INTELLIGENT RANKING**: Matches user ingredients against recipe requirements and scores them based on overlap.
* **STEP-BY-STEP INSTRUCTIONS**: Can transition from listing matches to providing a full culinary guide for a specific dish.
* **INTERACTIVE WORKFLOW**: Real-time visualization of the LangGraph logic directly in the Streamlit sidebar.

---

## <u>INSTALLATION & SETUP</u>

### 1. CLONE THE REPOSITORY
```
git clone (https://github.com/visali231996/recipe_agent.git)
cd recipe_agent
```
### 2.INSTALL DEPENDENCIES

Run the following command in your terminal to install the required Python libraries:

`pip install streamlit langgraph langchain_groq python-docx python-dotenv`

### 3. LAUNCH THE APP

```bash
streamlit run sample.py
```

### 4. RUNNING PYTEST
```
pytest test_sample.py
```



