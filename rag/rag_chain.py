import sys
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

try:
    from rag.retriever import FAISSRetriever
except ImportError:
    from retriever import FAISSRetriever

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize LLM (OpenAI if key exists, else Ollama)
def get_llm():
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        except Exception:
            pass
            
    try:
        from langchain_ollama import OllamaLLM
        model = OllamaLLM(model="qwen2.5:3b")
        # Basic check if Ollama is reachable (optional, might slow down startup)
        return model
    except Exception:
        pass

    # Fallback or Mock if neither is available or reachable
    class MockLLM:
        def invoke(self, prompt, **kwargs):
            return "AI Response: Based on your request, I recommend looking at the products listed below. (Ollama/OpenAI not reachable, showing mock response)"
        
        # Compatibility with LCEL
        def __or__(self, other):
            from langchain_core.runnables import RunnableParallel
            return self
        
        # Handle chain invocation if used directly
        def __call__(self, *args, **kwargs):
            return self.invoke(args[0] if args else "")

    return MockLLM()

llm = get_llm()

# 2. Initialize Retriever
retriever = FAISSRetriever(top_k=3)

# 3. Load Prompts from YAML
PROMPTS_FILE = BASE_DIR / "rag" / "prompts" / "prompt_templates.yaml"
try:
    with open(PROMPTS_FILE, "r") as f:
        PROMPTS = yaml.safe_load(f)
except Exception as e:
    print(f"Error loading prompt templates at {PROMPTS_FILE}: {e}")
    PROMPTS = {}

def get_best_prompt(query, preferences=""):
    """
    Selects the most relevant prompt template based on keywords 
    in the query or preferences.
    """
    text = (query + " " + preferences).lower()
    
    if "budget" in text or "cheap" in text or "affordable" in text:
        return PROMPTS.get("budget_recommendation")
    if "luxury" in text or "premium" in text or "expensive" in text:
        return PROMPTS.get("luxury_recommendation")
    if "wood" in text or "oak" in text or "pine" in text or "teak" in text:
        return PROMPTS.get("wooden_recommendation")
    if "office" in text or "work" in text or "desk" in text or "study" in text:
        return PROMPTS.get("office_recommendation")
    if "minimal" in text or "simple" in text or "scandi" in text:
        return PROMPTS.get("minimal_recommendation")
    if "industrial" in text or "metal" in text or "raw" in text:
        return PROMPTS.get("industrial_recommendation")
    if "modern" in text or "contemporary" in text or "trend" in text:
        return PROMPTS.get("modern_recommendation")
        
    return PROMPTS.get("standard_recommendation")

def format_inr(amount_usd):
    """
    Converts USD to INR (1 USD = 83 INR) and formats with Indian numbering system.
    Example: 1000 USD -> ₹83,000
    """
    amount_inr = int(float(amount_usd) * 83)
    s = str(amount_inr)
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    # Indian numbering: commas after first 3 digits, then every 2 digits
    out = ""
    while len(remaining) > 2:
        out = "," + remaining[-2:] + out
        remaining = remaining[:-2]
    if remaining:
        out = remaining + out
    return f"₹{out},{last_three}"

def format_retrieved_products(products):
    """
    Converts the list of product dictionaries into a formatted string 
    for the LLM context, with prices in INR.
    """
    if not products:
        return "No matching products found in the catalog."
    
    formatted_list = []
    for i, product in enumerate(products, 1):
        usd_price = product.get('price', 0)
        inr_price = format_inr(usd_price)
        
        item = (
            f"Product {i}:\n"
            f"- Title: {product.get('title')}\n"
            f"- Category: {product.get('category')}\n"
            f"- Price: {inr_price}\n"
            f"- Color: {product.get('color')}\n"
            f"- Style: {product.get('style')}\n"
            f"- Material: {product.get('material')}\n"
            f"- Description: {product.get('description')}\n"
        )
        formatted_list.append(item)
    
    return "\n".join(formatted_list)

def generate_recommendation(query, preferences=""):
    """
    Modular RAG Workflow:
    1. Retrieve Products: Uses the FAISSRetriever (existing FAISS pipeline).
    2. Format Context: Prepares product metadata for the prompt.
    3. Generate AI Recommendation: Uses Ollama to create a structured recommendation response.
    """
    print(f"Searching for: '{query}'...")
    
    try:
        # Step 1: Retrieval
        retrieved_products = retriever.get_relevant_products(query)
        
        if not retrieved_products:
            return "ERR_NO_PRODUCTS"

        # Step 2: Context Preparation
        context = format_retrieved_products(retrieved_products)

        # Step 3: LLM Generation
        selected_template = get_best_prompt(query, preferences)
        
        if not selected_template:
            return "ERR_TEMPLATE_LOAD"

        prompt = PromptTemplate(
            input_variables=["context", "query", "preferences"],
            template=selected_template
        )
        
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "context": context,
            "query": query,
            "preferences": preferences
        })
        return response
        
    except Exception as e:
        print(f"RAG Chain Error: {e}")
        return "ERR_CONNECTION"

if __name__ == "__main__":
    # Quick Test
    query = "I need a stylish wooden wardrobe for my bedroom."
    prefs = "Budget is around $500."
    
    print("--- GenAI Recommender System ---")
    recommendation = generate_recommendation(query, prefs)
    print("\nRecommendation Result:\n")
    print(recommendation)
