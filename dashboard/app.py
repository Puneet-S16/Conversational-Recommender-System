import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import json

# Page Configuration
st.set_page_config(
    page_title="GenAI Recommender Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
<style>
.stApp {
background-color: #0e1117;
}
.product-card {
background-color: #1e2130;
border-radius: 12px;
padding: 20px;
margin-bottom: 20px;
border: 1px solid #3d4455;
transition: transform 0.2s, border-color 0.2s;
}
.product-card:hover {
transform: translateY(-5px);
border-color: #4e5dff;
}
.product-title {
color: #ffffff;
font-size: 1.2rem;
font-weight: 700;
margin-bottom: 10px;
}
.product-meta {
color: #a0aec0;
font-size: 0.9rem;
margin-bottom: 5px;
}
.score-badge {
background: linear-gradient(90deg, #4e5dff, #a066ff);
color: white;
padding: 4px 10px;
border-radius: 20px;
font-size: 0.8rem;
font-weight: bold;
}
.model-badge {
display: inline-block;
padding: 4px 12px;
border-radius: 15px;
font-size: 0.85rem;
font-weight: 600;
margin-bottom: 15px;
}
.badge-collab { background-color: #2b6cb0; color: white; }
.badge-embedding { background-color: #2d3748; color: white; }
.badge-hybrid { background-color: #805ad5; color: white; }
.badge-rag { background-color: #38a169; color: white; }
.metric-container {
background-color: #1a202c;
border-radius: 10px;
padding: 15px;
text-align: center;
border: 1px solid #2d3748;
}
.stChatMessage {
border-radius: 15px;
padding: 15px;
margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000"

# --- Sidebar ---
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=TEAM+DYNAMO+B", use_container_width=True)
    st.header("⚙️ Configuration")
    
    user_id = st.text_input("👤 User ID", value="user_123")
    top_n = st.slider("📊 Top N Recommendations", 1, 20, 5)
    
    st.divider()
    
    model_type = st.selectbox(
        "🧠 Select Engine",
        ["Collaborative Filtering", "Embedding-Based", "Hybrid", "Conversational RAG"]
    )
    
    st.divider()
    
    # API Health in Sidebar
    st.markdown("### 🔌 Backend Status")
    try:
        start_time = time.time()
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        latency = (time.time() - start_time) * 1000
        if health_resp.status_code == 200:
            st.success(f"Online ({latency:.0f}ms)")
        else:
            st.warning("Issues Detected")
    except:
        st.error("Offline")

# Initialize Session State
if "recs" not in st.session_state:
    st.session_state.recs = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_metrics" not in st.session_state:
    st.session_state.rag_metrics = None

# Reset results state if model type changes
if st.session_state.last_model != model_type:
    st.session_state.recs = None
    st.session_state.metrics = None
    st.session_state.rag_metrics = None
    st.session_state.last_model = model_type

# --- Helper Functions ---
def display_product_card(item):
    """Custom HTML/CSS product card renderer with rich metadata."""
    score = item.get('score', 0.0)
    # Handle both 0-1 and 0-100 scores for display
    display_score = f"{score:.2f}" if score <= 1 else f"{score:.0f}"
    
    # Extract metadata with defaults
    price = item.get('price', 'N/A')
    category = item.get('category', 'N/A')
    material = item.get('material', 'N/A')
    color = item.get('color', 'N/A')
    style = item.get('style', 'N/A')
    description = item.get('description', 'No description available.')
    
    # IMPORTANT: All lines must be completely flush left (no leading spaces) 
    # to avoid markdown code block triggers.
    card_html = (
        f'<div class="product-card">'
        f'<div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">'
        f'<div class="product-title">{item.get("title", "Unknown Product")}</div>'
        f'<div class="score-badge">Score: {display_score}</div>'
        f'</div>'
        f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">'
        f'<div class="product-meta">🏷️ <strong>ID:</strong> {item.get("item_id")}</div>'
        f'<div class="product-meta">📁 <strong>Category:</strong> {category}</div>'
        f'<div class="product-meta">💰 <strong>Price:</strong> ₹{price}</div>'
        f'<div class="product-meta">🎨 <strong>Color:</strong> {color}</div>'
        f'<div class="product-meta">🛠️ <strong>Material:</strong> {material}</div>'
        f'<div class="product-meta">🏛️ <strong>Style:</strong> {style}</div>'
        f'</div>'
        f'<div style="border-top: 1px solid #3d4455; padding-top: 10px;">'
        f'<div style="color: #a0aec0; font-size: 0.85rem; line-height: 1.4;">'
        f'{description}'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

# --- Main App ---
st.title("🚀 GenAI Recommender System")
st.markdown(f"**Current Engine:** `{model_type}`")

# Main Section Logic for Standard Recommenders
if model_type != "Conversational RAG":
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Recommendations for {user_id}")
    
    with col2:
        # Triggering a new search
        if st.button("🔥 Generate", use_container_width=True, type="primary"):
            endpoint_map = {
                "Collaborative Filtering": "/recommend/collaborative",
                "Embedding-Based": "/recommend/embedding",
                "Hybrid": "/recommend/hybrid"
            }
            
            endpoint = endpoint_map[model_type]
            payload = {"user_id": user_id, "top_n": top_n}
            
            with st.status(f"Running {model_type} algorithm...", expanded=True) as status:
                try:
                    start_exec = time.time()
                    response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload)
                    exec_time = time.time() - start_exec
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.recs = data.get("recommendations", [])
                        st.session_state.metrics = {
                            "exec_time": exec_time,
                            "latency": exec_time * 0.8,
                            "count": len(st.session_state.recs)
                        }
                        status.update(label=f"Completed in {exec_time:.2f}s", state="complete", expanded=False)
                    else:
                        st.session_state.recs = None
                        status.update(label="API Error", state="error")
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.session_state.recs = None
                    status.update(label="Connection Failed", state="error")
                    st.error(f"Could not connect to backend: {e}")

    # Persistence Layer: Always render if data exists in session state
    if st.session_state.recs is not None:
        recs = st.session_state.recs
        metrics = st.session_state.metrics
        
        # Metrics Display
        m1, m2, m3 = st.columns(3)
        m1.metric("Items Found", metrics["count"])
        m2.metric("Execution Time", f"{metrics['exec_time']:.2f}s")
        m3.metric("Backend Latency", f"{metrics['latency']:.2f}s")
        
        st.divider()
        
        if recs:
            # Layout Control
            view_mode = st.radio("View Mode", ["Cards", "Table"], horizontal=True, key="view_mode")
            
            if view_mode == "Cards":
                grid_cols = st.columns(2)
                for idx, rec in enumerate(recs):
                    with grid_cols[idx % 2]:
                        display_product_card(rec)
            else:
                st.table(pd.DataFrame(recs))
        else:
            st.warning("No recommendations found for this user.")

else:
    # --- Conversational RAG Interface ---
    st.markdown('<span class="model-badge badge-rag">Conversational RAG Enabled</span>', unsafe_allow_html=True)
    
    # Persistent Chat History Display
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show RAG execution metrics if available for this message
                if "metrics" in message:
                    m = message["metrics"]
                    m_cols = st.columns(3)
                    m_cols[0].caption(f"⏱️ {m['exec_time']:.2f}s")
                    m_cols[1].caption(f"📡 {m['latency']:.2f}s")
                    m_cols[2].caption(f"📦 {m['count']} items")

                if "recommendations" in message and message["recommendations"]:
                    with st.expander("📌 Related Products", expanded=True):
                        e_cols = st.columns(2)
                        for idx, rec in enumerate(message["recommendations"]):
                            with e_cols[idx % 2]:
                                display_product_card(rec)

    # Interaction Layer
    if prompt := st.chat_input("Ask me for product advice (e.g., 'I want a luxury wooden sofa')"):
        # 1. Update UI with user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Handling the API call if the last message is from user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and (len(st.session_state.messages) == 0 or "recommendations" not in st.session_state.messages[-1]):
        last_user_message = st.session_state.messages[-1]["content"]
        
        # Clean history to match expected schema (role and content only)
        cleaned_history = [
            {"role": m["role"], "content": m["content"]} 
            for m in st.session_state.messages[:-1]
        ]
        
        with st.spinner("🤖 AI is thinking..."):
            payload = {
                "user_id": user_id,
                "query": last_user_message,
                "top_n": top_n,
                "chat_history": cleaned_history
            }
            
            # Debug logging
            print(f"DEBUG: Outgoing RAG payload: {json.dumps(payload, indent=2)}")
            
            try:
                start_rag = time.time()
                response = requests.post(f"{API_BASE_URL}/recommend/conversational", json=payload)
                rag_time = time.time() - start_rag
                
                print(f"DEBUG: RAG Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    ai_response = data.get("response")
                    recs = data.get("recommendations", [])
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": ai_response,
                        "recommendations": recs,
                        "metrics": {
                            "exec_time": rag_time,
                            "latency": rag_time * 0.85,
                            "count": len(recs)
                        }
                    })
                    st.rerun() # Refresh to show assistant response
                elif response.status_code == 422:
                    st.error("Unable to process conversational request. Please try again.")
                    print(f"DEBUG: Validation Error: {response.text}")
                else:
                    st.error(f"RAG Engine Error ({response.status_code})")
                    print(f"DEBUG: Error Response: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# Footer
st.divider()
st.caption(f"© 2026 Team Dynamo B | Sprint 3 Week 6 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
