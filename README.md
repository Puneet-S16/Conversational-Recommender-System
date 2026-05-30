# Conversational Recommender System

A professional, end-to-end recommendation engine combining traditional Collaborative Filtering, Embedding-based Retrieval, and Generative AI (RAG) for personalized product discovery.

## Project Overview

This project implements a multi-stage recommendation pipeline designed for modern e-commerce. It features:
- Collaborative Filtering: User-item interaction modeling.
- Semantic Retrieval: Product embeddings for content-based matching.
- Hybrid Ranking: Multi-signal ranking using XGBoost (Learning to Rank).
- Conversational RAG: A Generative AI interface that allows users to ask for recommendations in natural language.

## Features

- FastAPI Backend: High-performance API endpoints for different recommendation engines.
- Streamlit Dashboard: An interactive UI for searching products and chatting with the AI recommender.
- RAG Implementation: Uses LangChain and Ollama (or OpenAI) to generate context-aware product suggestions.
- Vector Database: FAISS-powered semantic search for sub-second retrieval.

## Project Structure

```text
+- api/                      # FastAPI Backend
|  +- routes/               # API Endpoint definitions
|  +- main.py               # API Entry point
|  +- services.py           # Business logic and model singletons
|  +- recommender_utils.py  # Shared utilities and data loaders
+- dashboard/                # Streamlit UI
|  +- app.py                # Dashboard entry point
+- rag/                      # Generative AI Logic
|  +- prompts/              # System prompt templates
|  +- rag_chain.py          # RAG pipeline implementation
|  +- retriever.py          # FAISS retriever bridge
+- models/                   # Model weights and metadata (Included for out-of-the-box use)
+- data/                     # Data samples (Included for out-of-the-box use)
+- docs/                     # Technical documentation
+- requirements.txt          # Project dependencies
```

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Puneet-S16/Conversational-Recommender-System.git
   cd Conversational-Recommender-System
   ```

Note: Sample datasets and pretrained model artifacts (~2MB) are included in the repository, allowing the system to be fully functional immediately after installation.

2. Set up a Virtual Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Environment Variables:
   Create a .env file in the root directory:
   ```env
   OPENAI_API_KEY=your_key_here  # Optional: defaults to Ollama if not provided
   ```

## Running the Application

1. Start the FastAPI Backend:
   ```bash
   python -m api.main
   ```

2. Launch the Dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

---
*Developed for my professional portfolio. This project demonstrates skills in Python, Machine Learning (LTR), NLP (RAG), and Full-Stack development for Data Science.*
