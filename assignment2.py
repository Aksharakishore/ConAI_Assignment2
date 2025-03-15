import os
import pickle
import numpy as np
import faiss
import yfinance as yf
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Constants
INDEX_FILE = "financial_index.faiss"
DATA_FILE = "financial_data.pkl"
EMBEDDINGS_FILE = "financial_embeddings.pkl"

# Load or create FAISS index, financial data, and embeddings
if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE) and os.path.exists(EMBEDDINGS_FILE):
    # Load existing index, data, and embeddings
    index = faiss.read_index(INDEX_FILE)
    with open(DATA_FILE, "rb") as f:
        financial_data = pickle.load(f)
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
else:
    # Fetch financial data using yfinance
    st.write("📥 Downloading financial data...")
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y", interval="1mo")
    df.to_csv("financial_data.csv")

    # Preprocess financial data
    financial_data = df.to_string().split("\n")

    # Embed financial data
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(financial_data)

    # Create and save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

    # Save financial data and embeddings
    with open(DATA_FILE, "wb") as f:
        pickle.dump(financial_data, f)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

# Step 2: Basic RAG Implementation
def retrieve_chunks(query, index, embeddings, financial_data, top_k=3):
    """Retrieve top-k relevant chunks using FAISS."""
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [financial_data[i] for i in indices[0]]

# Step 3: Streamlit UI
def build_ui():
    """Build a Streamlit UI for the RAG system."""
    st.title("Financial Question Answering System")
    query = st.text_input("Enter your financial question:")

    if query:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_chunks(query, index, embeddings, financial_data)

        # Generate answer using a small language model
        try:
            answer = generate_answer(query, relevant_chunks)
            st.write(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {e}")

        # Display retrieved chunks
        st.write("**Relevant Financial Data:**")
        for i, chunk in enumerate(relevant_chunks):
            st.write(f"{i + 1}. {chunk}")

# Step 4: Response Generation
def generate_answer(query, relevant_chunks):
    """Generate an answer using a small language model."""
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
    context = " ".join(relevant_chunks)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    answer = generator(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
    return answer.split("Answer:")[-1].strip()

# Main Execution
if __name__ == "__main__":
    build_ui()
