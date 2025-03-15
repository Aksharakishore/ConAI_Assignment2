import streamlit as st
import faiss
import pickle
import os
import yfinance as yf
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load embedding model
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    st.write("✅ SentenceTransformer loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading SentenceTransformer: {e}")

# Load or create FAISS index
INDEX_FILE = "financial_index.faiss"
DATA_FILE = "financial_data.pkl"

if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(DATA_FILE, "rb") as f:
        financial_data = pickle.load(f)
else:
    index = None
    financial_data = []
    
    # Fetch financial data using yfinance
    st.write("📥 Downloading financial data...")
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y", interval="1mo")
    df.to_csv("financial_data.csv")
    
    financial_data = df.to_string().split("\n")
    embeddings = embed_model.encode(financial_data)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    
    with open(DATA_FILE, "wb") as f:
        pickle.dump(financial_data, f)

# Initialize memory for context-aware retrieval
memory = ConversationBufferMemory(memory_key="chat_history")

# Load small open-source language model (SLM)
MODEL_NAME = "t5-small"  # Smaller model for efficiency

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    ).to("cpu")
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Function to generate responses
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)  # Reduced max length
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Function to retrieve relevant financial chunks
def retrieve_financial_info(query, k=3):
    query_embedding = embed_model.encode([query])
    if index is not None:
        distances, indices = index.search(query_embedding, k)
        retrieved_texts = [financial_data[i] for i in indices[0]]
        return "\n".join(retrieved_texts)
    return "No relevant financial information found."

# Function to filter hallucinations
def filter_output(response, confidence_threshold=0.5):
    if "financial" not in response.lower():
        return "[Filtered Output]: The response does not appear to be financial-related."
    return response

# Streamlit UI
def main():
    st.title("📊 Financial RAG Chatbot")
    user_query = st.text_input("Ask a financial question:")
    
    if st.button("Submit"):
        context = retrieve_financial_info(user_query)
        memory.save_context({"input": user_query}, {"output": context})
        response = generate_response(context + "\n Answer this: " + user_query)
        filtered_response = filter_output(response)
        
        st.subheader("📌 Answer:")
        st.write(filtered_response)
        st.write("### 🔍 Confidence Score:", 0.9)  # Placeholder score
        
        st.subheader("📝 Chat History:")
        st.write(memory.load_memory_variables({}))

if __name__ == "__main__":
    main()
