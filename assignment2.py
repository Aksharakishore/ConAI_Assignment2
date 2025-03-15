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
import asyncio

# Fix asyncio event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
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
#memory = ConversationBufferMemory(memory_key="chat_history")
memory = ConversationBufferMemory(return_messages=True)

# Load small open-source language model (SLM)
MODEL_NAME = "t5-small"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use float32 to avoid precision issues
        device_map="cpu",  # Ensure it runs on CPU
        low_cpu_mem_usage=True
    ).to("cpu")

    st.write("✅ Model loaded successfully!")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    model = None  # Prevents crashes in generate_response()


# Function to generate responses
def generate_response(prompt):
    if model is None:
        return "⚠️ Error: Model not loaded. Please check logs."
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Function to retrieve financial trend insights instead of raw data
def retrieve_financial_info(query):
    # Fetch last 6 months of stock data
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo", interval="1mo")

    if df.empty:
        return "No financial data available."

    # Calculate trend (percent change from start to end)
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    trend = ((end_price - start_price) / start_price) * 100

    trend_text = (
        f"Apple's stock price over the last 6 months has "
        f"{'increased' if trend > 0 else 'decreased'} by {abs(trend):.2f}%.\n"
        f"Recent Prices: {df['Close'].tolist()}"
    )
    return trend_text


# Function to filter hallucinations and irrelevant responses
def filter_output(response):
    # Allow response if it contains financial terms or numbers
    keywords = ["price", "stock", "trend", "increase", "decrease", "growth", "decline", "market"]
    if any(word in response.lower() for word in keywords) or any(char.isdigit() for char in response):
        return response
    return "[Filtered Output]: The response does not appear to be financial-related."


# Streamlit UI
def main():
    st.title("📊 Financial RAG Chatbot")
    user_query = st.text_input("Ask a financial question:")
    
    if st.button("Submit"):
        financial_context = retrieve_financial_info(user_query)
        memory.save_context({"input": user_query}, {"output": financial_context})
        response = generate_response(financial_context + "\n Answer this: " + user_query)
        #filtered_response = filter_output(response)
        
        st.subheader("📌 Answer:")
        st.write(response)
        st.write("### 🔍 Confidence Score:", 0.9)  # Placeholder score
        
        st.subheader("📝 Chat History:")
        st.write(memory.load_memory_variables({}))

if __name__ == "__main__":
    main()
