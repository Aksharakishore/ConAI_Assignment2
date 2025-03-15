import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import streamlit as st
from transformers import pipeline

# Step 1: Data Preprocessing
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text, chunk_size=256):
    """Split text into smaller chunks."""
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 2: Basic RAG Implementation
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    """Embed text chunks using a pre-trained model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

def create_faiss_index(embeddings):
    """Create a FAISS index for vector storage and retrieval."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_chunks(query, index, embeddings, chunks, top_k=3):
    """Retrieve top-k relevant chunks using FAISS."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Step 3: Advanced RAG Implementation (Hybrid Search: BM25 + Dense Retrieval)
def hybrid_search(query, chunks, bm25, index, embeddings, top_k=3):
    """Combine BM25 and dense retrieval for hybrid search."""
    # BM25 retrieval
    bm25_scores = bm25.get_scores(query.split())
    bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]

    # Dense retrieval
    dense_results = retrieve_chunks(query, index, embeddings, chunks, top_k)

    # Combine results
    combined_results = list(set(bm25_indices.tolist() + [chunks.index(chunk) for chunk in dense_results]))
    return [chunks[i] for i in combined_results]

# Step 4: Streamlit UI
def build_ui():
    """Build a Streamlit UI for the RAG system."""
    st.title("Financial Question Answering System")
    query = st.text_input("Enter your financial question:")

    if query:
        # Input-side guardrail
        if not is_financial_query(query):
            st.write("Please ask a financial-related question.")
            return

        # Retrieve relevant chunks
        relevant_chunks = hybrid_search(query, chunks, bm25, index, embeddings)

        # Generate answer using a small language model
        answer = generate_answer(query, relevant_chunks)
        st.write(f"**Answer:** {answer}")

        # Display retrieved chunks
        st.write("**Relevant Chunks:**")
        for i, chunk in enumerate(relevant_chunks):
            st.write(f"{i + 1}. {chunk}")

# Step 5: Guardrail Implementation (Input-Side)
def is_financial_query(query):
    """Validate if the query is financial-related."""
    financial_keywords = ["revenue", "profit", "loss", "income", "cash flow", "balance sheet", "financial", "earnings"]
    return any(keyword in query.lower() for keyword in financial_keywords)

# Step 6: Response Generation
def generate_answer(query, relevant_chunks):
    """Generate an answer using a small language model."""
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
    context = " ".join(relevant_chunks)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    answer = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
    return answer.split("Answer:")[-1].strip()

# Main Execution
if __name__ == "__main__":
    # Load and preprocess financial data
    pdf_path = "financial_statement.pdf"  # Replace with your PDF file path
    text = extract_text_from_pdf(pdf_path)
    chunks = preprocess_text(text)

    # Embed chunks and create FAISS index
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)

    # Create BM25 index
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    # Build and run the Streamlit UI
    build_ui()
