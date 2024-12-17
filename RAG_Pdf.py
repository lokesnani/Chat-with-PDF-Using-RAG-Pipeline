import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from langchain.llms import OpenAI
import streamlit as st
import pandas as pd
import re

# Step 1: Extract text from specific pages of a PDF
def extract_text_from_specific_pages(pdf_path, page_numbers):
    reader = PdfReader(pdf_path)
    extracted_text = {}
    for page_number in page_numbers:
        if 0 <= page_number < len(reader.pages):
            text = reader.pages[page_number].extract_text()
            extracted_text[page_number + 1] = text
        else:
            extracted_text[page_number + 1] = f"Page {page_number + 1} out of range."
    return extracted_text


# Step 2: Parse text for specific information
def extract_unemployment_data(page_text):
    # Extract unemployment data based on degree types
    pattern = r"(\w+\sdegree):\s([\d.]+)%"
    matches = re.findall(pattern, page_text)
    return {degree_type: float(unemployment) for degree_type, unemployment in matches}


def extract_tabular_data(page_text):
    # Extract tabular data
    lines = page_text.split("\n")
    table_data = []
    for line in lines:
        if re.search(r"\d+", line):  # Example heuristic: rows with numbers
            table_data.append(line.split())
    return pd.DataFrame(table_data)


# Step 3: Chunk text into smaller sections
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# Step 4: Embed text chunks and store in FAISS
def embed_chunks(chunks, model):
    return model.encode(chunks)


def store_embeddings_in_faiss(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# Step 5: Retrieve relevant chunks
def retrieve_relevant_chunks(query_embedding, index, chunks, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# Step 6: Generate response
def generate_response_from_chunks(chunks, query, llm):
    context = "\n\n".join(chunks)
    prompt = f"Context from the PDF:\n{context}\n\nAnswer the query:\n{query}"
    return llm(prompt)


# Step 7: Process comparison queries
def process_comparison_query(query, pdf_texts):
    fields = re.findall(r"\b(degree|unemployment|data|comparison)\b", query, re.IGNORECASE)
    comparison_data = {}
    for pdf_name, text in pdf_texts.items():
        if "unemployment" in fields:
            extracted_data = extract_unemployment_data(text)
        else:
            extracted_data = extract_tabular_data(text)
        comparison_data[pdf_name] = extracted_data
    return comparison_data


# RAG Pipeline
def rag_pipeline(pdf_path, query, model, llm, specific_pages=None, top_k=5):
    extracted_text = extract_text_from_specific_pages(pdf_path, specific_pages or [])
    text = " ".join(extracted_text.values())
    chunks = chunk_text(text)

    # Embed chunks and store in FAISS
    embeddings = embed_chunks(chunks, model)
    index = store_embeddings_in_faiss(embeddings)

    # Query processing
    query_embedding = model.encode([query])
    retrieved_chunks = retrieve_relevant_chunks(query_embedding, index, chunks, top_k)

    # Generate response
    return generate_response_from_chunks(retrieved_chunks, query, llm)


# Streamlit App
st.title("Chat with PDFs: RAG Pipeline")
st.subheader("Upload your PDF and ask questions interactively!")

# Cache models to avoid reloading
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = OpenAI(temperature=0, api_key="your-openai-api-key")
    return embedding_model, llm


embedding_model, llm = load_models()

# File uploader
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Enter your question:")
specific_pages_input = st.text_input("Specify page numbers (comma-separated, optional):")

if uploaded_pdf:
    # Save the uploaded PDF
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    pdf_path = "uploaded_file.pdf"

    # Handle user query
    if query:
        try:
            # Parse specific pages if provided
            if specific_pages_input:
                specific_pages = [int(page.strip()) - 1 for page in specific_pages_input.split(",") if page.strip().isdigit()]
            else:
                specific_pages = None

            # Process the query
            st.info("Processing your request, please wait...")
            response = rag_pipeline(pdf_path, query, embedding_model, llm, specific_pages=specific_pages)
            st.subheader("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question to interact with the PDF.")
else:
    st.warning("Please upload a PDF to get started!")
