import streamlit as st
import openai
import os
import tempfile
import tiktoken
from pinecone import Pinecone
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Helper: Split text into chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)

# Helper: Get embeddings from OpenAI
# Generate embeddings
def get_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

# Upload PDF and process
st.title("📄 Document Uploader to Pinecone")

uploaded_file = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.success("File successfully uploaded. Now chunking and embedding...")

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk)
            index.upsert([
                {
                    "id": f"{uploaded_file.name}_{i}",
                    "values": emb,
                    "metadata": {"text": chunk, "source": uploaded_file.name}
                }
            ])
        except Exception as e:
            st.error(f"Error uploading chunk {i}: {str(e)}")

    st.success(f"{len(chunks)} chunks uploaded to Pinecone!")

# Chat interface
st.header("💬 Ask a question")

query = st.text_input("Enter your question:")
if query:
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # Get top 5 chunks of text
    context = "\n\n".join([match['metadata']['text'] for match in results['matches']])

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[

        ]
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )


    st.markdown("### Answer:")
    st.write(response.choices[0].message.content)
