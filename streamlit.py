import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Embedding function
def get_embedding(text: str, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

# Chunking function
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)

# Upload and process document
st.title("📄 Upload Document to Pinecone")

uploaded_file = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])
if uploaded_file:
    # Extract text
    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    else:
        text = uploaded_file.read().decode("utf-8")

    chunks = chunk_text(text)

    with st.spinner("Embedding and uploading..."):
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk)
                index.upsert(
                    vectors=[
                        {
                            "id": f"{uploaded_file.name}_{i}",
                            "values": embedding,
                            "metadata": {"text": chunk, "source": uploaded_file.name}
                        }
                    ]
                )
            except Exception as e:
                st.error(f"Error with chunk {i}: {str(e)}")
                continue

    st.success(f"{len(chunks)} chunks uploaded to Pinecone!")

# Chat section
st.header("💬 Chat with your documents")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_message = st.chat_input("Ask something about your documents...")
if user_message:
    # Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": user_message})

    # Retrieve context
    def retrieve_context(query: str, top_k=5):
        query_embedding = get_embedding(query)
        result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        contexts = [m["metadata"]["text"] for m in result.matches if "metadata" in m and "text" in m["metadata"]]
        return "\n\n".join(contexts)

    context = retrieve_context(user_message)

    # Build system prompt
    system_prompt = f"""
You are a tutor with access to a knowledge base.

[Knowledge Base]
{context}
"""

    # Generate reply
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}] + st.session_state.chat_history
    )

    reply = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# Display chat history
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])
