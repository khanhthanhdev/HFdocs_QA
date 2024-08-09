import streamlit as st
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

# Load environment variables and configure settings
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini and global settings
@st.cache_resource
def configure_llm():
    llm = Gemini(model="models/gemini-1.5-pro", api_key=GOOGLE_API_KEY, temperature=0.5)
    Settings.llm = llm
    Settings.system_prompt = "You are an expert on the HuggingFace platform and your job is to answer technical questions. Assume that all questions are related to the HuggingFace platform. Keep your answers technical and based on facts – do not hallucinate features."
    return llm

llm = configure_llm()

# Configure local embedding model
@st.cache_resource
def configure_embedding_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load and index data
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the HuggingFace docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embed_model = configure_embedding_model()
        return VectorStoreIndex.from_documents(docs, embed_model=embed_model)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about HuggingFace!"}
    ]
if "chat_engine" not in st.session_state:
    index = load_data()
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Streamlit UI
st.header("Chat with the HuggingFace docs 💬 📚")

# Handle user input
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            st.write("Sources:")
            for source in message["sources"]:
                st.write(f"- {source}")

# Generate response if last message is from user
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            
            # Display source documents
            if response.source_nodes:
                st.write("Sources:")
                for node in response.source_nodes:
                    doc_path = node.node.metadata.get('file_path', 'Unknown')
                    st.write(f"- {doc_path}")
            
            # Add response to message history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response.response,
                "sources": [node.node.metadata.get('file_path', 'Unknown') for node in response.source_nodes]
            })