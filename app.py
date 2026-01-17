import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to the path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="LangChain RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 LangChain RAG System")
st.write("Retrieval-Augmented Generation powered by LangChain and Gemini")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'documents' not in st.session_state:
    st.session_state.documents = []

# Sidebar
with st.sidebar:
    st.header("📚 Configuration")
    st.write("Configure your RAG system here")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        st.success("✅ API Key configured")
    else:
        st.error("❌ API Key not found in .env")

# Main content
st.subheader("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF or text files",
    accept_multiple_files=True,
    type=['pdf', 'txt', 'docx']
)

if uploaded_files:
    st.info(f"Uploaded {len(uploaded_files)} file(s)")

st.subheader("Ask Questions")
query = st.text_input("Enter your question about the documents:")

if query:
    st.info("Processing your query...")
    st.write("Response will appear here")
