import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to the path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.utils.llm_handler import LLMHandler
from src.utils.document_handler import DocumentHandler
from src.utils.rag_system import RAGSystem

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="LangChain RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 LangChain RAG System")
st.write("Retrieval-Augmented Generation powered by LangChain and Gemini")

# Initialize session state
if 'rag_system' not in st.session_state:
    try:
        llm = LLMHandler()
        st.session_state.rag_system = RAGSystem(llm)
        st.session_state.api_ready = True
    except ValueError as e:
        st.session_state.api_ready = False
        st.error(f"⚠️ {e}")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

# Sidebar
with st.sidebar:
    st.header("📚 Configuration")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        st.success("✅ API Key configured")
    else:
        st.error("❌ API Key not found in .env")
    
    if st.session_state.api_ready:
        st.success("✅ LLM Ready")
    else:
        st.error("❌ LLM Not Available")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )
    
    if uploaded_files:
        if st.button("📥 Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Save files temporarily
                    temp_dir = Path("./temp_uploads")
                    temp_dir.mkdir(exist_ok=True)
                    
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = temp_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(str(file_path))
                    
                    # Ingest documents
                    if st.session_state.api_ready:
                        result = st.session_state.rag_system.ingest_documents(file_paths)
                        st.success(f"✅ Processed {result['documents_processed']} document(s) into {result['chunks_created']} chunks")
                        st.session_state.uploaded_documents.extend(file_paths)
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

with col2:
    st.subheader("📊 Status")
    st.metric("Documents Uploaded", len(st.session_state.uploaded_documents))
    st.metric("Chunks Created", len(st.session_state.rag_system.processed_chunks) if st.session_state.api_ready else 0)

st.divider()

# Question Answering Section
st.subheader("❓ Ask Questions")

if st.session_state.uploaded_documents:
    query = st.text_input("Enter your question about the documents:")
    
    if query:
        if st.button("🔍 Search"):
            with st.spinner("Searching and generating answer..."):
                try:
                    if st.session_state.api_ready:
                        answer = st.session_state.rag_system.answer_question(query)
                        st.success("Answer:")
                        st.write(answer)
                    else:
                        st.error("LLM is not available")
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("📥 Please upload documents first to ask questions")

st.divider()

# Document Summary Section
if st.session_state.uploaded_documents:
    st.subheader("📋 Document Summary")
    
    if st.button("📊 Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                if st.session_state.api_ready:
                    summary = st.session_state.rag_system.summarize_documents()
                    st.success("Summary:")
                    st.write(summary)
                else:
                    st.error("LLM is not available")
            except Exception as e:
                st.error(f"Error: {e}")
