# Langchain RAG

A Retrieval-Augmented Generation (RAG) system built with LangChain and Gemini AI.

## Features

- 🔍 Document ingestion and processing
- 🤖 LLM-powered question answering
- 📚 Vector embeddings and semantic search
- 🔗 LangChain integration
- ⚡ Fast retrieval and response generation

## Installation

```bash
# Clone the repository
git clone https://github.com/Divinekrishna/langchain-rag.git
cd langchain-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

2. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
langchain-rag/
├── src/
│   ├── utils/
│   │   ├── llm_handler.py
│   │   ├── document_handler.py
│   │   └── rag_system.py
│   └── __init__.py
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

## Usage

[Coming soon...]

## License

MIT
