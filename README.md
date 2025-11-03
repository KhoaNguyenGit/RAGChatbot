# RAG Medical QA Chatbot

This project is a local AI-powered question-answering system for medical documents. It leverages **Qwen2.5-3B-Instruct** for text generation and semantic search using embeddings to provide accurate responses to user queries based on a medical book or dataset.

---

## Project Structure

```bash
├── main.py # Main entry point to run the application
├── requirements.txt # Python dependencies
├── setup.sh # Setup script for environment and models
│
├── config/ # Configuration files for the system
│ └── configs.py
│
├── data/ # Input data files
│ └── Medical_book.pdf
│
├── model/ # Pretrained models used for embeddings, ranking, and LLM
│ ├── Qwen2.5-3B-Instruct
│ ├── all-MiniLM-L6-v2
│ └── mxbai-rerank-large-v1
│
├── src/ # Source code
│ ├── __init__.py
│ ├── helpers.py # Helper functions
│ │
│ ├── data_processing/pdf # PDF processing modules
│ │ ├── __init__.py
│ │ ├── pdf_loader.py
│ │ └── text_spliter.py
│ │
│ ├── llm/ # Local LLM interface
│ │ ├── __init__.py
│ │ └── pdf_loader.py
│ │
│ ├── rag_pipline/ # Retrieval-Augmented Generation pipeline
│ │ ├── __init__.py
│ │ └── rag_retrive.py
│ │
│ └── vector_db/ # Vector database integration (Qdrant)
│   ├── __init__.py
│   ├── qdrant_databse.py
│   └── qdrant_utils.py
│
├── static/ # Static files (CSS)
│ └── style.css
├── template/ # HTML templates for web interface
│ └── index.html
```

## Features

- **Document Processing**: Converts PDFs into searchable text chunks.
- **Vector Search**: Uses `all-MiniLM-L6-v2` embeddings and Qdrant for fast semantic search.
- **Reranking**: Leverages `mxbai-rerank-large-v1` to rank retrieved results for more accurate answers.
- **Local LLM Generation**: Uses `Qwen2.5-3B-Instruct` for generating responses from retrieved knowledge.
- **Web Interface**: Simple web UI for asking questions and viewing answers.

---

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <project_folder>
    ```

2. Create venv python:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

3. Create the ```model/``` folder and download models:
    ```bash
    mkdir -p model
    cd model

    git clone <Qwen2.5-3B-Instruct_repo_url> Qwen2.5-3B-Instruct
    git clone <all-MiniLM-L6-v2_repo_url> all-MiniLM-L6-v2
    git clone <mxbai-rerank-large-v1_repo_url> mxbai-rerank-large-v1

    cd ..
    ```

---

## Usage

Run the main application:

```bash
python main.py
