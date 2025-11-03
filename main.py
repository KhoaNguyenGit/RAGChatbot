import os
import torch

from uuid import uuid4
from src.data_processing.pdf.pdf_loader import PDFLoader
from src.data_processing.pdf.text_splitter import PDFTextSplitter
from src.rag_pipline.rag_retrive import retrieve_and_respond
from src.vector_db.qdrant_database import QdrantDBClient
from src.helpers import hugging_face_embeddings
from src.vector_db.qdrant_utils import checking_connection
from config.configs import DATA_PATH, EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL
from typing import List
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv


load_dotenv()

def process_pdf(data_path: str):
    loader = PDFLoader(data_path)
    documents = loader.load_pdf_file()

    splitter = PDFTextSplitter(chunk_size=500, chunk_overlap=20)
    filtered_data = splitter.filter_to_minimal_docs(documents)
    chunks = splitter.split(filtered_data)

    return chunks

def save_to_database(chunks: List[Document], embedding_model: str, host: str, api_key: str, collection_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding = hugging_face_embeddings(embedding_model, device)
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedding.embed_documents(texts)
    
    qdrantdb = QdrantDBClient(host=host, api_key=api_key, collection_name=collection_name)
    qdrantdb.ensure_collection(vector_size=384)

    print(f"Finished embedding {len(vectors)} documents")

    points = [
        {
            "id": chunk.metadata.get("id"),
            "vector": vector,
            "payload": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source")
            }
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    
    BATCH_SIZE = 100
    batches = [points[i:i + BATCH_SIZE] for i in range(0, len(points), BATCH_SIZE)]
    
    print(f"Starting upload of {len(points)} vectors in {len(batches)} batches...")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(qdrantdb.upsert_vectors, batch): idx for idx, batch in enumerate(batches)}

        for future in futures:
            idx = futures[future]
            try:
                future.result(timeout=120)
            except Exception as e:
                print(f"Batch {idx + 1} failed: {e}")
            
def delete_collection(host: str, api_key: str, collection_name: str):
    qdrantdb = QdrantDBClient(host=host, api_key=api_key, collection_name=collection_name)
    qdrantdb.delete_collection()

def retrieve_vector( embedding_model: str, host: str, api_key: str, collection_name: str):
    qdrantdb = QdrantDBClient(host=host, api_key=api_key, collection_name=collection_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    embedding = hugging_face_embeddings(embedding_model, device)
    query = "What acne is?"
    query_vector = embedding.embed_query(query)
    results = qdrantdb.search_vectors(query_vector=query_vector, top_k=8)
    return results

def chat_llm():
    system_prompt = system_prompt

def main():
    data_path = DATA_PATH
    host = os.getenv("QDRANT_END_POINT")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = "biomedical_docs"
        
    # chunks = process_pdf(data_path)
    
    # Qdrant saving data
    # save_to_database(chunks, EMBEDDING_MODEL, host, qdrant_api_key, collection_name)

    # Qdrant deleting collection
    # delete_collection(host, api_key=qdrant_api_key, collection_name=collection_name)
    
    # Qdrant retrieve vector
    results = retrieve_vector(EMBEDDING_MODEL, host, qdrant_api_key, collection_name)
    print("\n=== Retrieved Vectors ===\n")
    for result in results:
        print(f"{result}\n\n")

    # query = "what do you know about acne?"
    # response = retrieve_and_respond(
    #     query=query,
    #     embedding_model=EMBEDDING_MODEL,
    #     reranker_model=RERANKER_MODEL,
    #     llm_model=LLM_MODEL,
    #     host=host,
    #     api_key=qdrant_api_key,
    #     collection_name=collection_name,
    #     top_k_retrive=10,
    #     top_k_rerank=3,
    # )
    # print("\n=== Final Response ===\n")
    # print(response)

if __name__ == "__main__":
    main()