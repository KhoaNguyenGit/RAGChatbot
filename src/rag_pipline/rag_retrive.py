from langchain.schema import Document
from typing import List

from sentence_transformers import CrossEncoder
from src.vector_db.qdrant_database import QdrantDBClient
from src.helpers import hugging_face_embeddings
from src.helpers import rerank_loader
from src.llm.local_llm import LocalLLM
from typing import Dict, Any

def retrieve_and_respond(
    query: str,
    embedding_model: str,
    reranker_model: str,
    llm_model: str,
    host: str,
    api_key: str,
    collection_name: str,
    top_k_retrive: int,
    top_k_rerank: int
):

    embedding = hugging_face_embeddings(embedding_model, device="cpu")
    llm = LocalLLM(model_name=llm_model)
    rerank_model = rerank_loader(reranker_model, device='cpu')
    
    query_vector = embedding.embed_query(query)

    qdrant = QdrantDBClient(host=host, api_key=api_key, collection_name=collection_name)
    retrive_docs = qdrant.search_vectors(query_vector=query_vector, top_k=top_k_retrive)
    documents = [hit.payload.get("text", "") for hit in retrive_docs]
    
    results = []
    results = rerank_model.rank(query, documents, return_documents=True, top_k=top_k_rerank)

    context = [doc.get('text', '') for doc in results]
    context_str = "\n\n".join(context)

    messages = [
        {
            "role": "system", 
            "content": (
            "You are a Medical assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
        )},
        {
            "role": "user", 
            "content": f"Context:\n{context_str}\n\nQuestion:\n{query}\n\n"
        }
    ]
    
    print(messages)  # Debug: Print the constructed prompt

    response = llm.generate(messages)

    return response