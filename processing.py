from src.vector_db.qdrant_database import QdrantDBClient


def execute_pipeline(
    query: str,
    host: str,
    api_key: str,
    collection_name: str,
    top_k_retrive: int,
    top_k_rerank: int,
    embedding,
    llm,
    rerank_model,
):
    
    query_vector = embedding.embed_query(query)

    qdrant = QdrantDBClient(host=host, api_key=api_key, collection_name=collection_name)
    retrive_docs = qdrant.search_vectors(query_vector=query_vector, top_k=top_k_retrive)
    documents = [hit.get("text", "") for hit in retrive_docs]
    
    if len(documents)<1:
        context_str = ""
        
    else:
        results = []
        results = rerank_model.rank(query, documents, return_documents=True, top_k=top_k_rerank)
        context = [doc.get('text', '') for doc in results]
        context_str = "\n\n".join(context)


    messages = [
        {
            "role": "system", 
            "content": (
            "You are a helpful AI assistant. "
            "If the user's question is general or conversational (e.g., 'who are you?', greetings, or chit-chat), "
            "respond naturally and ignore the provided context. "
            "If the question is medical or health-related, act as a Medical assistant and use the retrieved context "
            "to answer concisely (maximum three sentences). "
            "If you don't know the answer, say that you don't know."
        )},
        {
            "role": "user", 
            "content": f"Context:\n{context_str}\n\nQuestion:\n{query}\n\n"
        }
    ]
    
    print(messages)  # Debug: Print the constructed prompt

    response = llm.generate(messages)

    return response