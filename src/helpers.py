from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

def hugging_face_embeddings(embedding_model: str, device: str):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": device})
    print("load embedding model")
    return embeddings

def rerank_loader(rerank_model: str, device: str):
    rerank_model = CrossEncoder(rerank_model, device=device)
    print("load rerank model")
    return rerank_model