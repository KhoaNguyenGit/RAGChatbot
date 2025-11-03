import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from src.helpers import hugging_face_embeddings, rerank_loader
from src.llm.local_llm import LocalLLM
from processing import execute_pipeline
from config.configs import EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL

load_dotenv()

app = Flask(__name__)
CORS(app)

embedding = hugging_face_embeddings(EMBEDDING_MODEL, device="cpu")
llm = LocalLLM(model_name=LLM_MODEL)
rerank_model = rerank_loader(RERANKER_MODEL, device='cpu')

qdrant_host = os.getenv("QDRANT_END_POINT")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = "biomedical_docs"

@app.route("/api/v1/query", methods=["POST"])
def execute():
    data = request.get_json()
    query = data.get("query", "")

    response = execute_pipeline(
        query=query,
        host=qdrant_host,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        top_k_retrive=10,
        top_k_rerank=3,
        embedding=embedding,
        llm=llm,
        rerank_model=rerank_model
    )
    print("\n=== Final Response ===\n")
    print(response)
    
    return jsonify({"status": "done", "response": response})
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8190, debug=False)