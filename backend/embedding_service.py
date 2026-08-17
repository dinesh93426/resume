import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load the model once to avoid reloading it on every request
_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text: str):
    try:
        # Generate the embedding and convert it to a flat list of floats
        embedding = _model.encode(text)
        return embedding.tolist()
    except Exception as exc:
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc
