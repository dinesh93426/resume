from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from chunking import chunk_text
from embedding_service import generate_embedding
from rag_service import generate_answer
from vector_store import search_embeddings, store_embeddings


app = FastAPI(title="AI Resume Assistant API")


class SearchRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "AI Resume Assistant API is running"}


@app.post("/upload")
def upload_resume(file: UploadFile = File(...)):
    try:
        
        from pypdf import PdfReader  
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="PDF dependency missing. Install pypdf or PyPDF2.",
            ) from exc

    reader = PdfReader(file.file)
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)

    text = "\n".join(text_parts).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    chunks = chunk_text(text)
    embeddings = [generate_embedding(chunk) for chunk in chunks]
    store_embeddings(chunks, embeddings)

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
    }


@app.post("/search")
def search_resume(payload: SearchRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    query_embedding = generate_embedding(question)
    results = search_embeddings(query_embedding)
    documents = results.get("documents", [])

    relevant_chunks = documents[0] if documents else []
    if not relevant_chunks:
        return {
            "question": question,
            "answer": "No resume context found yet. Upload a resume first.",
            "relevant_chunks": [],
        }

    context = "\n\n".join(relevant_chunks)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "relevant_chunks": relevant_chunks,
    }
