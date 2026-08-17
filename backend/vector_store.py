import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="resume_collection")


def store_embeddings(chunks, embeddings):
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
    )


def search_embeddings(query_embedding, top_k=3):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )