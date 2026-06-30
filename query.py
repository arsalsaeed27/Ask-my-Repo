from sentence_transformers import SentenceTransformer
import chromadb

print("Loading model and connecting to database...")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("code_chunks")

def retrieve(question, n_results =3):
    question_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings = question_embedding,
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    # q1: testin javascript logic
    question = "How does the app fetch data from NASA's API?"
    print(f"\nSearching for: '{question}'\n")
    
    results = retrieve(question)
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    for doc, meta in zip(documents, metadatas):
        print(f"--- {meta['file']} (lines {meta['start_line']}-{meta['end_line']}) ---")
    
        print(doc[:200] + "...\n")