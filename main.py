import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# -----------------------------
# Step 1: Load Embedding Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Step 2: Initialize ChromaDB
# -----------------------------
chroma_client = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

collection = chroma_client.get_or_create_collection(name="semantic_search")

# -----------------------------
# Step 3: Load Documents
# -----------------------------
with open("data/sample_docs.txt", "r") as file:
    documents = file.readlines()

documents = [doc.strip() for doc in documents if doc.strip()]

# -----------------------------
# Step 4: Generate Embeddings
# -----------------------------
embeddings = model.encode(documents)

# -----------------------------
# Step 5: Store in Vector DB
# -----------------------------
for i, doc in enumerate(documents):
    collection.add(
        documents=[doc],
        embeddings=[embeddings[i].tolist()],
        ids=[str(i)]
    )

print("Documents stored successfully!\n")

# -----------------------------
# Step 6: Query Function
# -----------------------------
def semantic_search(query, top_k=2):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]

# -----------------------------
# Step 7: Test Search
# -----------------------------
if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    results = semantic_search(user_query)

    print("\nTop Results:")
    for result in results:
        print("-", result)
