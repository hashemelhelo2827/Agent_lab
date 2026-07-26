import chromadb
import os 
import requests
from dotenv import load_dotenv

dotenv_path=os.path.join(os.path.dirname(__file__),'../',r'04_rag_vector_db/.env')
load_dotenv(dotenv_path=dotenv_path)
API_KEY=os.getenv('GEMINI_API_KEY')

def embed_batch(texts):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={API_KEY}"
    payload = [
            {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]}}
            for text in texts
        ]
    resp =requests.post(url, json={"requests": payload})
    data=resp.json()
    return[e['values'] for e in data['embeddings']]


base = os.path.dirname(__file__)
rules=[]
with open(os.path.join(base, "company_policies.txt"), encoding="utf-8") as f:
    rules = f.read().split("###")


titles = []
bodies = []
for chunk in rules:
    lines = chunk.strip().split("\n", 1)
    if len(lines) < 2:
        continue
    titles.append(lines[0].strip())
    bodies.append(lines[1].strip())


client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="policies",
    metadata={"hnsw:space": "cosine"}
    )


if collection.count() == 0:
    embeddings = embed_batch(bodies)
    collection.add(
         ids=[f"policy_{i}" for i in range(len(titles))],
        embeddings=embeddings,
        metadatas=[{"title": t} for t in titles],
        documents=bodies
    )
else: print(f"Loaded {collection.count()} policies")

question = input("Ask a question: ")
query_embedding = embed_batch([question])[0]
results = collection.query(query_embeddings=[query_embedding], n_results=3)

THRESHOLD = 0.5
matches = []
for i in range(len(results["ids"][0])):
    dist = results["distances"][0][i]
    if dist > THRESHOLD:
        continue
    matches.append((dist, results["metadatas"][0][i], results["documents"][0][i]))

if not matches:
    print("No matching policies found.")
else:
    print(f"\nMatches for: {question}\n")
    for rank, (dist, meta, doc) in enumerate(matches, 1):
        print(f" {rank}. {meta['title']} (distance: {dist:.3f})")
        print(f"    {doc[:100]}{'...' if len(doc) > 100 else ''}")
        print()