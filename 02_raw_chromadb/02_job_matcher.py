import csv
import os
import requests
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
import json

dotenv_path = os.path.join(os.path.dirname(__file__), "..", "04_rag_vector_db", ".env")
load_dotenv(dotenv_path)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def embed_batch(texts):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={API_KEY}"
    payload = [
        {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]}}
        for text in texts
    ]
    resp = requests.post(url, json={"requests": payload})
    data = resp.json()
    if "embeddings" not in data:
        print("API Error:", json.dumps(data, indent=2))
        return []
    return [e["values"] for e in data["embeddings"]]

client_db = chromadb.Client()
collection = client_db.get_or_create_collection(
    name="job_catalog",
    metadata={"hnsw:space": "cosine"}
)

base = os.path.dirname(__file__)
jobs = []
with open(os.path.join(base, "jobs.txt"), newline="", encoding="utf-8") as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    for row in reader:
        jobs.append({
            "id": row[0], "title": row[1], "company": row[2],
            "description": row[3], "remote": row[4] == "True",
            "salary_min": int(row[5]), "location": row[6]
        })

descriptions = [j["description"] for j in jobs]
embeddings = embed_batch(descriptions)

collection.add(
    ids=[j["id"] for j in jobs],
    embeddings=embeddings,
    metadatas=[{
        "title": j["title"], "company": j["company"],
        "remote": j["remote"], "salary_min": j["salary_min"],
        "location": j["location"]
    } for j in jobs],
    documents=descriptions
)

user_input = input("Enter your job preferences: ")

prompt = (
    f"Extract job preferences from: {user_input}\n"
    "Return JSON with:\n"
    "- 'skills': list of skills\n"
    "- 'remote': bool (true if they want remote work)\n"
    "- 'salary_min': number (minimum salary they want)"
)
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    response_format={'type': 'json_object'},
    messages=[{"role": "user", "content": prompt}]
)
prefs = json.loads(response.choices[0].message.content)

query_text = ", ".join(prefs["skills"])
query_embedding = embed_batch([query_text])[0]

where_clause = {"$and": [
    {"remote": prefs["remote"]},
    {"salary_min": {"$gte": prefs["salary_min"]}}
]}

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where=where_clause
)

THRESHOLD = 0.4
matches = []
for i, meta in enumerate(results["metadatas"][0]):
    dist = results["distances"][0][i]
    if dist > THRESHOLD:
        continue
    matches.append((dist, meta))

if not matches:
    print("No jobs match your filters. Showing unfiltered results instead.")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    matches = []
    for i, meta in enumerate(results["metadatas"][0]):
        dist = results["distances"][0][i]
        if dist > THRESHOLD:
            continue
        matches.append((dist, meta))

print(f"\nMatching jobs for: {user_input}\n")
print("Recommended:")
for rank, (dist, meta) in enumerate(matches, 1):
    remote_tag = "remote" if meta["remote"] else "on-site"
    print(f"  {rank}. {meta['title']} at {meta['company']} ({remote_tag})")
    print(f"     Salary: ${meta['salary_min']:,} | Location: {meta['location']}")
    print(f"     Match distance: {dist:.4f}")
    print()
