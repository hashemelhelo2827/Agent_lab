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
API_KEY=os.getenv("GEMINI_API_KEY")



# ⚡ BATCH EMBEDDING FUNCTION
def embed_batch(texts):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={API_KEY}"
    
    # Construct requests payload for all items at once
    requests_payload = [
        {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]}
        } for text in texts
    ]
    
    resp = requests.post(url, json={"requests": requests_payload})
    data = resp.json()
    return [e["values"] for e in data["embeddings"]]

# 🛠️ Set cosine similarity distance space
client = chromadb.Client()
collection = client.get_or_create_collection(
     name="movie_catalog",
    metadata={"hnsw:space": "cosine"}
)

base = os.path.dirname(__file__)
movies = []
with open(os.path.join(base, "movie.txt"), newline="", encoding="utf-8") as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    for row in reader:
        movies.append({"id": row[0], "title": row[1], "genre": row[2], "year": int(row[3]), "plot": row[4]})


plots = [m["plot"] for m in movies]
embeddings = embed_batch(plots)

collection.add(
    ids=[m["id"] for m in movies],
    embeddings=embeddings,
    metadatas=[{"title": m["title"], "genre": m["genre"], "year": m["year"]} for m in movies],
    documents=plots
)

query_title = input("What movie u wanna like: ")
client = OpenAI(
    api_key=API_KEY,  
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
prompt_instructions = (
    f"get meta data about this movie : {query_title}. "
    "You must return your response strictly as a JSON object with the following keys:\n"
    "- 'title': movie name \n"
    "- 'genre': type of the movie only one and use short tags\n"
    "- 'year': the published year\n"
    "- 'plot': one sentence illustrate the main idea of the movie"
) 
response=client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={'type':'json_object'},
        messages =[{"role":'user','content':prompt_instructions}]
    )
    
raw_jason=response.choices[0].message.content

query_movie=json.loads(raw_jason)

query_embedding = embed_batch([query_movie["plot"]])[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

exists = collection.get(where={"title": query_movie["title"]})
label = "You liked" if exists["ids"] else "Similar to"
print(f"\n{label}: {query_movie['title']} ({query_movie['genre']}, {query_movie['year']})\n")
THRESHOLD = 0.4
GENRE_PENALTY = 0.15
matches = []
for i, meta in enumerate(results["metadatas"][0]):
    dist = results["distances"][0][i]
    if meta["title"].lower() == query_title.lower():
        continue
    penalty = GENRE_PENALTY if meta["genre"] != query_movie["genre"].lower() else 0
    if dist + penalty > THRESHOLD:
        continue
    matches.append((dist, meta))

if not matches:
    print("No similar movies found.")
else:
    print("Recommended:")
    for rank, (dist, meta) in enumerate(matches, 1):
        print(f"  {rank}. {meta['title']} ({meta['genre']}, {meta['year']}) - {dist:.4f}")
