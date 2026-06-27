import chromadb

with open("policies.txt", "r") as file:
    policy_text = file.read()

chunks = policy_text.split("## ")
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection("policies")
except:
    pass

collection = chroma_client.get_or_create_collection(name="policies")

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"Indexed {len(chunks)} policy chunks.")