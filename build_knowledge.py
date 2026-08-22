import ollama
import pickle

# ── Step 1: Read the knowledge text ──────────────────────
with open("career_knowledge.txt", "r") as f:
    text = f.read()

print(f"Total characters in knowledge base: {len(text)}")

# ── Step 2: Chunk the text ────────────────────────────────
# 700 chars per chunk with 100 char overlap so context isn't lost at edges
chunk_size = 700
overlap     = 100
chunks = []

i = 0
while i < len(text):
    chunks.append(text[i : i + chunk_size])
    i += chunk_size - overlap          # slide forward with overlap

print(f"Total chunks created: {len(chunks)}")

# ── Step 3: Embed each chunk with nomic-embed-text ────────
# Make sure Ollama is running and you have pulled the model:
# ollama pull nomic-embed-text

embeddings = []
for idx, chunk in enumerate(chunks):
    response  = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
    embeddings.append(response["embedding"])
    print(f"  Embedded chunk {idx + 1}/{len(chunks)}", end="\r")

print(f"\nAll {len(chunks)} chunks embedded!")

# ── Step 4: Save to pkl ───────────────────────────────────
with open("knowledge.pkl", "wb") as f:
    pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

print("knowledge.pkl saved! You can now run app.py")