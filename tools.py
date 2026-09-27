from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import pypdf
from duckduckgo_search import DDGS

_model = SentenceTransformer("all-MiniLM-L6-v2")
_index = None
_chunks = []

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def build_index_from_pdf(pdf_file):
    global _index, _chunks
    reader = pypdf.PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    _chunks = chunk_text(full_text)
    embeddings = _model.encode(_chunks)
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(np.array(embeddings).astype("float32"))
    return len(_chunks)

def pdf_search(query: str, k: int = 3) -> str:
    if _index is None:
        return "No PDF has been uploaded yet."
    q_emb = _model.encode([query]).astype("float32")
    distances, indices = _index.search(q_emb, k)
    results = [_chunks[i] for i in indices[0] if i < len(_chunks)]
    return "\n---\n".join(results) if results else "No relevant passages found."

def web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    return "\n---\n".join(f"{r['title']}: {r['body']}" for r in results)

def calculator(expression: str) -> str:
    try:
        allowed = "0123456789+-*/(). "
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "Expression contains disallowed characters."
    except Exception as e:
        return f"Error: {e}"
