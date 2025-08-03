from pdfminer.high_level import extract_text as extract_pdf
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text(file_path, ext):
    if ext == 'pdf':
        return extract_pdf(file_path)
    elif ext == 'docx':
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return ""

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks, convert_to_tensor=False)

def build_faiss_index(vectors):
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    return index

def semantic_search(index, query_vector, chunks, k=3):
    D, I = index.search(np.array([query_vector]), k)
    return [chunks[i] for i in I[0]]

def embed_query(query):
    return model.encode([query], convert_to_tensor=False)[0]  # Fixed here
