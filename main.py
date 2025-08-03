from fastapi import FastAPI, UploadFile, File, Form
import tempfile, os
from utils import (
    extract_text,
    chunk_text,
    embed_chunks,
    build_faiss_index,
    semantic_search,
    embed_query
)
from transformers import pipeline

app = FastAPI()

# Load the text generation model
qa_model = pipeline("text2text-generation", model="google/flan-t5-large")

@app.post("/query/")
async def query_file(file: UploadFile = File(...), query: str = Form(...)):
    # Extract file extension
    ext = file.filename.split('.')[-1]

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Extract text from file
    text = extract_text(tmp_path, ext)
    os.remove(tmp_path)

    # Fail early if text is empty
    if not text.strip():
        return {"error": "Could not extract text from the document."}

    # Chunk, embed, and build index
    chunks = chunk_text(text)
    vectors = embed_chunks(chunks)
    index = build_faiss_index(vectors)

    # Embed the query and search
    query_vec = embed_query(query)
    relevant_chunks = semantic_search(index, query_vec, chunks)
    context = "\n\n".join(relevant_chunks)

    # Prompt to encourage clear, concise, policy-specific answers
    prompt = f"""
You are a policy expert. Based on the context below from an insurance policy document, answer the user's question as clearly and factually as possible.

Context:
{context}

Question:
{query}

Respond in a single sentence, summarizing the exact benefit, rule, or clause if found:
"""

    # Get answer from model
    raw_output = qa_model(prompt, max_length=256, do_sample=False)[0]['generated_text']
    answer = raw_output.strip().replace("\n", " ")
    cleaned_chunks = [chunk.strip().replace("\n", " ") for chunk in relevant_chunks]

    # Return structured response
    return {
        "query": query,
        "answers": [answer],
        "context_used": cleaned_chunks
    }
