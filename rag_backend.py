# rag_backend.py
import os
import tempfile
import uuid
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from supabase import create_client, Client

# chroma
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions as ef

# Groq
from groq import Groq

#.env
from dotenv import load_dotenv

load_dotenv()

# -------------------- CONFIG / ENV --------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY environment variable.")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY for Supabase.")

# initialize Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Sentence-transformers embedding model (free)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # dim = 384
sbert = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Chroma client (local persistent)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


# Langchain-style embedding wrapper for Chroma
embedding_fn = ef.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

# FastAPI
app = FastAPI(title="Capstone RAG Backend (Chroma + Supabase + Gemini)")

# -------------------- Pydantic Schemas --------------------
class TrainRequest(BaseModel):
    model_id: str
    model_name: Optional[str] = None
    persona: Optional[str] = None
    instructions: Optional[str] = None
    guardrails: Optional[str] = None

class ChatRequest(BaseModel):
    model_id: str
    query: str
    top_k: Optional[int] = 3

# -------------------- Helpers --------------------
def fetch_web_content(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # remove scripts/styles
        for s in soup(["script", "style"]):
            s.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return f"__FAILED_TO_FETCH__ {e}"

def ensure_model_row(model_id: str, model_name: Optional[str] = None):
    """Ensure a row exists in supabase 'models' table."""
    # Check existence
    res = supabase.table("models").select("*").eq("model_id", model_id).execute()
    if res.data and len(res.data) > 0:
        return res.data[0]

    # Create
    payload = {
        "model_id": model_id,
        "model_name": model_name or model_id,
        "persona": "",
        "instructions": "",
        "guardrails": ""
    }
    supabase.table("models").insert(payload).execute()
    return payload

def chroma_collection_for_model(model_id: str):
    """Return or create a Chroma collection for given model_id (persistence on disk)."""
    col_name = f"model_{model_id}"
    # create_collection will return existing if exists with same name
    try:
        collection = chroma_client.get_collection(name=col_name)
    except Exception:
        collection = chroma_client.create_collection(name=col_name, embedding_function=embedding_fn)
    return collection

# -------------------- ENDPOINTS --------------------

@app.post("/upload-docs")
async def upload_docs(
    model_id: str = Form(...),
    file: UploadFile = File(None),
    url: Optional[str] = Form(None),
):
    """
    Upload a file (PDF/TXT) or a URL to attach to a model.
    This stores the document metadata & text into Supabase 'documents' table (text_content).
    It DOES NOT build embeddings; call /train to build vectors.
    """
    ensure_model_row(model_id)

    uploaded_texts = []

    if file:
        filename = file.filename
        content = await file.read()
        # If PDF, use PyPDFLoader
        if filename.lower().endswith(".pdf"):
            # Save to temp file so PyPDFLoader can read
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs:
                text = d.page_content
                uploaded_texts.append((filename, text))
        else:
            # treat as plain text
            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                text = str(content)
            uploaded_texts.append((filename, text))

    if url:
        text = fetch_web_content(url)
        uploaded_texts.append((url, text))

    # Insert into Supabase documents table
    for src, text in uploaded_texts:
        doc_payload = {
            "doc_id": str(uuid.uuid4()),
            "model_id": model_id,
            "file_name": src if file else None,
            "url": src if url else None,
            "text_content": text
        }
        supabase.table("documents").insert(doc_payload).execute()

    return {"status": "success", "message": f"Uploaded {len(uploaded_texts)} item(s) for model {model_id}."}


@app.post("/train")
def train_model(req: TrainRequest):
    """
    Build embeddings and populate Chroma collection for this model.
    Also update model metadata in Supabase (persona/instructions/guardrails).
    """
    # Ensure model exists
    ensure_model_row(req.model_id, req.model_name)

    # Update model metadata in Supabase
    update_payload = {
        "persona": req.persona or "",
        "instructions": req.instructions or "",
        "guardrails": req.guardrails or "",
        "updated_at": "now()"
    }
    supabase.table("models").update(update_payload).eq("model_id", req.model_id).execute()

    # Fetch all documents for this model (optional)
    res = supabase.table("documents").select("*").eq("model_id", req.model_id).execute()
    docs = res.data or []

    # If no documents, just update model metadata and return success
    if not docs:
        return {
            "status": "success", 
            "message": f"Updated model metadata for {req.model_id}. No documents to process.",
            "documents_processed": 0
        }

    # Chunking strategy: simple split by length -> you can replace with smarter splitter
    chunks = []
    METACHUNK_SIZE = 1000  # chars per chunk, adjustable
    for d in docs:
        text = d.get("text_content") or ""
        # split into chunks
        for i in range(0, len(text), METACHUNK_SIZE):
            chunk_text = text[i:i + METACHUNK_SIZE].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": d["doc_id"],
                    "chunk_text": chunk_text
                })

    # Create/get chroma collection
    collection = chroma_collection_for_model(req.model_id)

    # Prepare metadata lists
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"]} for c in chunks]
    documents_texts = [c["chunk_text"] for c in chunks]

    # Upsert into chroma (it will compute embeddings using embedding_fn)
    # If collection already has items, we remove and re-add to avoid duplicates.
    try:
        collection.delete(ids=ids)  # attempt delete in case duplicates
    except Exception:
        pass

    collection.add(
        ids=ids,
        documents=documents_texts,
        metadatas=metadatas
    )

    # PersistentClient auto-persists, no need to call persist() explicitly

    return {"status": "success", "message": f"Built vector index with {len(chunks)} chunks for model {req.model_id}."}


@app.post("/chat")
def chat(req: ChatRequest):
    """
    RAG: retrieve top-k chunks from Chroma for model_id, then call Gemini
    with persona/instructions/guardrails injected from Supabase model row.
    Also logs the chat in Supabase chat_logs table.
    """
    # ensure model exists and metadata
    res = supabase.table("models").select("*").eq("model_id", req.model_id).execute()
    model_row = (res.data or [None])[0]
    if not model_row:
        return {"error": "Model not found. Build the model first."}

    # ensure collection exists
    collection = chroma_collection_for_model(req.model_id)

    # compute query embedding using same SBERT model (embedding_fn could be used)
    q_embedding = sbert.encode([req.query])[0].tolist()

    # retrieve top-k using chroma (query by embedding)
    try:
        result = collection.query(query_embeddings=[q_embedding], n_results=req.top_k,
                                  include=["documents", "metadatas", "distances"])
    except Exception as e:
        return {"error": f"Failed to query vector DB: {e}"}

    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    context_parts = []
    for i, d in enumerate(docs):
        md = metadatas[i] if i < len(metadatas) else {}
        context_parts.append(f"Source doc: {md.get('doc_id', 'unknown')} \n{d}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else ""

    # Build system prompt from model metadata + guardrails
    persona = model_row.get("persona") or "Helpful assistant."
    instructions = model_row.get("instructions") or "Be clear and concise."
    guardrails = model_row.get("guardrails") or "Do not hallucinate; cite context."

    system_prompt = f"""
Persona: {persona}
Instructions: {instructions}
Guardrails: {guardrails}
"""

    final_prompt = f"""{system_prompt}

Context:
{context}

User question:
{req.query}

Answer the question based only on the context above. If the answer is not present in context, say you don't know or suggest where to look.
"""

    # Call Groq
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nUser question:\n{req.query}\n\nAnswer the question based only on the context above. If the answer is not present in context, say you don't know or suggest where to look."}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        answer_text = response.choices[0].message.content
    except Exception as e:
        # Fallback response when Groq fails
        answer_text = f"I'm a financial advisor assistant. Based on the available context, I cannot provide specific advice at this time. Please ensure your Groq API key is valid and has access to the model. Error: {e}"

    # log chat into supabase chat_logs
    log_payload = {
        "chat_id": str(uuid.uuid4()),
        "model_id": req.model_id,
        "user_query": req.query,
        "answer": answer_text,
        "context_used": context[:3000]  # limit size
    }
    supabase.table("chat_logs").insert(log_payload).execute()

    return {
        "query": req.query,
        "answer": answer_text,
        "sources": [m.get("doc_id") for m in metadatas],
        "distances": distances
    }


@app.get("/metrics")
def metrics():
    """Simple overview"""
    res = supabase.table("models").select("model_id, model_name").execute()
    models = [m["model_id"] for m in (res.data or [])]
    return {"total_models": len(models), "models": models}

# -------------------- Run locally for dev --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_backend:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
