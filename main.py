import os
import pickle
import json
import lancedb
import asyncio
import hashlib
import aiohttp
import shutil
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Import the core logic components
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMExtractionStrategy, LLMConfig
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Text, CompositeElement
import camelot

# --- Configuration & Models ---
load_dotenv()
API_TOKEN = "3e0af1293189f8f9129d33fb7d568a40c997067afa84f791cbd17e9404b5a35c"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]
# ... (Other Pydantic models: QueryResponse, ClauseSelectionResponse, AdjudicationResponse remain the same as before)

# --- Ingestion & Core Building Logic (Adapted from Colab) ---
def run_ingestion_for_file(pdf_path: str) -> List[CompositeElement]:
    print(f"  - Running ingestion for: {pdf_path}")
    elements = partition_pdf(filename=pdf_path, strategy="hi_res", infer_table_structure=True)
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        if tables.n > 0:
            for table in tables:
                elements.append(Text(f"TABLE:\n\n{table.df.to_markdown()}\n"))
    except Exception:
        pass # Ignore camelot errors for non-standard PDFs
    chunks = chunk_by_title(elements, max_characters=1024)
    for chunk in chunks:
        chunk.metadata.source = os.path.basename(pdf_path)
    return chunks

def build_cognitive_core(chunks: List[CompositeElement], output_dir: str):
    print(f"  - Building Cognitive Core in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    corpus_texts = [chunk.text for chunk in chunks]
    
    # Vector Store
    embedding_model = SentenceTransformer('all-Mini-LM-L6-v2')
    vectors = embedding_model.encode(corpus_texts)
    data_for_db = [{"vector": v, "text": t, "source": c.metadata.source} for v, t, c in zip(vectors, corpus_texts, chunks)]
    db = lancedb.connect(os.path.join(output_dir, "db.lancedb"))
    db.create_table("policies", data=data_for_db)
    
    # Sparse Index
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(os.path.join(output_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    print("  - ✅ Core built successfully.")

# --- The Adjudicator Engine (Unchanged) ---
class Adjudicator:
    # ... (The entire Adjudicator class from our last version goes here, unchanged) ...
    # It will be initialized dynamically with the correct paths.

# --- FastAPI Application & Caching Logic ---
app = FastAPI(title="HackRx Dynamic Adjudication Engine")
bearer_scheme = HTTPBearer()

# In-memory cache to map document URLs to their Adjudicator instances
ADJUDICATOR_CACHE = {}

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")

async def get_or_create_adjudicator(doc_url: str) -> Adjudicator:
    if doc_url in ADJUDICATOR_CACHE:
        print(f"✅ Found cached adjudicator for: {doc_url}")
        return ADJUDICATOR_CACHE[doc_url]

    print(f"🔥 No cache found. Building new adjudicator for: {doc_url}")
    
    # Create a unique, safe directory name from the URL hash
    url_hash = hashlib.md5(doc_url.encode()).hexdigest()
    core_path = os.path.join(CACHE_DIR, url_hash)
    
    # Download the file
    temp_file_path = os.path.join(CACHE_DIR, f"{url_hash}.pdf")
    async with aiohttp.ClientSession() as session:
        async with session.get(doc_url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=400, detail="Could not download document from URL.")
            with open(temp_file_path, 'wb') as f:
                f.write(await resp.read())
    
    # Run the full pipeline
    chunks = run_ingestion_for_file(temp_file_path)
    build_cognitive_core(chunks, core_path)
    
    # Initialize and cache the new Adjudicator
    adjudicator = Adjudicator(
        db_path=os.path.join(core_path, "db.lancedb"),
        bm25_path=os.path.join(core_path, "bm25.pkl")
    )
    ADJUDICATOR_CACHE[doc_url] = adjudicator
    
    # Clean up the downloaded PDF
    os.remove(temp_file_path)
    
    return adjudicator

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def process_queries(request: QueryRequest):
    try:
        adjudicator = await get_or_create_adjudicator(request.documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        
    tasks = [adjudicator.adjudicate_single_question(q) for q in request.questions]
    results = await asyncio.gather(*tasks)
    
    answers = []
    for res in results:
        if res and 'justification' in res:
            answers.append(res['justification'])
        else:
            answers.append("An error occurred while generating a response for this question.")
            
    return QueryResponse(answers=answers)
