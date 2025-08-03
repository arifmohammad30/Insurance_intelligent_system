import os
import asyncio
import json
import shutil
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import openai

# --- Environment and Configuration ---
from dotenv import load_dotenv
load_dotenv()

# --- Core Libraries ---
import lancedb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# --- Unstructured for Ingestion ---
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import CompositeElement

# --- Models for API and Structured Output ---
class AdjudicationResult(BaseModel):
    decision: str = Field(description="The final decision: 'Approved' or 'Rejected'.")
    amount: Optional[str] = Field(description="The approved payout amount, if applicable. Otherwise 'Not Applicable'.")
    justification: str = Field(description="A concise, step-by-step explanation for the decision.")
    clauses: List[str] = Field(description="The verbatim text of the policy clauses used for the decision.")

class APIRequest(BaseModel):
    documents: str
    questions: List[str]

class APIResponse(BaseModel):
    answers: List[AdjudicationResult]

# --- Global Engine, Security, and Cache ---
adjudicator_engine = None
ARTIFACTS_DIR = "/tmp/hackrx_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
CORE_CACHE = {} # In-memory cache for cognitive cores

API_TOKEN = os.getenv("HACKRX_API_TOKEN")
auth_scheme = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing authentication token")
    return credentials

# --- The Adjudicator Engine Class ---
class Adjudicator:
    def __init__(self):
        print("🚀 Initializing the Adjudicator Engine...")
        
        # Configure the client to use Groq's API
        self.llm_client = openai.OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

        print("  - Initializing models...")
        self.dense_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        print("  - ✅ All models initialized.")

    async def ingest_document_from_url(self, doc_url: str) -> List[CompositeElement]:
        print(f"⚡ Performing real-time ingestion for URL: {doc_url}")
        try:
            response = requests.get(doc_url)
            response.raise_for_status()
            temp_file_path = os.path.join(ARTIFACTS_DIR, os.path.basename(doc_url).split('?')[0])
            with open(temp_file_path, "wb") as f:
                f.write(response.content)
            
            elements = partition(filename=temp_file_path)
            chunks = chunk_by_title(elements, max_characters=1024)
            os.remove(temp_file_path)
            print(f"  - ✅ Ingested and chunked into {len(chunks)} blocks.")
            return chunks
        except Exception as e:
            raise RuntimeError(f"Failed to ingest document from URL: {e}")

    def build_in_memory_core(self, chunks: List[CompositeElement]) -> Dict:
        print("🧠 Building In-Memory Cognitive Core...")
        if not chunks: return None
        corpus_texts = [chunk.text for chunk in chunks]
        vectors = self.dense_model.encode(corpus_texts)
        temp_db_path = os.path.join(ARTIFACTS_DIR, "temp_db")
        if os.path.exists(temp_db_path): shutil.rmtree(temp_db_path)
        db = lancedb.connect(temp_db_path)
        table = db.create_table("temp_doc", data=[{"vector": v, "text": t} for v, t in zip(vectors, corpus_texts)])
        tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        print("  - ✅ In-Memory Core is ready.")
        return {"table": table, "bm25": bm25, "corpus": corpus_texts}

    def _retrieve_and_rerank(self, query: str, core: Dict, top_k: int = 15) -> List[str]:
        if not core: return []
        print(f"  - Performing Hybrid Retrieval for query: '{query}'")
        query_vector = self.dense_model.encode(query)
        vector_results_df = core["table"].search(query_vector).limit(20).to_pandas()
        tokenized_query = query.lower().split(" ")
        bm25_scores = core["bm25"].get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
        combined_texts = list(dict.fromkeys(vector_results_df['text'].tolist() + [core["corpus"][i] for i in top_bm25_indices]))
        sentence_pairs = [[query, text] for text in combined_texts]
        scores = self.cross_encoder.predict(sentence_pairs, show_progress_bar=False)
        reranked_results = sorted(zip(scores, combined_texts), key=lambda x: x[0], reverse=True)
        print(f"  - ✅ Retrieved and reranked top {top_k} clauses.")
        return [text for score, text in reranked_results[:top_k]]

    async def adjudicate(self, query: str, core: Dict) -> AdjudicationResult:
        candidate_clauses = self._retrieve_and_rerank(query, core=core)
        if not candidate_clauses:
            return AdjudicationResult(decision="Error", amount="N/A", justification="Could not retrieve any clauses.", clauses=[])
        
        # Stage 1: Clause Selection
        print("\n  - Adjudication Stage 1: Filtering for hyper-relevant clauses...")
        numbered_candidates = "\n\n".join([f"--- Clause {i+1} ---\n{chunk}" for i, chunk in enumerate(candidate_clauses)])
        selection_prompt = f"From the numbered policy clauses below, extract the full, verbatim text of ALL clauses directly relevant to the user's claim: '{query}'. Consider coverage, waiting periods, and exclusions. Respond ONLY with a JSON object with a single key 'relevant_clauses' which is a list of strings."
        
        try:
            response = self.llm_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": selection_prompt}],
                response_format={"type": "json_object"}
            )
            selection_json = json.loads(response.choices[0].message.content)
            final_clauses = selection_json.get("relevant_clauses", [])
            if not final_clauses:
                print("  - ⚠️ Stage 1 returned no selection. Using all clauses as fallback.")
                final_clauses = candidate_clauses
            else:
                print(f"  - ✅ Stage 1 complete. Refined context to {len(final_clauses)} essential clauses.")
        except Exception as e:
            print(f"  - ⚠️ Stage 1 failed: {e}. Using all clauses as fallback.")
            final_clauses = candidate_clauses
            
        # Stage 2: Final Verdict
        print("\n  - Adjudication Stage 2: Making final decision...")
        final_context = "\n\n---\n\n".join(final_clauses)
        verdict_prompt = f"""You are a hyper-vigilant insurance claim adjudicator. You must follow these strict rules:
        1.  Analyze the user's claim against the **Relevant Policy Clauses ONLY**.
        2.  If the user asks a "what is" or "explain" question, provide a direct, factual answer in the 'decision' field. Do not use words like 'Approved' or 'Rejected'.
        3.  For the 'amount' field, if no specific monetary value is relevant, you **MUST** return the string "Not Applicable". Do not return the number 0 or a null value.
        4.  Provide a step-by-step 'justification' for your decision. The justification must be a single string, not a list.
        5.  Generate a JSON object with your 'decision', 'amount', and 'justification'.

        **User's Claim:** '{query}'
        
        **Relevant Policy Clauses ONLY:** {final_context}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": verdict_prompt}],
                response_format={"type": "json_object"}
            )
            final_json = json.loads(response.choices[0].message.content)

            # Defensive coding to prevent Pydantic validation errors
            if 'justification' in final_json and isinstance(final_json['justification'], list):
                final_json['justification'] = ' '.join(map(str, final_json['justification']))

            if 'amount' not in final_json or final_json['amount'] is None:
                final_json['amount'] = "Not Applicable"
            else:
                final_json['amount'] = str(final_json['amount'])

            final_json['clauses'] = final_clauses
            return AdjudicationResult(**final_json)
        except Exception as e:
            return AdjudicationResult(decision="Error", amount="Not Applicable", justification=f"LLM response parsing failed: {e}", clauses=final_clauses)

# --- FastAPI Application ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global adjudicator_engine
    adjudicator_engine = Adjudicator()
    print("✅ Adjudication Engine is initialized and ready.")
    yield
    print("Adjudication Engine is shutting down.")

app = FastAPI(
    title="HackRx Winning Adjudication Engine",
    version="Final-Groq",
    lifespan=lifespan
)

@app.post("/hackrx/run", response_model=APIResponse, tags=["Adjudication"])
async def run_adjudication(request: APIRequest, token: str = Depends(verify_token)):
    if not adjudicator_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized.")
    
    doc_url = request.documents
    request_core = None

    if doc_url in CORE_CACHE:
        print("⚡️ Cache HIT. Using pre-built in-memory core.")
        request_core = CORE_CACHE[doc_url]
    else:
        print("🥶 Cache MISS. Performing real-time ingestion and build.")
        try:
            chunks = await adjudicator_engine.ingest_document_from_url(doc_url)
            request_core = adjudicator_engine.build_in_memory_core(chunks)
            CORE_CACHE[doc_url] = request_core
        except Exception as e:
            print(f"FATAL ERROR during ingestion/build: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    if not request_core:
        raise HTTPException(status_code=500, detail="Failed to build or retrieve cognitive core.")

    try:
        tasks = [adjudicator_engine.adjudicate(q, request_core) for q in request.questions]
        results = await asyncio.gather(*tasks)
        return APIResponse(answers=results)
    except Exception as e:
        print(f"FATAL ERROR during adjudication: {e}")
        raise HTTPException(status_code=500, detail=str(e))