import os
import asyncio
import json
import pickle
import shutil
from typing import List, Dict, Optional

# --- Environment and Configuration ---
from dotenv import load_dotenv
load_dotenv()

# --- Core Libraries ---
import lancedb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Unstructured and Crawl4AI for Ingestion & LLM ---
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import CompositeElement
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMExtractionStrategy, LLMConfig

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

# --- Global Engine and Artifacts ---
adjudicator_engine = None
ARTIFACTS_DIR = "/tmp/hackrx_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- The Adjudicator Engine Class ---
class Adjudicator:
    """The definitive, high-accuracy engine for automated claim adjudication."""
    def __init__(self):
        print("🚀 Initializing the Adjudicator Engine...")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        print("  - Initializing models...")
        self.dense_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        
        # --- MODEL UPGRADE ---
        self.llm_config = LLMConfig(provider="openai/gpt-4o-mini", api_token=OPENAI_API_KEY)
        print("  - ✅ All models initialized. Using openai/gpt-4o-mini for adjudication.")

    async def ingest_document_from_url(self, doc_url: str) -> List[CompositeElement]:
        """Performs real-time, on-the-fly ingestion of a document URL."""
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
            raise HTTPException(status_code=400, detail=f"Failed to ingest document from URL: {e}")

    def build_in_memory_core(self, chunks: List[CompositeElement]) -> Dict:
        """Builds an in-memory Cognitive Core for a single document."""
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
        """Performs hybrid retrieval and reranking using the provided cognitive core."""
        if not core: return []
        
        print(f"  - Performing Hybrid Retrieval for query: '{query}'")
        query_vector = self.dense_model.encode(query)
        vector_results_df = core["table"].search(query_vector).limit(20).to_pandas()
        
        tokenized_query = query.lower().split(" ")
        bm25_scores = core["bm25"].get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
        
        combined_texts = list(dict.fromkeys(
            vector_results_df['text'].tolist() + [core["corpus"][i] for i in top_bm25_indices]
        ))

        sentence_pairs = [[query, text] for text in combined_texts]
        scores = self.cross_encoder.predict(sentence_pairs, show_progress_bar=False)
        reranked_results = sorted(zip(scores, combined_texts), key=lambda x: x[0], reverse=True)
        
        print(f"  - ✅ Retrieved and reranked top {top_k} clauses.")
        return [text for score, text in reranked_results[:top_k]]

    async def adjudicate(self, query: str, core: Dict) -> AdjudicationResult:
        """Runs the definitive two-stage adjudication process."""
        
        candidate_clauses = self._retrieve_and_rerank(query, core=core)
        if not candidate_clauses:
            return AdjudicationResult(decision="Error", amount="N/A", justification="Could not retrieve any clauses.", clauses=[])

        # STAGE 1: Clause Selection
        print("\n  - Adjudication Stage 1: Filtering for hyper-relevant clauses...")
        class ClauseSelectionResponse(BaseModel):
            relevant_clauses: List[str] = Field(description="A list containing the full, verbatim text of the most relevant policy clauses.")

        numbered_candidates = "\n\n".join([f"--- Clause {i+1} ---\n{chunk}" for i, chunk in enumerate(candidate_clauses)])
        selection_prompt = f"From the numbered policy clauses below, extract the full, verbatim text of ALL clauses directly relevant to the user's claim: '{query}'. Consider coverage, waiting periods, and exclusions. Respond ONLY with a JSON object containing a list of strings."
        
        selection_strategy = LLMExtractionStrategy(llm_config=self.llm_config, schema=ClauseSelectionResponse.model_json_schema(), instruction=selection_prompt, extraction_type="schema")
        async with AsyncWebCrawler() as crawler:
            selection_result = await crawler.arun(url=f"raw://{numbered_candidates}", config=CrawlerRunConfig(extraction_strategy=selection_strategy))
        
        final_clauses = candidate_clauses
        if selection_result.success and selection_result.extracted_content:
            try:
                selection_json = json.loads(selection_result.extracted_content)[0]
                extracted_texts = selection_json.get("relevant_clauses", [])
                if extracted_texts:
                    final_clauses = extracted_texts
                    print(f"  - ✅ Stage 1 complete. Refined context to {len(final_clauses)} essential clauses.")
            except Exception as e:
                print(f"  - ⚠️ Stage 1 parsing failed: {e}. Using all clauses as fallback.")

        # STAGE 2: Final Verdict
        print("\n  - Adjudication Stage 2: Making final decision...")
        class AdjudicationResponse(BaseModel):
            decision: str
            amount: Optional[str]
            justification: str
        
        final_context = "\n\n---\n\n".join(final_clauses)
        verdict_prompt = f"""You are a hyper-vigilant insurance claim adjudicator. Follow this reasoning process:
        1. **Facts:** What are the key facts from the user's claim: '{query}'?
        2. **Rules:** From the **Relevant Policy Clauses ONLY** below, what are the specific rules and waiting periods?
        3. **Decision:** Compare the facts to the rules and make a final decision.
        4. **Output:** Generate a JSON object with your decision and a justification summarizing your reasoning.
        **Relevant Policy Clauses ONLY:** {final_context}"""
        
        verdict_strategy = LLMExtractionStrategy(llm_config=self.llm_config, schema=AdjudicationResponse.model_json_schema(), instruction=verdict_prompt, extraction_type="schema")
        async with AsyncWebCrawler() as crawler:
            final_result = await crawler.arun(url="raw://placeholder", config=CrawlerRunConfig(extraction_strategy=verdict_strategy))

        if final_result.success and final_result.extracted_content:
            try:
                final_json = json.loads(final_result.extracted_content)[0]
                final_json['clauses'] = final_clauses
                return AdjudicationResult(**final_json)
            except Exception as e:
                return AdjudicationResult(decision="Error", amount="N/A", justification=f"LLM response parsing failed: {e}", clauses=final_clauses)
        
        return AdjudicationResult(decision="Error", amount="N/A", justification="LLM adjudication failed.", clauses=final_clauses)

# --- FastAPI Application ---
app = FastAPI(title="HackRx Winning Adjudication Engine", version="4.0.0")

@app.on_event("startup")
async def startup_event():
    global adjudicator_engine
    adjudicator_engine = Adjudicator()

@app.post("/hackrx/run", response_model=APIResponse, tags=["Adjudication"])
async def run_adjudication(request: APIRequest):
    if not adjudicator_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized.")
    
    try:
        chunks = await adjudicator_engine.ingest_document_from_url(request.documents)
        request_core = adjudicator_engine.build_in_memory_core(chunks)
        if not request_core:
            raise HTTPException(status_code=500, detail="Failed to build cognitive core.")

        tasks = [adjudicator_engine.adjudicate(q, request_core) for q in request.questions]
        results = await asyncio.gather(*tasks)
        
        return APIResponse(answers=results)
    except Exception as e:
        print(f"FATAL ERROR during adjudication: {e}")
        raise HTTPException(status_code=500, detail=str(e))