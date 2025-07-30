import os
import pickle
import json
import lancedb
import asyncio
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Import the core logic components
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMExtractionStrategy, LLMConfig

# --- Configuration & Models ---
load_dotenv()
# The API token required by the hackathon judging platform
API_TOKEN = "3e0af1293189f8f9129d33fb7d568a40c997067afa84f791cbd17e9404b5a35c"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in .env file. Please create it.")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class ClauseSelectionResponse(BaseModel):
    relevant_clause_numbers: List[int]

class AdjudicationResponse(BaseModel):
    decision: str
    amount: Optional[str] = "Not Applicable"
    justification: str

# --- The Adjudicator Engine ---
class Adjudicator:
    def __init__(self, db_path, bm25_path):
        print("🚀 Initializing Adjudicator Engine...")
        db = lancedb.connect(db_path)
        self.table = db.open_table("policies")
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        self.corpus_texts = self.table.to_pandas()['text'].tolist()
        
        self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Main model for reasoning
        self.llm_config = LLMConfig(provider="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1", api_token=TOGETHER_API_KEY)
        # Specialized model for instruction-following (JSON selection)
        self.selection_llm_config = LLMConfig(provider="together_ai/meta-llama/Llama-3-8b-chat-hf", api_token=TOGETHER_API_KEY)
        print("✅ Engine Ready!")

    def _retrieve_and_rerank(self, query: str, top_k: int) -> List[str]:
        query_vector = self.dense_model.encode(query)
        vector_results = self.table.search(query_vector).limit(20).to_pandas()
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
        combined_indices = set(vector_results.index) | set(top_bm25_indices)
        sentence_pairs = [[query, self.corpus_texts[i]] for i in combined_indices]
        scores = self.cross_encoder.predict(sentence_pairs)
        reranked_results = sorted(zip(scores, combined_indices), key=lambda x: x[0], reverse=True)
        return [self.corpus_texts[idx] for score, idx in reranked_results[:top_k]]

    async def adjudicate_single_question(self, query: str) -> Optional[Dict]:
        print(f"\n🚀 Running Full Adjudication for: '{query}'")
        
        # --- Part 1: Attack & Defend Retrieval ---
        defend_clauses = self._retrieve_and_rerank(query, top_k=10)
        attack_query = f"reasons to reject or exclude a claim for {query} including waiting periods and specific exclusions"
        attack_clauses = self._retrieve_and_rerank(attack_query, top_k=5)
        
        combined_clauses = []
        seen_texts = set()
        for clause in defend_clauses + attack_clauses:
            if clause not in seen_texts:
                combined_clauses.append(clause)
                seen_texts.add(clause)
        
        candidate_clauses = combined_clauses
        if not candidate_clauses: return None

        # --- Part 2: Two-Stage Logic (Selector) ---
        print(f"  - [Stage 1] Selecting from {len(candidate_clauses)} candidates...")
        numbered_candidates = "\n\n".join([f"--- Clause {i+1} ---\n{chunk}" for i, chunk in enumerate(candidate_clauses)])
        selection_prompt = f"""From the numbered policy clauses below, identify ALL clauses that are directly relevant to making a decision on the user's claim. Respond ONLY with a JSON object containing a list of the relevant integer numbers.
**User's Claim:** "{query}"
**Numbered Policy Clauses:**\n{numbered_candidates}"""

        selection_strategy = LLMExtractionStrategy(llm_config=self.selection_llm_config, schema=ClauseSelectionResponse.model_json_schema(), instruction=selection_prompt, extraction_type="schema")
        
        final_clauses = candidate_clauses # Default
        try:
            async with AsyncWebCrawler() as crawler:
                selection_result = await crawler.arun(url="raw://placeholder", config=CrawlerRunConfig(extraction_strategy=selection_strategy))
            if selection_result.success and selection_result.extracted_content:
                raw_output = json.loads(selection_result.extracted_content)
                selection_json = raw_output[0] if isinstance(raw_output, list) and raw_output else raw_output
                if selection_json and isinstance(selection_json, dict):
                    selected_indices = [i - 1 for i in selection_json.get("relevant_clause_numbers", [])]
                    if selected_indices:
                        final_clauses = [candidate_clauses[i] for i in selected_indices if 0 <= i < len(candidate_clauses)]
        except Exception:
            pass # Fallback to using all clauses
        print(f"  - [Stage 1] ✅ Refined to {len(final_clauses)} essential clauses.")

        # --- Part 3: Advanced Prompting (Verdict) ---
        print("  - [Stage 2] Making final decision...")
        final_context = "\n\n---\n\n".join(final_clauses)
        verdict_prompt = f"""You are an expert insurance claim adjudicator. Your task is to analyze the user's claim against a small, highly relevant set of policy clauses and produce a final, structured decision in JSON format.
First, think step-by-step to outline your reasoning based on the provided clauses. Check for waiting periods, exclusions, and coverage limits.
Second, based on your reasoning, produce the final JSON object.

**--- EXAMPLE ---**
**Claim:** "I have a 3-month-old policy and need cataract surgery."
**Clauses:** "Clause 4.b: A waiting period of 24 months applies to cataract surgery."
**Reasoning:**
1. The user's policy is 3 months old.
2. Clause 4.b states a 24-month waiting period for cataract surgery.
3. The user is within the waiting period.
4. Therefore, the claim must be rejected.
**Final JSON:**
{{
  "decision": "Rejected",
  "amount": "Not Applicable",
  "justification": "The claim for cataract surgery is rejected as the 3-month-old policy is within the 24-month waiting period specified in Clause 4.b."
}}
**--- END EXAMPLE ---**

**--- CURRENT CASE ---**
**Claim:** "{query}"
**Clauses:**\n{final_context}

**Reasoning:**
1. [Your reasoning steps here]

**Final JSON:**
"""

        verdict_strategy = LLMExtractionStrategy(llm_config=self.llm_config, schema=AdjudicationResponse.model_json_schema(), instruction=verdict_prompt, extraction_type="schema")
        
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url="raw://placeholder", config=CrawlerRunConfig(extraction_strategy=verdict_strategy))
            if not result.success or not result.extracted_content: return None
            raw_output = json.loads(result.extracted_content)
            response_dict = raw_output[0] if isinstance(raw_output, list) and raw_output else raw_output
            return response_dict if isinstance(response_dict, dict) else None
        except Exception:
            return None

# --- FastAPI Application ---
app = FastAPI(title="HackRx Adjudication Engine")
bearer_scheme = HTTPBearer()

adjudicator = Adjudicator(db_path="./policies.lancedb", bm25_path="./bm25_index.pkl")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def process_queries(request: QueryRequest):
    tasks = [adjudicator.adjudicate_single_question(q) for q in request.questions]
    results = await asyncio.gather(*tasks)
    
    answers = []
    for res in results:
        if res and 'justification' in res:
            answers.append(res['justification'])
        else:
            answers.append("An error occurred while generating a response for this question.")
            
    return QueryResponse(answers=answers)
