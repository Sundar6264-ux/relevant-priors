import os
import json
import hashlib
import logging
import asyncio
import time

import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()

# April 2026 Update: gemini-3.1-flash-lite is the current production standard
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(
    model_name="gemini-3.1-flash-lite", 
    generation_config={"temperature": 0}
)

cache = {}

# Refined Prompt: Few-Shot examples and clear anatomical boundaries
PROMPT = """You are a radiology logic engine. Identify relevant priors for a radiologist.

RELEVANCE RULES:
1. ANATOMICAL BOUNDARY: The brain is clinically isolated from the trunk.
   - BRAIN imaging is NOT relevant for CHEST, ABDOMEN, or PELVIS studies.
   - CHEST/ABDOMEN imaging is NOT relevant for BRAIN/HEAD studies.
2. MODALITY IGNORED: CT is relevant for MRI if they cover the same body part.
3. NEARBY REGIONS: Neck (Cervical Spine) is relevant for Brain studies.

EXAMPLES:
- Current: [MRI BRAIN], Priors: [1. CT HEAD, 2. X-RAY CHEST] -> [true, false]
- Current: [CT CHEST], Priors: [1. MRI BRAIN, 2. CT ABDOMEN] -> [false, true]

Reply ONLY with a JSON array of booleans. No text or markdown."""

def make_key(current_desc, prior_desc):
    s = current_desc.lower().strip() + "|||" + prior_desc.lower().strip()
    return hashlib.md5(s.encode()).hexdigest()

async def classify_async(current_study, prior_studies):
    """Processes all priors for a single case asynchronously."""
    if not prior_studies:
        return []

    keys = [make_key(current_study["study_description"], p["study_description"])
            for p in prior_studies]
    
    uncached_indices = [i for i, k in enumerate(keys) if k not in cache]

    if uncached_indices:
        batch = [prior_studies[i] for i in uncached_indices]
        
        prompt = (
            f"{PROMPT}\n\n"
            f"Current study: {current_study['study_description']}\n"
            "Prior studies:\n"
        )
        for n, p in enumerate(batch, 1):
            prompt += f"{n}. {p['study_description']}\n"

        try:
            # Use the async SDK call to prevent blocking the event loop
            response = await model.generate_content_async(prompt)
            raw = response.text.strip().strip("`").replace("json", "").strip()
            verdicts = json.loads(raw)
            
            if not isinstance(verdicts, list) or len(verdicts) != len(batch):
                raise ValueError("List mismatch")
                
        except Exception as e:
            log.warning("Gemini/Parse failure: %s. Defaulting to True.", e)
            verdicts = [True] * len(batch)

        for idx, verdict in zip(uncached_indices, verdicts):
            cache[keys[idx]] = bool(verdict)

    return [cache[k] for k in keys]

@app.post("/predict")
async def predict(req: Request):
    body = await req.json()
    cases = body.get("cases", [])
    t_start = time.time()

    # CRITICAL OPTIMIZATION: Run all cases in parallel
    tasks = [classify_async(c["current_study"], c.get("prior_studies", [])) for c in cases]
    results = await asyncio.gather(*tasks)

    predictions = []
    for case, case_verdicts in zip(cases, results):
        case_id = case["case_id"]
        for prior, v in zip(case.get("prior_studies", []), case_verdicts):
            predictions.append({
                "case_id": case_id,
                "study_id": prior["study_id"],
                "predicted_is_relevant": v
            })

    log.info("Processed %d cases in %.2fs", len(cases), time.time() - t_start)
    return JSONResponse({"predictions": predictions})

@app.get("/health")
def health():
    return {"status": "ok", "cache_size": len(cache)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))