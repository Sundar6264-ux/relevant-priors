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

# April 2026 Stable Standard
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(
    model_name="gemini-3.1-flash-lite", 
    generation_config={"temperature": 0}
)

cache = {}

# CRITICAL: Added explicit separation for Head vs. Body studies to fix Case 9988776
PROMPT = """You are a radiology logic engine. Identify relevant priors for a new scan.

RULES FOR RELEVANCE:
1. MODALITY IGNORED: CT is relevant for MRI (and vice versa) if the body part matches.
2. REGIONAL ISOLATION:
   - BRAIN/HEAD studies are NOT relevant for CHEST, ABDOMEN, or PELVIS.
   - CHEST/ABDOMEN studies are NOT relevant for BRAIN/HEAD.
3. NEARBY REGIONS: Cervical Spine (Neck) IS relevant for Brain studies.
4. TRUNK CLUSTERING: Chest, Abdomen, and Pelvis are often relevant to each other.

EXAMPLES:
- Current: [MRI BRAIN], Prior: [CT HEAD] -> true
- Current: [CT CHEST], Prior: [MRI BRAIN] -> false
- Current: [CT ABDOMEN], Prior: [CT CHEST] -> true

Reply ONLY with a JSON array of booleans. No text, no markdown."""

def make_key(current_desc, prior_desc):
    s = current_desc.lower().strip() + "|||" + prior_desc.lower().strip()
    return hashlib.md5(s.encode()).hexdigest()

async def classify_async(current_study, prior_studies):
    if not prior_studies:
        return []

    keys = [make_key(current_study["study_description"], p["study_description"])
            for p in prior_studies]
    
    uncached_indices = [i for i, k in enumerate(keys) if k not in cache]

    if uncached_indices:
        batch = [prior_studies[i] for i in uncached_indices]
        prompt = f"{PROMPT}\n\nCurrent study: {current_study['study_description']}\nPriors:\n"
        for n, p in enumerate(batch, 1):
            prompt += f"{n}. {p['study_description']}\n"

        try:
            response = await model.generate_content_async(prompt)
            raw = response.text.strip().strip("`").replace("json", "").strip()
            verdicts = json.loads(raw)
            if not isinstance(verdicts, list) or len(verdicts) != len(batch):
                raise ValueError("Response length mismatch")
        except Exception as e:
            log.warning("Gemini/Parse error: %s. Falling back to True.", e)
            verdicts = [True] * len(batch)

        for idx, v in zip(uncached_indices, verdicts):
            cache[keys[idx]] = bool(v)

    return [cache[k] for k in keys]

@app.post("/predict")
async def predict(req: Request):
    body = await req.json()
    cases = body.get("cases", [])
    t0 = time.time()

    # Parallelize case processing for speed
    tasks = [classify_async(c["current_study"], c.get("prior_studies", [])) for c in cases]
    results = await asyncio.gather(*tasks)

    predictions = []
    for case, case_verdicts in zip(cases, results):
        for prior, v in zip(case.get("prior_studies", []), case_verdicts):
            predictions.append({
                "case_id": case["case_id"],
                "study_id": prior["study_id"],
                "predicted_is_relevant": v
            })

    log.info("Batch processed in %.2fs", time.time() - t0)
    return JSONResponse({"predictions": predictions})

@app.get("/health")
def health():
    return {"status": "ok", "cache_size": len(cache)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)