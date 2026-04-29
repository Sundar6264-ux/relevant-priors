import os
import json
import hashlib
import logging
import time

import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# gemini-2.5-flash-lite: free tier, 15 RPM, 1000 req/day — more than enough
# for the 996-case eval since we batch all priors per case into one call
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite-preview-06-17",
    generation_config={"temperature": 0}  # deterministic, we want consistent verdicts
)

# cache keyed on (current_desc, prior_desc) — radiology names repeat a ton
# across patients, so this cuts down API calls significantly on the full eval
cache = {}

# prompt went through a few versions. main lesson: have to explicitly say
# cross-modality is fine (CT HEAD is relevant for MRI BRAIN — same region).
# first version was too strict and tanked accuracy on those cases.
PROMPT = """You help radiologists decide which prior imaging studies are worth showing
when they read a new scan.

A prior is RELEVANT if it covers the same or nearby body region — modality doesn't
have to match. CT HEAD is useful when reading MRI BRAIN. MRI SPINE CERVICAL is
borderline but usually useful for a brain study. CHEST X-RAY is not useful for
a brain study. When unsure, lean toward relevant.

You'll get a current study and a numbered list of priors.
Reply with ONLY a JSON array of true/false, one per prior, same order.
Nothing else. No explanation. No markdown. Just the array.

Example: [true, false, true]"""


def make_key(current_desc, prior_desc):
    s = current_desc.lower().strip() + "|||" + prior_desc.lower().strip()
    return hashlib.md5(s.encode()).hexdigest()


def classify(current_study, prior_studies):
    if not prior_studies:
        return []

    keys = [make_key(current_study["study_description"], p["study_description"])
            for p in prior_studies]

    uncached = [i for i, k in enumerate(keys) if k not in cache]

    if uncached:
        batch = [prior_studies[i] for i in uncached]

        prompt = (
            f"{PROMPT}\n\n"
            f"Current study: {current_study['study_description']} "
            f"(date: {current_study['study_date']})\n\n"
            "Prior studies:\n"
        )
        for n, p in enumerate(batch, 1):
            prompt += f"{n}. {p['study_description']} (date: {p['study_date']})\n"

        t0 = time.time()
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            log.info("gemini: %d priors → %s  (%.1fs)", len(batch), raw, time.time() - t0)
        except Exception as e:
            log.error("gemini call failed: %s", e)
            # if the API call itself fails, default to showing everything
            for i in uncached:
                cache[keys[i]] = True
            return [cache[k] for k in keys]

        # strip markdown fences if gemini decides to add them anyway
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        try:
            verdicts = json.loads(raw)
            if not isinstance(verdicts, list) or len(verdicts) != len(batch):
                raise ValueError(f"expected {len(batch)} booleans, got: {raw}")
        except Exception as e:
            # default to true on parse failure — skipping a prediction = wrong answer
            log.warning("couldn't parse response (%s), defaulting all to true", e)
            verdicts = [True] * len(batch)

        for i, verdict in zip(uncached, verdicts):
            cache[keys[i]] = bool(verdict)

    return [cache[k] for k in keys]


@app.post("/predict")
async def predict(req: Request):
    body = await req.json()
    cases = body.get("cases", [])
    log.info("request: challenge=%s  cases=%d", body.get("challenge_id"), len(cases))

    predictions = []
    for case in cases:
        case_id = case["case_id"]
        current = case["current_study"]
        priors = case.get("prior_studies", [])

        log.info("  case %s — %s — %d prior(s)",
                 case_id, current["study_description"], len(priors))

        verdicts = classify(current, priors)

        for prior, verdict in zip(priors, verdicts):
            predictions.append({
                "case_id": case_id,
                "study_id": prior["study_id"],
                "predicted_is_relevant": verdict
            })

    log.info("returning %d predictions", len(predictions))
    return JSONResponse({"predictions": predictions})


@app.get("/health")
def health():
    return {"status": "ok", "cache_size": len(cache)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
