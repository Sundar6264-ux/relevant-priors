#!/usr/bin/env python3
# tests the exact request/response format from the challenge spec
# usage:
#   python test.py                                          # local
#   python test.py --url https://your-app.railway.app      # deployed

import json, sys, time, urllib.request, urllib.error

url = "http://localhost:8000/predict"
if "--url" in sys.argv:
    url = sys.argv[sys.argv.index("--url") + 1]

# exact format from the challenge brief
request_body = {
    "challenge_id": "relevant-priors-v1",
    "schema_version": 1,
    "generated_at": "2026-04-16T12:00:00.000Z",
    "cases": [
        {
            "case_id": "1001016",
            "patient_id": "606707",
            "patient_name": "Andrews, Micheal",
            "current_study": {
                "study_id": "3100042",
                "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                "study_date": "2026-03-08"
            },
            "prior_studies": [
                {
                    "study_id": "2453245",
                    "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                    "study_date": "2020-03-08"
                },
                {
                    "study_id": "992654",
                    "study_description": "CT HEAD WITHOUT CNTRST",
                    "study_date": "2021-03-08"
                }
            ]
        },
        {
            "case_id": "1001017",
            "patient_id": "707808",
            "patient_name": "Smith, Jane",
            "current_study": {
                "study_id": "3100099",
                "study_description": "CT CHEST WITH CONTRAST",
                "study_date": "2026-04-01"
            },
            "prior_studies": [
                {
                    "study_id": "1122334",
                    "study_description": "CT CHEST WITHOUT CONTRAST",
                    "study_date": "2025-04-01"
                },
                {
                    "study_id": "5566778",
                    "study_description": "PET CT WHOLE BODY",
                    "study_date": "2024-12-01"
                },
                {
                    "study_id": "9988776",
                    "study_description": "MRI BRAIN WITHOUT CONTRAST",
                    "study_date": "2025-01-10"
                },
                {
                    "study_id": "3344556",
                    "study_description": "X-RAY CHEST 2 VIEWS",
                    "study_date": "2024-06-15"
                }
            ]
        }
    ]
}

# what a radiologist would want to see
expected = {
    "2453245": True,   # same exact study — obvious yes
    "992654":  True,   # CT head for brain MRI — same region, different modality
    "1122334": True,   # prior chest CT for current chest CT — yes
    "5566778": True,   # whole body PET includes chest — yes
    "9988776": False,  # brain MRI for chest CT — no
    "3344556": True,   # chest x-ray for chest CT — same region
}

print(f"POST {url}")
t0 = time.time()

try:
    req = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        result = json.loads(r.read())
except urllib.error.URLError as e:
    print(f"\ncouldn't connect: {e}")
    print("is the server running?  →  python main.py")
    sys.exit(1)

elapsed = time.time() - t0
predictions = result.get("predictions", [])

print(f"\n{len(predictions)} predictions in {elapsed:.1f}s\n")
print(f"{'case_id':<12} {'study_id':<10} {'predicted':<12} {'expected':<10} result")
print("-" * 58)

correct = total = 0
for p in predictions:
    sid = p["study_id"]
    predicted = p["predicted_is_relevant"]
    exp = expected.get(sid, "?")
    match = predicted == exp if exp != "?" else None
    if match is not None:
        total += 1
        if match: correct += 1
    mark = "ok" if match else ("WRONG" if match is False else "?")
    print(f"{p['case_id']:<12} {sid:<10} {str(predicted):<12} {str(exp):<10} {mark}")

if total:
    print(f"\naccuracy: {correct}/{total} = {correct/total*100:.0f}%")

print("\nfull response:")
print(json.dumps(result, indent=2))
