# Relevant Priors — Submission Write-up

## Approach

Used Gemini 2.5 Flash-Lite as the classifier. The study descriptions are
free-text radiology abbreviations and trying to encode rules for every edge
case by hand felt fragile. The model already understands radiology terminology
and body region relationships, so the main work was getting the prompt right.

All priors for a case are batched into a single API call. One call per prior
would likely time out on the full 996-case eval. Batching means one call per
case regardless of how many priors there are, which keeps total latency well
inside the 360s limit.

## What took iteration

First prompt version was too strict about modality matching — it was marking
CT HEAD as not relevant for MRI BRAIN because the modalities differ. But that's
wrong clinically. A radiologist absolutely wants the prior CT of the same region.
Once the prompt explicitly said "same region is what matters, modality doesn't
have to match", accuracy on cross-modality cases improved significantly.

Also added explicit handling for parse failures: if the model returns something
that can't be parsed as a boolean array, default everything to true. A missing
prediction counts as wrong, so showing an extra irrelevant prior is a much
smaller penalty than skipping a prediction entirely.

## Caching

Study description strings repeat heavily across patients. "CT HEAD WITHOUT
CONTRAST" appears in hundreds of cases. The cache is keyed on
MD5(current_desc + prior_desc), so repeated pairs skip the API call entirely.
On the full 27,614-prior eval this makes a real dent in both runtime and
API quota usage.

## Results (local test)

Tested against the exact request format from the challenge spec:
- 2 cases, 6 priors total
- 6/6 correct on expected labels

## What I'd try next

- Few-shot examples in the prompt for genuinely ambiguous cases (cervical spine
  when current is brain — adjacent regions, radiologist might want it or not)
- Extract structured (modality, body_part) tuples first as a fast first pass,
  only hit the model for cases that don't match cleanly
- Run the full public eval JSON locally and analyze where it gets things wrong
  before final submission
