"""
Value-type classifier.

Two backends are provided:

1. `classify_llm`  — few-shot LLM classification via the Anthropic API.
   This is the intended production path. It builds a prompt that embeds the
   codebook definitions and a few labelled examples per class, then asks the
   model to return a single label. Requires ANTHROPIC_API_KEY in the env.

2. `classify_baseline` — a transparent, dependency-free keyword baseline.
   This lets the whole pipeline (and the validation report) run offline with
   no API key, so the repository is reproducible by anyone. It is deliberately
   simple and is NOT the method being showcased — it is a fallback / sanity
   baseline against which the LLM approach would be compared.

The design point being demonstrated is the SEPARATION of:
  - classification (swappable backend), and
  - validation against a human gold standard (see evaluate.py),
so that the language model is treated as a measurement instrument whose
agreement with expert judgement is measured, not assumed.
"""

import os
import re

CODEBOOK = """\
You are classifying short statements about why a person values a natural place.
Assign exactly ONE of three value types, following these definitions:

- intrinsic: nature (or an element of it) has worth in itself, independent of
  any use or benefit to humans. Cues: "value in itself", "own right to exist",
  "regardless of what we get", "for its own sake".

- instrumental: nature is valued as a means to a human end - a benefit,
  resource, service, or function that could in principle be provided otherwise.
  Cues: carbon storage, drinking water, flood protection, tourism, economy,
  "useful for", "provides", "protects us from".

- relational: what is valued is the meaningful relationship between the person
  (or community) and the place - identity, memory, belonging, care, continuity.
  Cues: "part of who I am", "our family", "memories", "I feel responsible",
  "the bond I have".

Disambiguation:
- If human benefit is mentioned but the point is the RELATIONSHIP (care,
  identity, belonging), label relational, not instrumental.
- If care for nature is mentioned but the point is nature's OWN worth
  (not a relationship), label intrinsic, not relational.
"""

FEWSHOT = [
    ("Nature here has value in itself, regardless of what we get from it.", "intrinsic"),
    ("This forest matters because it stores carbon for our climate.", "instrumental"),
    ("I walked here with my grandfather as a child; this place is part of who I am.", "relational"),
]

LABELS = ["intrinsic", "instrumental", "relational"]


def build_prompt(text):
    examples = "\n".join(f'Statement: "{t}"\nLabel: {l}' for t, l in FEWSHOT)
    return (
        f"{CODEBOOK}\n\n"
        f"Here are labelled examples:\n{examples}\n\n"
        f'Now classify this statement. Reply with only one word '
        f'(intrinsic, instrumental, or relational).\n\n'
        f'Statement: "{text}"\nLabel:'
    )


def classify_llm(text, model="claude-sonnet-4-5"):
    """Few-shot classification via the Anthropic API. Requires anthropic + key."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": build_prompt(text)}],
    )
    out = msg.content[0].text.strip().lower()
    for lab in LABELS:
        if lab in out:
            return lab
    return "unknown"


# --- offline keyword baseline so the repo runs with no API key ---
_KEYWORDS = {
    "intrinsic": [
        "value in itself", "in itself", "own right", "regardless of", "for its own sake",
        "an sich", "eigenes recht", "op zichzelf", "los van", "ongeacht", "in zichzelf",
        "unabhaengig", "waarde op zichzelf",
    ],
    "instrumental": [
        "carbon", "drinking water", "flood", "tourists", "economy", "fish", "useful",
        "provides", "protect our", "protects", "recreation",
        "co2", "trinkwasser", "ueberschwemmung", "touristen", "wirtschaft", "fisch",
        "drinkwater", "overstroming", "toeristen", "economie", "recreatie", "levert",
        "beschermen ons", "schuetzen unser",
    ],
    "relational": [
        "part of who i am", "our family", "memories", "responsible", "bond", "belongs to us",
        "gather here", "connected",
        "gehoert zu mir", "familie", "erinnerungen", "verantwortlich", "verbindung",
        "deel van wie ik ben", "familie verbonden", "herinneringen", "verantwoordelijk",
        "band die ik", "van ons",
    ],
}


def classify_baseline(text):
    t = text.lower()
    scores = {lab: 0 for lab in LABELS}
    for lab, kws in _KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[lab] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "relational"  # default fallback
