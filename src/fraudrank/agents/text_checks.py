"""Deterministic checks for user-facing text written by agents.

* every number stated must trace to the evidence the agent was given;
* no technical terms from a fixed list;
* no statement that fraud is proven (negation-aware, shared with the case-file check);
* a reading-ease score, so plain-English claims are measured rather than asserted.
"""
from __future__ import annotations

import math
import re

from .case_file import asserts_fraud

# Technical terms an end user should never have to decode. Matched as whole words.
JARGON = ["shap", "pr-auc", "roc", "auc", "log-odds", "lightgbm", "gradient boosting", "logistic regression",
          "random forest", "cross-validation", "f1", "recall", "precision", "hyperparameter", "feature importance",
          "percentile", "classifier", "embedding", "bm25", "pagerank", "z-score", "model score"]

NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
# Numbers written out in words escape the digit check above. Small ones ("two stays") read naturally and are
# checked against the 0-10 allowance; anything from eleven up is flagged, because the prompt asks for numbers
# written the same way as the evidence.
SPELLED = re.compile(r"\b(eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
                     r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b", re.I)


def numbers_in(obj) -> list[float]:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out += numbers_in(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(m.replace(",", "")) for m in NUM.findall(obj)]
    return out


def allowed_numbers(source) -> set[float]:
    """Numbers a text may state: every source number, common roundings, percentage and
    'lower than' complements. This is a diagnostic, not proof of factual support;
    the final grounding gate binds factual wording to a specific evidence field."""
    allowed = set()
    for v in numbers_in(source):
        cands = [v, round(v), round(v, 1), round(v, 2), v * 100, round(v * 100), round(v * 100, 1),
                 math.floor(v), 100 - v, math.floor(100 - v), round(100 - v),
                 round(v / 1000), round(v / 1e6, 1), round(v / 1e6, 2)]
        allowed.update(round(float(c), 4) for c in cands)
    return allowed


def text_fields(obj, skip=("provider", "caveat_id", "action", "feature", "pattern_id")) -> list[str]:
    if isinstance(obj, dict):
        return [t for k, v in obj.items() if k not in skip for t in text_fields(v, skip)]
    if isinstance(obj, list):
        return [t for v in obj for t in text_fields(v, skip)]
    return [obj] if isinstance(obj, str) else []


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(1, groups)


def reading_ease(text: str) -> float:
    """Flesch reading ease: 60-70 is plain English, 30-50 is college level."""
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * len(words) / sentences - 84.6 * syl / len(words)


def check_text(out: dict, source) -> dict:
    texts = text_fields(out)
    joined = " ".join(texts)
    allowed = allowed_numbers(source)
    stated = [float(m.replace(",", "")) for t in texts for m in NUM.findall(t)]
    unsupported = sorted({n for n in stated if round(n, 4) not in allowed})
    jargon = sorted({j for j in JARGON if re.search(rf"(?<![\w-]){re.escape(j)}(?![\w-])", joined.lower())})
    return {"numbers_stated": len(stated), "numbers_unsupported": len(unsupported),
            "unsupported_values": unsupported[:10], "jargon_terms": jargon,
            "spelled_numbers": sorted({m.lower() for t in texts for m in SPELLED.findall(t)}),
            "says_proven_fraud": any(asserts_fraud(t) for t in texts),
            "reading_ease": round(reading_ease(joined), 1), "words": len(joined.split())}
