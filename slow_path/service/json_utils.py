import json, re, difflib, math
from typing import Dict, Any, Optional
from .categories import CATEGORIES, SYNONYMS

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    json_re = re.compile(r"\{.*\}", re.DOTALL)
    m = json_re.search(text)
    if not m: 
        return None
    s = m.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def normalize_label(text: str) -> str:
    t = text.lower().strip()
    if t in CATEGORIES: 
        return t
    if t in SYNONYMS: 
        return SYNONYMS[t]
    
    # fuzzy match to closest category
    cand = difflib.get_close_matches(t, CATEGORIES, n=1, cutoff=0.6)
    return cand[0] if cand else "background"

def coerce_to_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    label = obj.get("label") or obj.get("class") or obj.get("category") or ""
    label = normalize_label(str(label))
    conf = obj.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    md = obj.get("metadata", {})
    if not isinstance(md, dict):
        md = {}
    return {"label": label, "confidence": conf, "metadata": md}

def simple_lemma(word: str) -> str:
    """Tiny helper to strip plural 's' or 'es' — crude lemmatization."""
    if word.endswith("ies"):
        return word[:-3] + "y"     # e.g. "bodies" -> "body"
    elif word.endswith("es"):
        return word[:-2]
    elif word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def extract_categories(text: str):
    """
    Extract categories using synonyms + fuzzy match.
    Lightweight version (no spaCy).
    """
    text = text.lower().strip()
    found = set()

    # Handling multi-word synonyms
    for phrase, mapped in SYNONYMS.items():
        if phrase in text:
            found.add(mapped)
            text = text.replace(phrase, "")

    # Spliting remaining words
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    for w in words:
        w = simple_lemma(w)
        if w in CATEGORIES:
            found.add(w)
        elif w in SYNONYMS:
            found.add(SYNONYMS[w])
        else:
            cand = difflib.get_close_matches(w, CATEGORIES, n=1, cutoff=0.75)
            if cand:
                found.add(cand[0])

    if not found:
        found.add("background")

    return list(found)

def json_extraction(decoded: str):

    m = re.search(r"\{.*\}", decoded, flags=re.DOTALL)
    s = m.group(0) if m else decoded.strip()

    try:
        obj = json.loads(s)
    except Exception:
        # Fallback if the model didn't return valid JSON
        obj = {"label": s[:64], "confidence": 0.0, "metadata": {"raw": s}}

    # Normalization
    label = str(obj.get("label", "")).strip()
    conf = obj.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    
    if math.isnan(conf) or math.isinf(conf):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    metadata = obj.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"meta": str(metadata)}

    return {"label": label, "confidence": conf, "metadata": metadata}
