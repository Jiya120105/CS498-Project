import json, re, difflib
from typing import Dict, Any, Optional, List
# import spacy
from .categories import CATEGORIES, SYNONYMS

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    m = JSON_RE.search(text)
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

# def extract_categories_nlp(text: str):
#     """
#     Extracts normalized categories from a sentence using synonyms,
#     lemmatization, and fuzzy matching.
#     """
#     text = text.lower().strip()
#     found = set()
#     nlp = spacy.load("en_core_web_sm")

#     # Step 1: Handle multi-word synonyms first (e.g. "soccer ball")
#     for phrase, mapped in SYNONYMS.items():
#         if phrase in text:
#             found.add(mapped)
#             # Optionally remove it to avoid double-counting
#             text = text.replace(phrase, "")

#     # Step 2: Lemmatize remaining words
#     doc = nlp(text)
#     words = [token.lemma_ for token in doc if token.is_alpha]

#     # Step 3: Match by synonym or fuzzy similarity
#     for w in words:
#         if w in CATEGORIES:
#             found.add(w)
#         elif w in SYNONYMS:
#             found.add(SYNONYMS[w])
#         else:
#             # Fuzzy match to known categories
#             cand = difflib.get_close_matches(w, CATEGORIES, n=1, cutoff=0.75)
#             if cand:
#                 found.add(cand[0])

#     # Step 4: Default fallback if nothing found
#     if not found:
#         found.add("background")

#     return list(found)


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

    # Step 1: multi-word synonyms
    for phrase, mapped in SYNONYMS.items():
        if phrase in text:
            found.add(mapped)
            text = text.replace(phrase, "")

    # Step 2: split remaining words
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
