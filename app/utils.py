import json, os, threading
import re, unicodedata, json, os
APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR,"data")
TAX_PATH = os.path.join(DATA_DIR,"taxonomy.json")

def normalize_text(t):
    t = unicodedata.normalize('NFKD', str(t))
    t = t.lower()
    t = re.sub(r'[^a-z0-9\s@]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

VPA_RE = re.compile(r'([a-z0-9\.\-_]{2,})@([a-z0-9\-_]{2,})')

def extract_vpa(text):
    m = VPA_RE.search(text)
    return m.group(1) if m else None

def load_taxonomy():
    if os.path.exists(TAX_PATH):
        with open(TAX_PATH) as f:
            return json.load(f)
    return {"version":"1.0","categories":[]}

def heuristic_predict(text, taxonomy):
    # check vpa aliases
    vpa = extract_vpa(text)
    vpa_aliases = taxonomy.get("vpa_aliases",{})
    if vpa and vpa in vpa_aliases:
        return vpa_aliases[vpa], 0.95, "vpa_alias"
    # exact alias match in categories
    for cat_obj in taxonomy.get("categories",[]):
        cat = cat_obj.get("id")
        aliases = cat_obj.get("aliases",[])
        for a in aliases:
            if a in text:
                return cat, 0.85, "alias_keyword"
    # keyword presence
    keyword_map = {}
    for cat_obj in taxonomy.get("categories",[]):
        for a in cat_obj.get("aliases",[]):
            keyword_map[a]=cat_obj.get("id")
    for token in text.split():
        if token in keyword_map:
            return keyword_map[token], 0.8, "keyword"
    return None, 0.0, "none"

# simple id generator
def next_id(logfile):
    if not os.path.exists(logfile):
        return 1
    try:
        with open(logfile) as f:
            lines = f.read().strip().splitlines()
            return (len(lines) + 1)
    except:
        return 1
_ALIAS_VOTES_PATH = os.path.join(DATA_DIR, "alias_votes.json")
_ALIAS_LOCK = threading.Lock()
_PROMOTE_THRESHOLD = 3  # change this if you want faster/slower promotion

def _read_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def add_alias_to_taxonomy(token: str, category: str, tax_path=TAX_PATH):
    token = token.strip().lower()
    if not token or len(token) < 2:
        return False
    try:
        tax = _read_json(tax_path)
    except:
        tax = {"version":"1.0","categories":[]}
    updated = False
    for cat_obj in tax.get("categories", []):
        if cat_obj.get("id") == category:
            aliases = set(cat_obj.get("aliases", []))
            if token not in aliases:
                aliases.add(token)
                cat_obj["aliases"] = sorted(list(aliases))
                updated = True
            break
    if updated:
        _write_json(tax_path, tax)
    return updated

def vote_alias(token: str, category: str, votes_path=_ALIAS_VOTES_PATH, threshold=_PROMOTE_THRESHOLD):
    """
    Increment vote for (token,category). If votes >= threshold, promote alias.
    Returns: { "token":..., "category":..., "votes":N, "promoted": True/False }
    """
    token = token.strip().lower()
    if not token or len(token) < 2:
        return {"ok": False, "reason": "invalid token"}

    key = f"{token}|{category}"
    with _ALIAS_LOCK:
        votes = _read_json(votes_path)
        votes[key] = votes.get(key, 0) + 1
        current = votes[key]
        # Auto-promote if threshold reached
        promoted = False
        if current >= threshold:
            promoted = add_alias_to_taxonomy(token, category)
            # if promoted, remove from votes to keep file clean
            if promoted:
                votes.pop(key, None)
        _write_json(votes_path, votes)
    return {"token": token, "category": category, "votes": current, "promoted": promoted}

def list_pending_aliases(votes_path=_ALIAS_VOTES_PATH):
    """
    Returns list of {token, category, votes} sorted by votes desc
    """
    votes = _read_json(votes_path)
    items = []
    for k, v in votes.items():
        try:
            token, category = k.split("|", 1)
        except:
            continue
        items.append({"token": token, "category": category, "votes": v})
    items.sort(key=lambda x: -x["votes"])
    return items

def approve_alias_admin(token: str, category: str, votes_path=_ALIAS_VOTES_PATH):
    """
    Force promote alias (admin). Returns True if added.
    """
    promoted = add_alias_to_taxonomy(token, category)
    # remove vote record if present
    with _ALIAS_LOCK:
        votes = _read_json(votes_path)
        key = f"{token}|{category}"
        if key in votes:
            votes.pop(key, None)
            _write_json(votes_path, votes)
    return promoted