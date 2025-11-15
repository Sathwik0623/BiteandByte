from fastapi import Body
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json, os
from app import utils, model as model_module
from fastapi.middleware.cors import CORSMiddleware
APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
TAX_PATH = os.path.join(DATA_DIR, "taxonomy.json")
PRED_LOG = os.path.join(DATA_DIR, "predictions.jsonl")
FEED_LOG = os.path.join(DATA_DIR, "feedback.jsonl")

app = FastAPI(title="Byte&Bite Transaction Categorizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # during dev use "*" or list specific origins like ["http://127.0.0.1:8001"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load taxonomy at startup
def load_taxonomy():
    if os.path.exists(TAX_PATH):
        with open(TAX_PATH) as f:
            return json.load(f)
    return {"version":"1.0","categories":[]}

taxonomy = load_taxonomy()

class PredictIn(BaseModel):
    transaction: str
    amount: Optional[float] = None
    user_id: Optional[str] = None

class FeedbackIn(BaseModel):
    transaction_id: int
    corrected_category: str
    user_id: Optional[str] = None
    notes: Optional[str] = None

@app.get("/taxonomy")
def get_taxonomy():
    return taxonomy

@app.post("/taxonomy")
async def upload_taxonomy(file: UploadFile = File(...), token: str = Form(...)):
    # simple token-based auth for demo
    if token != "admin-token":
        raise HTTPException(status_code=401, detail="unauthorized")
    body = await file.read()
    try:
        obj = json.loads(body)
    except:
        raise HTTPException(status_code=400, detail="invalid json")
    with open(TAX_PATH,"w") as f:
        json.dump(obj, f, indent=2)
    global taxonomy
    taxonomy = load_taxonomy()
    return {"status":"ok","version":taxonomy.get("version","unknown")}


@app.get("/")
def root():
    return {"status": "Byte&Bite API running", "endpoints": ["/predict", "/suggest", "/feedback", "/taxonomy"]}


@app.post("/predict")
def predict(inp: PredictIn):
    raw = inp.transaction or ""
    normalized = utils.normalize_text(raw)
    vpa = utils.extract_vpa(normalized)
    # 1. heuristics (vpa alias / keyword)
    cat, conf, method = utils.heuristic_predict(normalized, taxonomy)
    if method == "none":
        # fallback to ML model (if available) else simple rule
        cat_ml, conf_ml = model_module.predict_text(normalized)
        if conf_ml > conf:
            cat, conf, method = cat_ml, conf_ml, "model"
    # log prediction
    rec = {"transaction_id": utils.next_id(PRED_LOG), "raw_text": raw, "normalized": normalized,
           "predicted_category": cat, "confidence": conf, "method": method}
    with open(PRED_LOG,"a") as f:
        f.write(json.dumps(rec)+"\\n")
    return {"transaction_id": rec["transaction_id"], "category": cat, "confidence": conf, "method": method}

@app.post("/feedback")
def feedback(fb: dict = Body(...)):
    """
    Expected JSON:
    {
      "transaction_id": 123,
      "corrected_category": "grocery",
      "transaction_text": "milkshakes 50",   # optional, pass the text when available
      "add_alias": true,                     # optional boolean - user opted to add alias vote
      "user_id": "..."                       # optional
    }
    """
    # basic validation
    if "transaction_id" not in fb or "corrected_category" not in fb:
        raise HTTPException(status_code=400, detail="transaction_id and corrected_category required")

    rec = {
        "transaction_id": fb.get("transaction_id"),
        "corrected_category": fb.get("corrected_category"),
        "user_id": fb.get("user_id"),
        "notes": fb.get("notes"),
        "transaction_text": fb.get("transaction_text"),
        "ts": __import__("datetime").datetime.utcnow().isoformat()
    }
    # append to feedback log
    with open(FEED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    # If user opted to add alias (or if transaction_text is present),
    # perform a vote for the token. Strategy: use normalized full text or first token.
    try:
        add_alias_flag = bool(fb.get("add_alias", False))
        token_text = fb.get("transaction_text") or fb.get("corrected_text") or ""
        token_norm = utils.normalize_text(token_text).strip()
        # derive token to vote on: prefer the normalized full phrase if short, else first token
        token_to_vote = ""
        if token_norm:
            tokens = token_norm.split()
            if len(tokens) == 1 or len(token_norm) <= 20:
                token_to_vote = token_norm
            else:
                token_to_vote = tokens[0]  # quick heuristic: first token
        if add_alias_flag and token_to_vote:
            vote_result = utils.vote_alias(token_to_vote, rec["corrected_category"])
        else:
            vote_result = {"added": False}
    except Exception as e:
        vote_result = {"error": str(e)}

    return {"status": "ok", "vote": vote_result}

@app.post("/suggest")
def suggest(inp: PredictIn = Body(...)):
    """
    Accepts JSON body: {"transaction":"text", "amount": 123, "user_id":"..."}
    Returns top-3 suggestions.
    """
    transaction = inp.transaction or ""
    amount = inp.amount
    normalized = utils.normalize_text(transaction)
    candidates = []

    # heuristic first
    cat_h, conf_h, _ = utils.heuristic_predict(normalized, taxonomy)
    if cat_h:
        candidates.append({"category": cat_h, "confidence": conf_h, "reason": "heuristic"})

    # model suggestion
    cat_m, conf_m = model_module.predict_text(normalized)
    candidates.append({"category": cat_m, "confidence": conf_m, "reason": "model"})

    # amount heuristics
    if amount is not None:
        if amount <= 100:
            candidates.append({"category": "food", "confidence": 0.45, "reason": "small_amount_heuristic"})
        elif amount <= 500:
            candidates.append({"category": "shopping", "confidence": 0.35, "reason": "mid_amount_heuristic"})

    # sort & dedupe top 3
    seen = set(); out = []
    for c in sorted(candidates, key=lambda x: -x["confidence"]):
        if c["category"] in seen:
            continue
        out.append(c); seen.add(c["category"])
        if len(out) >= 3:
            break

    return {"suggestions": out}


# Admin endpoint: list pending alias votes
@app.get("/admin/pending_aliases")
def admin_pending_aliases():
    pending = utils.list_pending_aliases()
    return {"pending": pending}

# Admin endpoint: approve alias manually
@app.post("/admin/approve_alias")
def admin_approve_alias(payload: dict = Body(...)):
    # expected payload: {"token":"milkshake","category":"food", "admin_token": "admin-token"}
    admin_token = payload.get("admin_token")
    if admin_token != "admin-token":
        raise HTTPException(status_code=401, detail="unauthorized")
    token = payload.get("token")
    category = payload.get("category")
    if not token or not category:
        raise HTTPException(status_code=400, detail="token and category required")
    promoted = utils.approve_alias_admin(token, category)
    return {"promoted": promoted}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
