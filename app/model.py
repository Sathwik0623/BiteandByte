import os
# Simple model stub. If you have a HuggingFace finetuned model saved locally, set MODEL_PATH env var to load it.
MODEL_PATH = os.environ.get("MODEL_PATH","")
use_real_model = False
tokenizer = None
model = None

if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        use_real_model = True
    except Exception as e:
        print("Model load failed:", e)
        use_real_model = False

def predict_text(text):
    # Return (category, confidence)
    if use_real_model and tokenizer and model:
        try:
            inputs = tokenizer(text, truncation=True, return_tensors="pt")
            out = model(**inputs)
            import torch, numpy as np
            probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()[0]
            idx = int(probs.argmax())
            conf = float(probs[idx])
            # label mapping must be provided with the real model; fallback below
            return ("unknown", conf)
        except Exception as e:
            print("Model predict error:", e)
    # simple keyword fallback
    text = text.lower()
    if any(x in text for x in ["amz","amazon","flipkart","myntra"]):
        return ("shopping", 0.93)
    if any(x in text for x in ["starbucks","swiggy","pani","chai","juice","food","coffee","snack","chips"]):
        return ("food", 0.88)
    if any(x in text for x in ["kirana","store","mart","veg","grocery"]):
        return ("grocery", 0.87)
    if any(x in text for x in ["hpcl","shell","bpcl","fuel","petrol","diesel"]):
        return ("fuel", 0.9)
    if any(x in text for x in ["electricity","water","internet","broadband","bill","recharge"]):
        return ("utilities", 0.86)
    return ("other", 0.5)
