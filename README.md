# Byte&Bite - Transaction Categorizer (Prototype)

## Overview
Simple prototype for GHCI Round-2. FastAPI backend with a small frontend demo. Includes heuristics and optional HuggingFace model hook.

## Run (local)
1. Create virtual env and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start backend:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
3. Open `frontend/index.html` in browser (or serve via simple http server).

## Notes
- To use a real model, set environment variable `MODEL_PATH` to a local HF model path.
- taxonomy.json can be edited via `/taxonomy` upload (admin-token).

