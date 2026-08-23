"""
app.py
------
Flask web application for the Resume Ranking System.

Two endpoints:
  POST /api/rank/stored  — JD vs saved corpus vectors
  POST /api/rank/live    — JD + uploaded resumes, ranked on-the-fly
"""

import os
from flask import Flask, request, jsonify, render_template

from modules.extraction   import extract_text_from_bytes, extract_resume_batch
from modules.preprocessing import (
    preprocess_for_sbert, preprocess_for_w2v,
    preprocess_jd_for_sbert, preprocess_jd_for_w2v,
)
from modules.vectorizer import ModelBundle, rank_against_stored, rank_live_resumes

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB upload limit

# ── Paths — adjust these to match your deployment layout ─────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "/Users/prabinrimal/Desktop/New>>>>>/FINAL_WEBSITE/data")

W2V_MODEL_PATH     = os.path.join(DATA_DIR, "word2vec.model")
SBERT_MODEL_NAME   = "all-MiniLM-L6-v2"
W2V_VECTORS_PATH   = os.path.join(DATA_DIR, "resume_vectors_w2v.npy")
SBERT_VECTORS_PATH = os.path.join(DATA_DIR, "resume_vectors_sbert.npy")
W2V_META_PATH      = os.path.join(DATA_DIR, "resume_metadata_w2v.json")
SBERT_META_PATH    = os.path.join(DATA_DIR, "resume_metadata_sbert.json")

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading models and vectors — this may take a moment …")
bundle = ModelBundle(
    w2v_model_path      = W2V_MODEL_PATH,
    sbert_model_name    = SBERT_MODEL_NAME,
    w2v_vectors_path    = W2V_VECTORS_PATH,
    sbert_vectors_path  = SBERT_VECTORS_PATH,
    w2v_metadata_path   = W2V_META_PATH,
    sbert_metadata_path = SBERT_META_PATH,
)
print("Models ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/rank/stored", methods=["POST"])
def rank_stored():
    """
    Mode 1 — Compare JD against pre-saved resume vectors.

    Accepts either:
      • multipart/form-data with field 'jd_file' (PDF)
      • multipart/form-data with field 'jd_text' (plain text)
    """
    jd_text = ""

    if "jd_file" in request.files:
        f = request.files["jd_file"]
        if f.filename == "":
            return jsonify({"error": "No file selected."}), 400
        raw, err = extract_text_from_bytes(f.read())
        if err:
            return jsonify({"error": f"Could not extract JD text: {err}"}), 422
        jd_text = raw
    elif "jd_text" in request.form:
        jd_text = request.form["jd_text"].strip()
    else:
        return jsonify({"error": "Provide 'jd_file' (PDF) or 'jd_text' in the request."}), 400

    if not jd_text:
        return jsonify({"error": "JD text is empty."}), 400

    # Preprocess JD for both models
    jd_sbert = preprocess_jd_for_sbert(jd_text)
    jd_w2v   = preprocess_jd_for_w2v(jd_text)

    top_k = int(request.form.get("top_k", 10))
    results = rank_against_stored(jd_sbert, jd_w2v, bundle, top_k=top_k)

    return jsonify({
        "mode":    "stored",
        "top_k":   top_k,
        "results": results,
    })


@app.route("/api/rank/live", methods=["POST"])
def rank_live():
    """
    Mode 2 — Encode JD and uploaded resumes live, then rank.

    Expects:
      • 'jd_file' or 'jd_text'  — the Job Description
      • 'resume_files'           — one or more PDF resume files (up to 50)
    """
    # ── Parse JD ──────────────────────────────────────────────────────────────
    jd_text = ""
    if "jd_file" in request.files:
        f = request.files["jd_file"]
        raw, err = extract_text_from_bytes(f.read())
        if err:
            return jsonify({"error": f"Could not extract JD: {err}"}), 422
        jd_text = raw
    elif "jd_text" in request.form:
        jd_text = request.form["jd_text"].strip()
    else:
        return jsonify({"error": "Provide 'jd_file' (PDF) or 'jd_text'."}), 400

    if not jd_text:
        return jsonify({"error": "JD text is empty."}), 400

    # ── Parse resume files ────────────────────────────────────────────────────
    resume_files = request.files.getlist("resume_files")
    if not resume_files or all(f.filename == "" for f in resume_files):
        return jsonify({"error": "No resume files uploaded."}), 400
    if len(resume_files) > 50:
        return jsonify({"error": "Maximum 50 resumes per request."}), 400

    pdf_tuples = [(f.filename, f.read()) for f in resume_files if f.filename]

    # ── Extract text ──────────────────────────────────────────────────────────
    extracted = extract_resume_batch(pdf_tuples)

    failed    = [r for r in extracted if r["error"]]
    succeeded = [r for r in extracted if not r["error"]]

    if not succeeded:
        return jsonify({
            "error":   "All resume extractions failed.",
            "details": [{"filename": r["filename"], "error": r["error"]} for r in failed],
        }), 422

    # ── Preprocess ────────────────────────────────────────────────────────────
    processed = []
    for r in succeeded:
        processed.append({
            "resume_id":   r["resume_id"],
            "filename":    r["filename"],
            "sbert_text":  preprocess_for_sbert(r["raw_text"]),
            "w2v_tokens":  preprocess_for_w2v(r["raw_text"]),
        })

    jd_sbert = preprocess_jd_for_sbert(jd_text)
    jd_w2v   = preprocess_jd_for_w2v(jd_text)

    top_k   = int(request.form.get("top_k", 10))
    results = rank_live_resumes(jd_sbert, jd_w2v, processed, bundle, top_k=top_k)

    return jsonify({
        "mode":    "live",
        "top_k":   top_k,
        "total_uploaded": len(pdf_tuples),
        "successfully_extracted": len(succeeded),
        "failed_extractions": [{"filename": r["filename"], "error": r["error"]} for r in failed],
        "results": results,
    })


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
