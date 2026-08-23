# Resume Ranking System — Flask Web App

BERT + Word2Vec Hybrid Ensemble · Everest Engineering College

---

## Project Layout

```
resume_ranker/
├── app.py                      ← Flask application (entry point)
├── requirements.txt
├── data/                       ← Copy your saved artefacts here
│   ├── word2vec.model
│   ├── resume_vectors_w2v.npy
│   ├── resume_vectors_sbert.npy
│   ├── resume_metadata_w2v.json
│   └── resume_metadata_sbert.json
├── modules/
│   ├── extraction.py           ← PDF text extraction (pdfplumber)
│   ├── preprocessing.py        ← SBERT light-clean + W2V tokenisation
│   └── vectorizer.py           ← Encode + hybrid rank (both modes)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Quick Setup

### 1. Copy your saved artefacts into `data/`

```bash
cp /path/to/word2vec.model              data/
cp /path/to/resume_vectors_w2v.npy      data/
cp /path/to/resume_vectors_sbert.npy    data/
cp /path/to/resume_metadata_w2v.json    data/
cp /path/to/resume_metadata_sbert.json  data/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For spaCy NER-based name removal (optional):

```bash
python -m spacy download en_core_web_sm
```

### 3. Run

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Two Ranking Modes

### Mode 1 — Corpus Search (tab: "Corpus Search")

HR pastes or uploads a Job Description PDF.  
The system encodes the JD and ranks it against the **pre-built corpus
vectors** (1 429 resumes). No re-encoding of resumes required — very fast.

**Pipeline:**

```
JD text
  └─ preprocess_jd_for_sbert()  →  encode_sbert()  ─┐
  └─ preprocess_jd_for_w2v()   →  encode_w2v()   ─┤
                                                      ├─ hybrid cosine rank
                               stored SBERT matrix ──┘
                               stored W2V   matrix ──┘
```

### Mode 2 — Live Ranking (tab: "Live Ranking")

HR uploads a JD **and** 10–50 resume PDFs.  
Everything is extracted, preprocessed, and encoded on-the-fly.

**Pipeline:**

```
JD + Resume PDFs
  └─ extract_text_from_bytes()     ← pdfplumber
  └─ preprocess_for_sbert/w2v()   ← shared cleaning functions
  └─ encode_sbert / encode_w2v()  ← same models as corpus mode
  └─ cosine_similarity → hybrid score → rank
```

---

## Hybrid Scoring

```
hybrid_score = 0.60 × sbert_cosine + 0.40 × w2v_cosine
```

Both individual and hybrid scores are shown in the results.

---

## Configuration (top of `modules/vectorizer.py`)

| Variable      | Default | Description                         |
|---------------|---------|-------------------------------------|
| `TOP_K`       | 10      | Number of results to return         |
| `SBERT_WEIGHT`| 0.60    | Weight for SBERT cosine score       |
| `W2V_WEIGHT`  | 0.40    | Weight for Word2Vec cosine score    |
| `CHUNK_SIZE`  | 200     | Words per SBERT sliding-window chunk|
| `CHUNK_STRIDE`| 150     | Stride between consecutive chunks  |

---

## Production Deployment

For production use Gunicorn instead of the Flask dev server:

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

> **Note:** Use `-w 1` (single worker) so the ModelBundle is only loaded
> once. The models hold state in memory; multiple workers each load their
> own copy, which wastes RAM.
