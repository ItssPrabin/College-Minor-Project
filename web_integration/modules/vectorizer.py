

import ast
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K         = 10
SBERT_WEIGHT  = 0.60
W2V_WEIGHT    = 0.40
CHUNK_SIZE    = 200   # words per sliding-window chunk
CHUNK_STRIDE  = 150   # overlap between consecutive chunks


# ── Model loader (call once, reuse) ──────────────────────────────────────────

class ModelBundle:
    

    def __init__(
        self,
        w2v_model_path: str,
        sbert_model_name: str,
        w2v_vectors_path: str,
        sbert_vectors_path: str,
        w2v_metadata_path: str,
        sbert_metadata_path: str,
    ):
        print("[ModelBundle] Loading Word2Vec …")
        self.w2v_model: Word2Vec = Word2Vec.load(w2v_model_path)

        print("[ModelBundle] Loading SBERT …")
        self.sbert_model: SentenceTransformer = SentenceTransformer(sbert_model_name)

        print("[ModelBundle] Loading stored vectors …")
        self.w2v_matrix: np.ndarray   = np.load(w2v_vectors_path)
        self.sbert_matrix: np.ndarray = np.load(sbert_vectors_path)

        with open(w2v_metadata_path)   as f: w2v_meta   = json.load(f)
        with open(sbert_metadata_path) as f: sbert_meta = json.load(f)

        import pandas as pd
        self.w2v_df   = pd.DataFrame(w2v_meta)
        self.sbert_df = pd.DataFrame(sbert_meta)

        # Pre-build IDF from stored W2V corpus tokens
        self.idf = _build_idf(
            self.w2v_df["tokens_str"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x.split()
            ).tolist()
        )

        print(f"[ModelBundle] Ready — {self.w2v_matrix.shape[0]} resumes loaded.")


# ── IDF helper ────────────────────────────────────────────────────────────────

def _build_idf(corpus: list[list[str]]) -> dict[str, float]:
    N, counts = len(corpus), {}
    for doc in corpus:
        for t in set(doc):
            counts[t] = counts.get(t, 0) + 1
    return {t: np.log((N + 1) / (c + 1)) + 1 for t, c in counts.items()}


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [text]
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + CHUNK_SIZE]))
        if i + CHUNK_SIZE >= len(words):
            break
        i += CHUNK_STRIDE
    return chunks


def encode_sbert(text: str, sbert_model: SentenceTransformer) -> np.ndarray:
    chunks = _chunk_text(text)
    vecs   = sbert_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    vec    = vecs.mean(axis=0)
    norm   = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def encode_w2v(
    tokens: list[str],
    w2v_model: Word2Vec,
    idf: dict[str, float],
) -> np.ndarray:
    pairs = [(w2v_model.wv[t], idf.get(t, 1.0)) for t in tokens if t in w2v_model.wv]
    if not pairs:
        return np.zeros(w2v_model.vector_size)
    vs, ws = zip(*pairs)
    ws_arr = np.array(ws, dtype=float)
    ws_arr /= ws_arr.sum()
    return np.average(vs, axis=0, weights=ws_arr)


# ── Mode 1: rank new JD against stored corpus vectors ────────────────────────

def rank_against_stored(
    jd_sbert_text: str,
    jd_w2v_tokens: list[str],
    bundle: ModelBundle,
    top_k: int = TOP_K,
) -> list[dict]:
    
    jd_sbert_vec = encode_sbert(jd_sbert_text, bundle.sbert_model)
    jd_w2v_vec   = encode_w2v(jd_w2v_tokens,  bundle.w2v_model, bundle.idf)

    all_sbert = cosine_similarity(jd_sbert_vec.reshape(1, -1), bundle.sbert_matrix)[0]
    all_w2v   = cosine_similarity(jd_w2v_vec.reshape(1, -1),   bundle.w2v_matrix)[0]

    hybrid = SBERT_WEIGHT * all_sbert + W2V_WEIGHT * all_w2v

    # Align on resume_id (both metadata lists are ordered identically)
    results = []
    for i, row in bundle.sbert_df.iterrows():
        results.append({
            "resume_id":    row["resume_id"],
            "category":     row["category"],
            "sbert_score":  float(all_sbert[i]),
            "w2v_score":    float(all_w2v[i]),
            "hybrid_score": float(hybrid[i]),
            "preview":      str(row.get("resume_text", ""))[:300],
        })

    results.sort(key=lambda r: r["hybrid_score"], reverse=True)
    top = results[:top_k]
    for rank, r in enumerate(top, start=1):
        r["rank"] = rank
    return top


# ── Mode 2: live ranking of uploaded resumes ──────────────────────────────────

def rank_live_resumes(
    jd_sbert_text: str,
    jd_w2v_tokens: list[str],
    resumes: list[dict],           # [{ resume_id, filename, sbert_text, w2v_tokens }]
    bundle: ModelBundle,
    top_k: int = TOP_K,
) -> list[dict]:
   
    jd_sbert_vec = encode_sbert(jd_sbert_text, bundle.sbert_model)
    jd_w2v_vec   = encode_w2v(jd_w2v_tokens,  bundle.w2v_model, bundle.idf)

    results = []
    for r in resumes:
        if not r.get("sbert_text", "").strip():
            continue

        rv_sbert = encode_sbert(r["sbert_text"], bundle.sbert_model)
        rv_w2v   = encode_w2v(r.get("w2v_tokens", []), bundle.w2v_model, bundle.idf)

        s_score = float(cosine_similarity(jd_sbert_vec.reshape(1, -1), rv_sbert.reshape(1, -1))[0][0])
        w_score = float(cosine_similarity(jd_w2v_vec.reshape(1, -1),   rv_w2v.reshape(1, -1))[0][0])
        h_score = SBERT_WEIGHT * s_score + W2V_WEIGHT * w_score

        results.append({
            "resume_id":    r["resume_id"],
            "filename":     r.get("filename", f"resume_{r['resume_id']}"),
            "sbert_score":  s_score,
            "w2v_score":    w_score,
            "hybrid_score": h_score,
            "preview":      r["sbert_text"][:300],
        })

    results.sort(key=lambda r: r["hybrid_score"], reverse=True)
    top = results[:top_k]
    for rank, r in enumerate(top, start=1):
        r["rank"] = rank
    return top
