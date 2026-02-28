"""
CineMatch v2 — Hybrid Movie Recommender Web App
Loads sparse combined_matrix.pkl and computes similarity at query time.

Folder structure:
    app.py
    
    dataset/
        https://www.kaggle.com/datasets/ggtejas/tmdb-imdb-merged-movies-dataset
            
    models/
        combined_matrix.pkl
        movie_meta.pkl
        title_index.pkl
        titles_list.pkl
        
    templates/
        index.html
        readme.html


Run:
    pip install flask pandas numpy scikit-learn scipy
    python app.py

"""

import os, pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load artefacts once at startup ──────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

print("Loading model artefacts...")
# Ensure models exist
required_files = [
    "combined_matrix.pkl",
    "movie_meta.pkl",
    "title_index.pkl",
    "titles_list.pkl",
]
for f in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, f)):
        print(f"Error: {f} not found in {MODEL_DIR}")

combined = pickle.load(open(f"{MODEL_DIR}/combined_matrix.pkl", "rb"))
movie_meta = pickle.load(open(f"{MODEL_DIR}/movie_meta.pkl", "rb"))
title_index = pickle.load(open(f"{MODEL_DIR}/title_index.pkl", "rb"))
titles_list = pickle.load(open(f"{MODEL_DIR}/titles_list.pkl", "rb"))

meta_map = movie_meta.set_index("movie_name")
TMDB_IMG = "https://image.tmdb.org/t/p/w300"

print(f"✅ {len(titles_list):,} movies ready.")


# ── Helpers ─────────────────────────────────────────────────────
def get_meta(title):
    if title not in meta_map.index:
        return {}
    row = meta_map.loc[title]
    poster = str(row.get("poster_path", "") or "")
    return {
        "genres": str(row.get("genres", "") or "")[:60],
        "director": str(row.get("directors", "") or "")[:40],
        "cast": str(row.get("cast", "") or "")[:100],
        "overview": str(row.get("overview", "") or "")[:250],
        "vote_avg": round(float(row.get("vote_average", 0) or 0), 1),
        "vote_count": int(float(row.get("vote_count", 0) or 0)),
        "year": str(row.get("release_date", "") or "")[:4],
        "runtime": int(float(row.get("runtime", 0) or 0)),
        "tagline": str(row.get("tagline", "") or "")[:120],
        "poster": TMDB_IMG + poster if poster.startswith("/") else "",
        "keywords": str(row.get("keywords", "") or "")[:140],
    }


def do_recommend(movie_title, n=10):
    if movie_title not in title_index:
        return None, None
    idx = title_index[movie_title]
    query_vec = combined[idx]

    # Compute similarity scores
    scores = (combined @ query_vec.T).toarray().flatten()
    scores[idx] = -1  # Exclude the movie itself

    # Get top N
    top_idx = np.argpartition(scores, -n)[-n:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    results = []
    for i in top_idx:
        t = titles_list[i]
        m = get_meta(t)
        m["title"] = t
        m["score"] = round(float(scores[i]), 4)
        results.append(m)
    return results, get_meta(movie_title)


# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/readme.html")
def readme():
    return render_template("readme.html")


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").lower().strip()
    if len(q) < 2:
        return jsonify([])

    # Find matches
    matches = [t for t in titles_list if q in t.lower()][:30]
    out = []
    for t in matches:
        m = meta_map.loc[t] if t in meta_map.index else {}
        poster_path = str(m.get("poster_path", "") or "")

        # Construct full URL for the frontend
        poster_url = TMDB_IMG + poster_path if poster_path.startswith("/") else ""

        out.append(
            {
                "title": t,
                "year": str(m.get("release_date", "") or "")[:4],
                "genres": str(m.get("genres", "") or "")[:40],
                "poster": poster_url,  # Return full URL under 'poster' key
            }
        )
    return jsonify(out)


@app.route("/api/recommend")
def api_recommend():
    title = request.args.get("title", "").strip()
    n = min(int(request.args.get("n", 10)), 20)
    if not title:
        return jsonify({"error": "No title provided"}), 400

    results, input_meta = do_recommend(title, n=n)

    if results is None:
        close = [t for t in titles_list if title.lower() in t.lower()][:6]
        return jsonify({"error": "Not found", "suggestions": close}), 404

    input_meta["title"] = title
    return jsonify({"input": input_meta, "results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
