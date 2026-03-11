
#* ────────────────────────────────────────────────────────────────────────────────────────
#* MODULE 0 — MOUNT DRIVE & COPY ZIP
#* ────────────────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

import shutil, os, zipfile

ZIP_IN_DRIVE = "/content/drive/MyDrive/archiveTWO.zip"  #! https://www.kaggle.com/datasets/ggtejas/tmdb-imdb-merged-movies-dataset
shutil.copy(ZIP_IN_DRIVE, "/content/archiveTWO.zip")
os.makedirs("/content/data", exist_ok=True)
with zipfile.ZipFile("/content/archiveTWO.zip", "r") as z:
    z.extractall("/content/data")
print("✅ Unzipped:", os.listdir("/content/data"))


#. ───────────────────────────────────────────────────────────────────────────────────────
#. MODULE 1 — DEPENDENCIES
#. ───────────────────────────────────────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["pandas", "numpy", "scikit-learn", "scipy"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
print("✅ Dependencies ready.")


#? ───────────────────────────────────────────────────────────────────────────────────────
#? MODULE 2 — IMPORTS & CONFIG
#? ───────────────────────────────────────────────────────────────────────────────────────
import gc, re, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

DATA_DIR  = "/content/data"
MODEL_DIR = "/content/models"
os.makedirs(MODEL_DIR, exist_ok=True)

#? Signal weights                                                                        
W_GENRE    = 0.25
W_KEYWORDS = 0.20
W_PEOPLE   = 0.20
W_OVERVIEW = 0.15
W_NUMERIC  = 0.20

TOP_N_POPULAR = 50_000   #? only keepin' the 50K most-rated movies — plenty for recz    


#$ ───────────────────────────────────────────────────────────────────────────────────────
#$ MODULE 3 — LOAD & FILTER
#$ The dataset has 367K movies but most are
#$ obscure with 0 votes. We keep only the most
#$ relevant ones so the model is actually useful.
#$ ───────────────────────────────────────────────────────────────────────────────────────
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
df = pd.read_csv(os.path.join(DATA_DIR, csv_files[0]), low_memory=False)
df.columns = [c.strip().lower() for c in df.columns]

print(f"Raw rows: {len(df):,}")

# Keep released movies only
if "status" in df.columns:
    df = df[df["status"] == "Released"].copy()

df.dropna(subset=["title"], inplace=True)
df.drop_duplicates(subset=["title"], keep="first", inplace=True)

# Parse vote_count and keep top N most-voted movies
vc_col = next((c for c in ["vote_count", "numvotes"] if c in df.columns), None)
if vc_col:
    df[vc_col] = pd.to_numeric(df[vc_col], errors="coerce").fillna(0)
    df = df.nlargest(TOP_N_POPULAR, vc_col)

df.reset_index(drop=True, inplace=True)
df["movie_name"] = df["title"].str.strip()

print(f"✅ Working set: {len(df):,} movies (top {TOP_N_POPULAR:,} by votes)")

# Fill fields
TEXT_COLS = ["genres", "keywords", "directors", "writers", "cast", "overview"]
for col in TEXT_COLS:
    df[col] = df[col].fillna("") if col in df.columns else ""

NUM_COLS = ["vote_average", "vote_count", "revenue", "runtime",
            "averagerating", "numvotes", "popularity"]
for col in NUM_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
    else:
        df[col] = np.float32(0)

import psutil
print(f"💾 RAM: {psutil.Process().memory_info().rss/1e9:.1f} GB")


#$ ───────────────────────────────────────────────────────────────────────────────────────
#$ MODULE 4 — BUILD FEATURE VECTORS
#$ THE KEY INSIGHT:
#$   Store the vectors, NOT the similarity matrix.
#$   A 50K × 5K sparse matrix = ~50 MB.
#$   A 50K × 50K dense matrix = 10 GB.
#$   We compute similarity only at query time (instant).
#$ ───────────────────────────────────────────────────────────────────────────────────────

def clean(s):
    return re.sub(r"[,;|]+", " ", str(s)).lower().strip()

def people_str(row):
    parts = []
    if row["directors"]:
        d = clean(row["directors"]).replace(" ", "_")
        parts += [d, d, d]
    if row["writers"]:
        parts.append(clean(row["writers"]).replace(" ", "_"))
    if row["cast"]:
        parts += [n.strip().replace(" ", "_") for n in str(row["cast"]).split(",")[:5]]
    return " ".join(parts)

print("\n🔧 Building TF-IDF feature matrices (sparse)...")

tfidf_genre   = TfidfVectorizer(stop_words="english", max_features=300)
tfidf_kw      = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
tfidf_people  = TfidfVectorizer(stop_words="english", max_features=3000)
tfidf_ov      = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1,2))

M_genre   = normalize(tfidf_genre.fit_transform(df["genres"].apply(clean)).astype("float32"))
M_kw      = normalize(tfidf_kw.fit_transform(df["keywords"].apply(clean)).astype("float32"))
M_people  = normalize(tfidf_people.fit_transform(df.apply(people_str, axis=1)).astype("float32"))
M_ov      = normalize(tfidf_ov.fit_transform(df["overview"].apply(clean)).astype("float32"))

# Numeric features
num_cols_present = [c for c in NUM_COLS if c in df.columns]
num_arr = df[num_cols_present].values.astype("float32")
for i, col in enumerate(num_cols_present):
    if col in ["vote_count", "revenue", "numvotes", "popularity"]:
        num_arr[:, i] = np.log1p(num_arr[:, i])
num_arr = normalize(MinMaxScaler().fit_transform(num_arr).astype("float32"))
M_num = sp.csr_matrix(num_arr)

print(f"   Genre   : {M_genre.shape}  nnz={M_genre.nnz:,}")
print(f"   Keywords: {M_kw.shape}  nnz={M_kw.nnz:,}")
print(f"   People  : {M_people.shape}  nnz={M_people.nnz:,}")
print(f"   Overview: {M_ov.shape}  nnz={M_ov.nnz:,}")
print(f"   Numeric : {M_num.shape}")
print(f"💾 RAM: {psutil.Process().memory_info().rss/1e9:.1f} GB")


#$ ───────────────────────────────────────────────────────────────────────────────────────
#$ MODULE 5 — COMBINE INTO ONE WEIGHTED MATRIX
#$ Weighted stack → single sparse query matrix
#$ (50K × ~13K sparse, ~200 MB max)
#$ ───────────────────────────────────────────────────────────────────────────────────────
print("\n🔀 Building combined weighted feature matrix...")

# Stack all signals horizontally with their weights applied
# Result is one (N × total_features) sparse matrix per movie
combined = sp.hstack([
    M_genre   * W_GENRE,
    M_kw      * W_KEYWORDS,
    M_people  * W_PEOPLE,
    M_ov      * W_OVERVIEW,
    M_num     * W_NUMERIC,
], format="csr").astype("float32")

# Re-normalise rows so dot product = cosine similarity at query time
combined = normalize(combined)

print(f"   Combined matrix: {combined.shape}  nnz={combined.nnz:,}")
print(f"   Approx size: {combined.data.nbytes / 1e6:.0f} MB")
print(f"💾 RAM: {psutil.Process().memory_info().rss/1e9:.1f} GB")

# Free individual matrices
del M_genre, M_kw, M_people, M_ov, M_num, num_arr
gc.collect()
print(f"💾 RAM after cleanup: {psutil.Process().memory_info().rss/1e9:.1f} GB")


#$ ───────────────────────────────────────────────────────────────────────────────────────
#$ MODULE 6 — SAVE ARTEFACTS
#$ We save:
#$   combined.pkl  — the sparse feature matrix (~200 MB)
#$   movie_meta.pkl — title, genre, cast, poster etc.
#$   title_index.pkl — movie_name → row index lookup
#$ ───────────────────────────────────────────────────────────────────────────────────────
movie_meta = df[[
    "movie_name", "genres", "directors", "cast",
    "overview", "vote_average", "vote_count",
    "release_date", "runtime", "poster_path",
    "tagline", "keywords", "popularity"
]].copy().reset_index(drop=True)

title_index = {name: i for i, name in enumerate(df["movie_name"].tolist())}
titles_list = df["movie_name"].tolist()

print("\n💾 Saving artefacts...")
pickle.dump(combined,    open(f"{MODEL_DIR}/combined_matrix.pkl",  "wb"), protocol=4)
pickle.dump(movie_meta,  open(f"{MODEL_DIR}/movie_meta.pkl",       "wb"), protocol=4)
pickle.dump(title_index, open(f"{MODEL_DIR}/title_index.pkl",      "wb"), protocol=4)
pickle.dump(titles_list, open(f"{MODEL_DIR}/titles_list.pkl",      "wb"), protocol=4)

for f in os.listdir(MODEL_DIR):
    size = os.path.getsize(f"{MODEL_DIR}/{f}") / 1e6
    print(f"   ✅ {f:35s} {size:.1f} MB")

print(f"\n💾 Final RAM: {psutil.Process().memory_info().rss/1e9:.1f} GB")


#$ ──────────────────────────────────────────────────────────────────────────────────────
#$ MODULE 7 — RECOMMEND & SEARCH
#$ Query time: one sparse dot product (instant)
#$ ──────────────────────────────────────────────────────────────────────────────────────
meta_map = movie_meta.set_index("movie_name")

def recommend(movie_title, n=10):
    if movie_title not in title_index:
        close = [m for m in titles_list if movie_title.lower() in m.lower()]
        print(f"❌ '{movie_title}' not found.")
        if close:
            print("   Did you mean:\n   " + "\n   ".join(close[:8]))
        return

    idx = title_index[movie_title]
    query_vec = combined[idx]                        # (1 × features) sparse
    scores = (combined @ query_vec.T).toarray().flatten()  # (N,) dot products
    scores[idx] = -1                                  # exclude self

    top_idx = np.argpartition(scores, -n)[-n:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    rows = []
    for i in top_idx:
        title = titles_list[i]
        m = meta_map.loc[title] if title in meta_map.index else {}
        rows.append({
            "Movie":    title,
            "Score":    round(float(scores[i]), 4),
            "Genre":    str(m.get("genres",       ""))[:35],
            "Director": str(m.get("directors",    ""))[:25],
            "Rating":   round(float(m.get("vote_average", 0)), 1),
            "Year":     str(m.get("release_date", ""))[:4],
        })

    out = pd.DataFrame(rows)
    out.index = range(1, len(out) + 1)
    out.index.name = "Rank"
    print(f"\n🎬 Top {n} similar to '{movie_title}':\n")
    display(out)

def search(keyword):
    matches = [m for m in titles_list if keyword.lower() in m.lower()]
    print(f"🔍 '{keyword}' matches:")
    for i, m in enumerate(matches[:20], 1):
        print(f"   {i}. {m}")

print("\n✅ Done!  recommend('Inception')  |  search('dark')")


#$ ───────────────────────────────────────────────────────────────────────────────────
#$ MODULE 8 — INFERENCE  ✏️ TESTIN'
#$ ───────────────────────────────────────────────────────────────────────────────────
recommend("Inception")

#$ ───────────────────────────────────────────────────────────────────────────────────


# *################################# THE END SHUU ####################################
