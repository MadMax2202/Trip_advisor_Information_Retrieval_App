# TripAdvisor Review-Based Information Retrieval System

An **Information Retrieval (IR) system for recommending similar places using user reviews**.

The system retrieves **restaurants, hotels, or attractions that are similar to a given place** by analyzing review text.

This project was developed for the **Information Retrieval & NLP course**.

**Authors**
- Maxim Grossmann
- Geoffroy-Junior Gankoue-Dzon

---

# Project Goal

The goal of this project is to build a system that can **recommend similar places based only on review text**.

The core hypothesis is:

> Similar experiences are described using similar language in reviews.

Therefore, if two places have similar reviews, they likely correspond to similar real-world experiences.

The system retrieves and ranks candidate places by comparing a **query text** to **review-based representations of places**.

---

# Dataset

The system uses TripAdvisor datasets.

### Reviews dataset


data/reviews83325.csv


Contains **340k+ reviews**.

Each review includes:
- review text
- rating
- place ID

---

### Place metadata


data/Tripadvisor.csv


Contains:

- place ID
- place type
- metadata used for evaluation

---

### Additional metadata

Located in:


Extra_data/


These files contain fine-grained categories used for evaluation, including:

- cuisine types
- attraction categories
- restaurant types
- dietary restrictions

---

# Preprocessing

Preprocessing is done in the Jupyter notebook:


notebook/Data_preprocessing.ipynb


Main preprocessing steps:

- filtering English reviews
- text normalization
- metadata cleaning
- dataset construction

Two datasets are produced.

---

## df_models

Review-level dataset used for document construction.

Columns:


review_id
place_id
review_text
score


Saved as:


data/processed_data/df_models.parquet


---

## df_eval

Place-level dataset used to compute ground-truth relevance.

Saved as:


data/processed_data/df_eval.parquet


---

# Document Construction (Selection Methods)

Each place can have **many reviews**.

The system builds **one document per place** using multiple selection strategies.

Selection methods are implemented in:


selection/methods/


Examples include:

### Baseline

Concatenate all reviews.

---

### Review-level filtering

Examples:


B_medium_k10
B_medium_k20


These select **reviews closest to median length**.

---

### Sentence-level filtering

Reviews are split into sentences and ranked with TF-IDF.

---

### TF-IDF word budgets

Select the most discriminative words.

Examples:


D_tfidf_80
D_tfidf_120
D_tfidf_160


These methods produce compact lexical representations of places.

---

### Semantic extraction

Uses embeddings to select representative sentences.

---

### LLM summarization

Optional summarization using **Ollama + Qwen**.

---

# Retrieval Models

Retrieval models are implemented in:


models/


The project evaluates several approaches.

---

## BM25

Classical lexical retrieval model.


models/bm25.py


---

## TF-IDF

Vector space retrieval.


models/tfidf.py


---

## Dense Retrieval

Uses sentence embeddings.

Models used:


all-MiniLM-L6-v2
multi-qa-MiniLM-L6-cos-v1
all-mpnet-base-v2


Implemented in:


models/dense.py


---

## Hybrid Retrieval

Combines lexical and semantic similarity.

Formula:


score = α * dense_score + (1 − α) * lexical_score


Implemented in:


models/hybrid.py


---

## Dense Reranking

Two-stage retrieval:

1. dense retrieval
2. cross-encoder reranking

Cross-encoder used:


cross-encoder/ms-marco-MiniLM-L-6-v2


Implemented in:


models/dense_rerank.py


---

# Evaluation

Evaluation uses metadata to define relevance.

Two relevance levels are defined.

---

## Level 1 (L1)

Places share the same **broad type**:

- restaurant
- hotel
- attraction

---

## Level 2 (L2)

Places share **fine-grained categories** such as:

- cuisine
- attraction subtype
- hotel price range

---

# Metrics

The system evaluates rankings using:

### Ranking metrics

- MRR (Mean Reciprocal Rank)
- nDCG@k

### Retrieval metrics

- Precision@k
- Recall@k
- Hit Rate@k

for:


k = 1, 5, 10, 20


### Ranking error

Rank position of the **first relevant result**.

---

# Running the Project

## 1 Install dependencies

bash
pip install -r requirements.txt
2 Run preprocessing

Open and run:

notebook/Data_preprocessing.ipynb

This generates:

data/processed_data/df_models.parquet
data/processed_data/df_eval.parquet
3 Build document corpora

Run:

python scripts/build_corpora.py

This script:

loads df_models.parquet

runs all selection methods

creates one document per place

stores results in cache

Output directory:

data/processed_data/selection_cache/
4 Run BM25 screening

Run:

python evaluation/run_experiment_bm25.py

Purpose:

Evaluate all selection methods using BM25 to identify the best document construction strategies.

Results are saved in:

results/bm25_selection_methods_full_metrics.csv
results/bm25_top5_selection_methods.csv
5 Run full model evaluation

After selecting the top 5 methods, run:

python evaluation/evaluate_all.py

Inside evaluate_all.py, specify the selected methods:

TOP5_METHOD_NAMES = [
    "D_tfidf_80",
    "D_tfidf_160",
    "D_tfidf_120",
    "B_medium_k10",
    "B_medium_k20"
]

The script evaluates:

BM25

TF-IDF

Dense models

Hybrid retrieval

Dense + reranking

Results are saved in:

results/models_top5_selection_methods.csv
Experimental Pipeline
Preprocessing (Jupyter Notebook)
        ↓
Build place documents (selection methods)
        ↓
BM25 screening of selection methods
        ↓
Select top 5 document construction methods
        ↓
Evaluate multiple retrieval models
        ↓
Compare ranking performance
Key Findings

Main observations:

TF-IDF word budgets produce strong lexical representations.

Dense MPNet achieves the best ranking quality.

Hybrid retrieval provides robust overall performance.

Cross-encoder reranking significantly improves strict relevance.

Future Improvements

Possible extensions:

tuning hybrid weight α

type-specific retrieval models

improved review quality scoring

FAISS indexing for large-scale dense retrieval

learning-to-rank approaches

License

This project was developed for academic purposes as part of the Information Retrieval & NLP course.