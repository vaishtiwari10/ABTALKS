# Day 5 · Semantic Search — Finding Meaning, Not Just Keywords

## Objective
Build a working semantic search engine: embed a 20-sentence knowledge base as vectors, retrieve top-3 results for any query, and compare semantic search vs keyword search on the same queries.

## What's Inside

| File | Description |
|------|-------------|
| `day5_semantic_search.ipynb` | Complete notebook with dataset, embedding, search functions, 5 example queries, and analysis |

## Key Concepts
- **Semantic search** — embed query + documents into the same vector space, rank by cosine similarity
- **Keyword search** — count exact word overlaps between query and document (baseline)
- **Why semantic wins** — handles synonyms, paraphrasing, and question-style queries that keyword search misses entirely

## How to Run
```bash
pip install sentence-transformers
jupyter notebook day5_semantic_search.ipynb
```

## Connection to Day 4
Day 4 showed that embeddings capture meaning (similar sentences → similar vectors). Day 5 turns that insight into a **working search tool** — the exact mechanism behind RAG pipelines.
