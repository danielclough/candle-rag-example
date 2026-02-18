# Candle-RAG-example

Minimal pure Rust retrieval-augmented generation using [candle](https://github.com/huggingface/candle) for embeddings and vector search for retrieval - No Python, no APIs.

## How it works

1. **Embed** — text is chunked (~512 tokens) and embedded with [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
2. **Store** — 768-dim embeddings are stored in a vector database
3. **Search** — query text is embedded, nearest neighbors are returned by cosine distance

A single binary supports two storage backends, selected with `--backend`:

| Backend | Flag | Use case |
|---|---|---|
| SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) | `--backend sqlite` (default) | Local/portable, zero setup |
| PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) | `--backend pg` | Production, concurrent access |

## Prerequisites

- Rust 1.88+
- The nomic-embed-text-v1.5 model weights are downloaded automatically from Hugging Face on first run

For the PostgreSQL backend only:
- Docker (for the provided `docker-compose.yaml`)
- A `.env` file (see [PostgreSQL setup](#postgresql-setup))

## SQLite (default)

```bash
# Build
cargo build --release

# Ingest a CSV into a local SQLite database
cargo run --release -- ingest --csv-path tea_facts.csv

# Query
cargo run --release -- query --prompt "How do different cultures prepare and serve tea?"
cargo run --release -- query --prompt "What role do microorganisms play in tea production?"
cargo run --release -- query --prompt "How does climate affect tea quality and flavor?"
cargo run --release -- query --number 1 --prompt "How is matcha powder made?"
```

This creates a `rag.db` file in the current directory. Use `--db-path` to change it.

## PostgreSQL

```bash
# Start pgvector
docker compose up -d

# Ingest
cargo run --release -- ingest --backend pg --csv-path tea_facts.csv

# Query
cargo run --release -- query --backend pg --prompt "What makes some teas cost thousands of dollars?"
cargo run --release -- query --backend pg --prompt "Why does green tea taste bitter when you use boiling water?"
cargo run --release -- query --backend pg --prompt "How does tea help you relax without making you sleepy?"
cargo run --release -- query --number 1 --backend pg --prompt "What is L-theanine and what does it do?"
```

## CSV format

Both binaries expect a CSV with `title`, `content`, and `url` columns:

```csv
title,content,url
"Green Tea","Green tea is made from...","https://example.com/green-tea"
```

## CLI flags

```
--backend      Storage backend: "sqlite" (default) or "pg"
--cpu          Force CPU inference (default: auto-detect CUDA/Metal)
--csv-path     Path to CSV file (default: tea_facts.csv)
--prompt       Query text
--db-path      SQLite database path (default: rag.db)
--number       Number of queries to return
```

## Running tests

```bash
cargo test
```

Tests use a tiny (73KB) nomic-bert model with random weights committed in `tests/fixtures/tiny-nomic/`. This exercises the full candle inference path — tokenization, forward pass, mean pooling, L2 normalization — without downloading the real 550MB model.

Test coverage:

- **lib** — model loading, single/batch embedding, determinism, embed_one/batch consistency, CSV chunking with the real tokenizer
- **rag-sqlite** — sqlite-vec extension loading, insert/search roundtrip, KNN ordering, search limits, document/vector ID sync, full pipeline smoke test (tiny model → embed → sqlite-vec → nearest-neighbor query)

