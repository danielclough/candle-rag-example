use std::future::Future;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{self, Config as BertConfig, NomicBertModel};
use clap::Subcommand;
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::{PaddingParams, Tokenizer};

pub mod rag_pg;
pub mod rag_sqlite;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        #[cfg(target_os = "macos")]
        {
            Ok(Device::new_metal(0)?)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Device::Cpu)
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding model (nomic-bert)
// ---------------------------------------------------------------------------

pub struct EmbeddingModel {
    model: NomicBertModel,
    pub tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    pub fn load(dev: &Device) -> Result<Self> {
        let model_id = "nomic-ai/nomic-embed-text-v1.5";
        let revision = "main";
        println!("Loading embedding model: {model_id}");

        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = api.get("model.safetensors")?;

        let config: BertConfig = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, dev)? };
        let model = NomicBertModel::load(vb, &config)?;
        println!("Embedding model loaded.");
        Ok(Self {
            model,
            tokenizer,
            device: dev.clone(),
        })
    }

    /// Load from a local directory containing config.json, tokenizer.json,
    /// and model.safetensors (e.g. the tiny model in tiny-nomic).
    pub fn load_local(dir: &std::path::Path, dev: &Device) -> Result<Self> {
        let config: BertConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(E::msg)?;
        let weights_path = dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, dev)? };
        let model = NomicBertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device: dev.clone(),
        })
    }

    /// Embed a batch of texts, returns Vec<Vec<f32>> of normalized embeddings.
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Configure padding for batch processing.
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }

        let tokens = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(E::msg)?;

        let token_ids = tokens
            .iter()
            .map(|t| {
                let ids = t.get_ids().to_vec();
                Tensor::new(ids.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let attention_mask = tokens
            .iter()
            .map(|t| {
                let mask = t.get_attention_mask().to_vec();
                Tensor::new(mask.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;

        let hidden_states = self
            .model
            .forward(&token_ids, None, Some(&attention_mask))?;
        let embeddings = nomic_bert::mean_pooling(&hidden_states, &attention_mask)?;
        let embeddings = nomic_bert::l2_normalize(&embeddings)?;

        // Convert to Vec<Vec<f32>>
        let n = texts.len();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let emb: Vec<f32> = embeddings.get(i)?.to_dtype(DType::F32)?.to_vec1()?;
            result.push(emb);
        }
        Ok(result)
    }

    /// Embed a single text.
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let mut results = self.embed_batch(&texts)?;
        Ok(results.remove(0))
    }

    /// Count the actual tokens for a text using the model's tokenizer.
    pub fn token_count(&self, text: &str) -> Result<usize> {
        let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;
        Ok(encoding.get_ids().len())
    }
}

// ---------------------------------------------------------------------------
// CLI (shared between backends)
// ---------------------------------------------------------------------------

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Ingest CSV data: chunk, embed with nomic-bert, store in vector DB.
    Ingest {
        /// Path to the CSV file.
        #[arg(long, default_value = "tea_facts.csv")]
        csv_path: String,
        /// Specify postgres or sqlite.
        #[arg(long, default_value = "sqlite")]
        backend: String,
        /// Path to the SQLite database file.
        #[arg(long, default_value = "rag.db")]
        db_path: Option<String>,
    },
    /// Query: embed question, search vector DB, return similar documents.
    Query {
        /// The question to ask.
        #[arg(long)]
        prompt: String,
        /// Specify postgres or sqlite.
        #[arg(short = 'b', long, default_value = "sqlite")]
        backend: String,
        /// Path to the SQLite database file.
        #[arg(short = 'p', long, default_value = "rag.db")]
        db_path: Option<String>,
        /// Number of results to return.
        #[arg(short = 'n', long = "number", default_value_t = DEFAULT_RESULT_LIMIT)]
        number: usize,
    },
}

// ---------------------------------------------------------------------------
// Vector store trait & shared pipeline
// ---------------------------------------------------------------------------

const EMBED_BATCH_SIZE: usize = 8;
const DEFAULT_RESULT_LIMIT: usize = 5;

pub mod sql {
    pub mod pg {
        pub const CREATE_EXTENSION: &str = "CREATE EXTENSION IF NOT EXISTS vector";
        pub const CREATE_TABLE: &str = "\
            CREATE TABLE IF NOT EXISTS documents (
                id bigserial PRIMARY KEY,
                title text,
                url text,
                content text,
                tokens integer,
                embedding vector(768)
            )";
        pub const TRUNCATE: &str = "TRUNCATE TABLE documents";
        pub const INSERT: &str = "\
            INSERT INTO documents (title, url, content, tokens, embedding)
            VALUES ($1, $2, $3, $4, $5)";
        pub const SEARCH: &str = "\
            SELECT content, tokens FROM documents
            ORDER BY embedding <=> $1 LIMIT $2";
        pub const COUNT: &str = "SELECT COUNT(*) FROM documents";
    }

    pub mod sqlite {
        pub const CREATE_DOCUMENTS: &str = "\
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT,
                url TEXT,
                content TEXT,
                tokens INTEGER
            )";
        pub const CREATE_VEC: &str = "\
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
                document_id INTEGER PRIMARY KEY,
                embedding float[768]
            )";
        pub const CLEAR: &str = "DELETE FROM vec_documents; DELETE FROM documents;";
        pub const INSERT_DOC: &str = "\
            INSERT INTO documents (title, url, content, tokens)
            VALUES (?1, ?2, ?3, ?4)";
        pub const INSERT_VEC: &str = "\
            INSERT INTO vec_documents (document_id, embedding)
            VALUES (?1, ?2)";
        pub const SEARCH: &str = "\
            SELECT d.content, d.tokens
            FROM vec_documents v
            INNER JOIN documents d ON d.id = v.document_id
            WHERE v.embedding MATCH ?1
            AND k = ?2
            ORDER BY distance";
        pub const COUNT: &str = "SELECT COUNT(*) FROM documents";
    }
}

pub struct SearchResult {
    pub content: String,
    pub tokens: i32,
}

pub trait VectorStore: Send + Sync {
    fn clear(&self) -> impl Future<Output = Result<()>> + Send;
    fn insert_chunks(
        &self,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> impl Future<Output = Result<()>> + Send;
    fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> impl Future<Output = Result<Vec<SearchResult>>> + Send;
    fn count(&self) -> impl Future<Output = Result<i64>> + Send;
    fn name(&self) -> &str;
}

pub async fn run(store: &impl VectorStore, command: Command, dev: &Device) -> Result<()> {
    match command {
        Command::Ingest { csv_path, .. } => {
            let mut embedder = EmbeddingModel::load(dev)?;
            let chunks = load_and_chunk_csv(&csv_path, &embedder)?;

            let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
            for (i, batch) in chunks.chunks(EMBED_BATCH_SIZE).enumerate() {
                let texts: Vec<String> = batch
                    .iter()
                    .map(|c| format!("search_document: {} - {}", c.title, c.content))
                    .collect();
                let embs = embedder.embed_batch(&texts)?;
                all_embeddings.extend(embs);
                println!(
                    "Embedded batch {}/{} ({} chunks so far)",
                    i + 1,
                    chunks.len().div_ceil(EMBED_BATCH_SIZE),
                    all_embeddings.len()
                );
            }

            store.clear().await?;
            store.insert_chunks(&chunks, &all_embeddings).await?;
            println!(
                "Ingested {} chunks into {}.",
                store.count().await?,
                store.name()
            );
        }

        Command::Query { prompt, number, .. } => {
            let mut embedder = EmbeddingModel::load(dev)?;
            let query_text = format!("search_query: {prompt}");
            let query_emb = embedder.embed_one(&query_text)?;

            let docs: Vec<SearchResult> = store.search(&query_emb, number).await?;
            println!("\n--- Retrieved {} relevant documents ---\n", docs.len());
            for (i, doc) in docs.iter().enumerate() {
                println!("[{}] ({} tokens) {}\n", i + 1, doc.tokens, doc.content);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CSV loading & chunking
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
pub struct BlogPost {
    pub title: String,
    pub content: String,
    pub url: String,
}

pub struct Chunk {
    pub title: String,
    pub content: String,
    pub url: String,
    pub tokens: usize,
}

pub fn load_and_chunk_csv(path: &str, embedder: &EmbeddingModel) -> Result<Vec<Chunk>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut chunks = Vec::new();

    for result in reader.deserialize() {
        let post: BlogPost = result?;
        let words: Vec<&str> = post.content.split_whitespace().collect();
        let total_words = words.len();

        // Target ~340 tokens. 1 token ~ 3/4 word, so ideal_size ~ 256 words.
        let ideal_size = 256;
        let overlap = ideal_size / 5; // ~20% overlap
        let stride = ideal_size - overlap;

        if total_words <= ideal_size {
            let tokens = embedder.token_count(&post.content)?;
            chunks.push(Chunk {
                title: post.title.clone(),
                content: post.content.clone(),
                url: post.url.clone(),
                tokens,
            });
        } else {
            let mut start = 0;
            while start < total_words {
                let end = (start + ideal_size).min(total_words);
                let chunk_text: String = words[start..end].join(" ");
                let tokens = embedder.token_count(&chunk_text)?;
                if tokens > 0 {
                    chunks.push(Chunk {
                        title: post.title.clone(),
                        content: chunk_text,
                        url: post.url.clone(),
                        tokens,
                    });
                }
                if end == total_words {
                    break;
                }
                start += stride;
            }
        }
    }

    println!("Loaded {} chunks from {}", chunks.len(), path);
    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "  chunk {}: {} tokens ({})",
            i + 1,
            chunk.tokens,
            chunk.title
        );
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const TINY_MODEL_DIR: &str = "tiny-nomic";

    fn require_tiny_model() -> &'static Path {
        let dir = Path::new(TINY_MODEL_DIR);
        if !dir.join("model.safetensors").exists() {
            panic!("Tiny model fixtures not found. Run: cargo run --bin gen-tiny-model");
        }
        dir
    }

    #[test]
    fn smoke_load_tiny_model() {
        let dir = require_tiny_model();
        let _model = EmbeddingModel::load_local(dir, &Device::Cpu).unwrap();
    }

    #[test]
    fn smoke_embed_one() {
        let dir = require_tiny_model();
        let mut model = EmbeddingModel::load_local(dir, &Device::Cpu).unwrap();
        let emb = model.embed_one("hello world").unwrap();
        assert_eq!(emb.len(), 32, "tiny model has n_embd=32");
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "expected unit-norm embedding, got {norm}"
        );
    }

    #[test]
    fn smoke_embed_batch() {
        let dir = require_tiny_model();
        let mut model = EmbeddingModel::load_local(dir, &Device::Cpu).unwrap();
        let texts = vec!["a b c".into(), "x y z".into(), "1 2 3".into()];
        let embs = model.embed_batch(&texts).unwrap();
        assert_eq!(embs.len(), 3);
        for emb in &embs {
            assert_eq!(emb.len(), 32);
        }
        assert_ne!(embs[0], embs[1], "different inputs should differ");
    }

    #[test]
    fn smoke_token_count() {
        let dir = require_tiny_model();
        let model = EmbeddingModel::load_local(dir, &Device::Cpu).unwrap();
        let count = model.token_count("hello world").unwrap();
        // WordLevel tokenizer: [CLS] + "hello" + "world" + [SEP] = 4
        // (both words are UNK since only single letters are in vocab)
        assert!(count >= 2, "expected at least 2 tokens, got {count}");
    }

    #[test]
    fn smoke_deterministic() {
        let dir = require_tiny_model();
        let mut model = EmbeddingModel::load_local(dir, &Device::Cpu).unwrap();
        let a = model.embed_one("a b c").unwrap();
        let b = model.embed_one("a b c").unwrap();
        assert_eq!(a, b, "same input must produce identical embeddings");
    }

    #[test]
    fn smoke_embed_one_matches_batch() {
        let dir = require_tiny_model();
        let mut model = EmbeddingModel::load_local(dir, &Device::Cpu).unwrap();
        let single = model.embed_one("a b c").unwrap();
        let batch = model.embed_batch(&["a b c".into()]).unwrap();
        assert_eq!(single, batch[0]);
    }

    #[test]
    fn smoke_csv_chunking() {
        let dir = require_tiny_model();
        let model = EmbeddingModel::load_local(dir, &Device::Cpu).unwrap();

        let tmp = std::env::temp_dir().join("rag_rs_test_chunks.csv");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp).unwrap();
            writeln!(f, "title,content,url").unwrap();
            writeln!(f, "Tea,\"a b c d e f g\",https://example.com/tea").unwrap();
            writeln!(f, "Coffee,\"x y z\",https://example.com/coffee").unwrap();
        }

        let chunks = load_and_chunk_csv(tmp.to_str().unwrap(), &model).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].title, "Tea");
        assert_eq!(chunks[1].title, "Coffee");
        assert!(chunks[0].tokens > 0);
        assert!(chunks[1].tokens > 0);

        std::fs::remove_file(&tmp).ok();
    }
}
