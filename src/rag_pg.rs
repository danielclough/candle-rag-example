use crate::{Chunk, SearchResult, VectorStore, sql::pg};
use anyhow::Result;
use pgvector::Vector;
use sqlx::Row;
use sqlx::postgres::PgPoolOptions;

// ---------------------------------------------------------------------------
// pgvector VectorStore
// ---------------------------------------------------------------------------

pub struct PgStore {
    pub pool: sqlx::PgPool,
}

impl VectorStore for PgStore {
    async fn clear(&self) -> Result<()> {
        sqlx::query(pg::TRUNCATE).execute(&self.pool).await?;
        Ok(())
    }

    async fn insert_chunks(&self, chunks: &[Chunk], embeddings: &[Vec<f32>]) -> Result<()> {
        for (chunk, emb) in chunks.iter().zip(embeddings.iter()) {
            let vector = Vector::from(emb.clone());
            sqlx::query(pg::INSERT)
                .bind(&chunk.title)
                .bind(&chunk.url)
                .bind(&chunk.content)
                .bind(chunk.tokens as i32)
                .bind(vector)
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }

    async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let vector = Vector::from(query_embedding.to_vec());
        let rows = sqlx::query(pg::SEARCH)
            .bind(vector)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows
            .iter()
            .map(|r| SearchResult {
                content: r.get("content"),
                tokens: r.get("tokens"),
            })
            .collect())
    }

    async fn count(&self) -> Result<i64> {
        let (count,): (i64,) = sqlx::query_as(pg::COUNT).fetch_one(&self.pool).await?;
        Ok(count)
    }

    fn name(&self) -> &str {
        "pgvector"
    }
}

pub async fn connect_pg() -> Result<sqlx::PgPool> {
    let user = std::env::var("POSTGRES_USER")?;
    let password = std::env::var("POSTGRES_PASSWORD")?;
    let host = std::env::var("POSTGRES_HOST")?;
    let port = std::env::var("POSTGRES_PORT")?;
    let db = std::env::var("POSTGRES_DB")?;
    let url = format!("postgresql://{user}:{password}@{host}:{port}/{db}");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&url)
        .await?;

    sqlx::query(pg::CREATE_EXTENSION).execute(&pool).await?;
    sqlx::query(pg::CREATE_TABLE).execute(&pool).await?;

    Ok(pool)
}
