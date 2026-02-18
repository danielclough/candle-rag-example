use std::sync::Mutex;

use crate::{Chunk, SearchResult, VectorStore, sql::sqlite as sql};
use anyhow::Result;
use rusqlite::Connection;
use rusqlite::ffi::sqlite3_auto_extension;

// ---------------------------------------------------------------------------
// sqlite-vec VectorStore
// ---------------------------------------------------------------------------

pub struct SqliteStore {
    pub conn: Mutex<Connection>,
}

impl VectorStore for SqliteStore {
    async fn clear(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(sql::CLEAR)?;
        Ok(())
    }

    async fn insert_chunks(&self, chunks: &[Chunk], embeddings: &[Vec<f32>]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let mut doc_stmt = conn.prepare(sql::INSERT_DOC)?;
        let mut vec_stmt = conn.prepare(sql::INSERT_VEC)?;

        for (chunk, emb) in chunks.iter().zip(embeddings.iter()) {
            doc_stmt.execute(rusqlite::params![
                &chunk.title,
                &chunk.url,
                &chunk.content,
                chunk.tokens as i32,
            ])?;
            let doc_id = conn.last_insert_rowid();
            let emb_bytes: &[u8] = bytemuck::cast_slice(emb.as_slice());
            vec_stmt.execute(rusqlite::params![doc_id, emb_bytes])?;
        }
        Ok(())
    }

    async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let conn = self.conn.lock().unwrap();
        let query_bytes: &[u8] = bytemuck::cast_slice(query_embedding);
        let mut stmt = conn.prepare(sql::SEARCH)?;

        let results = stmt
            .query_map(rusqlite::params![query_bytes, limit], |row| {
                Ok(SearchResult {
                    content: row.get(0)?,
                    tokens: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(results)
    }

    async fn count(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(sql::COUNT, [], |row| row.get(0))?;
        Ok(count)
    }

    fn name(&self) -> &str {
        "sqlite-vec"
    }
}

pub fn connect_sqlite(path: &str) -> Result<Connection> {
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut std::ffi::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> i32,
        >(sqlite_vec::sqlite3_vec_init as *const ())));
    }

    let conn = Connection::open(path)?;

    conn.execute_batch(sql::CREATE_DOCUMENTS)?;
    conn.execute_batch(sql::CREATE_VEC)?;

    Ok(conn)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::device;

    use super::*;

    /// Make a 768-dim vector that is zero everywhere except at `hot_index`.
    fn sparse_vec(hot_index: usize, value: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; 768];
        v[hot_index] = value;
        v
    }

    #[test]
    fn sqlite_vec_extension_loads() {
        let conn = connect_sqlite(":memory:").unwrap();
        let version: String = conn
            .query_row("SELECT vec_version()", [], |r| r.get(0))
            .unwrap();
        assert!(
            version.starts_with('v'),
            "expected version string, got: {version}"
        );
    }

    #[tokio::test]
    async fn insert_and_search_roundtrip() {
        let conn = connect_sqlite(":memory:").unwrap();
        let store = SqliteStore {
            conn: Mutex::new(conn),
        };

        let chunks = vec![
            Chunk {
                title: "doc_a".into(),
                content: "alpha content".into(),
                url: "a".into(),
                tokens: 2,
            },
            Chunk {
                title: "doc_b".into(),
                content: "beta content".into(),
                url: "b".into(),
                tokens: 2,
            },
            Chunk {
                title: "doc_c".into(),
                content: "gamma content".into(),
                url: "c".into(),
                tokens: 2,
            },
        ];

        let embeddings = vec![sparse_vec(0, 1.0), sparse_vec(1, 1.0), sparse_vec(2, 1.0)];

        store.insert_chunks(&chunks, &embeddings).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 3);

        let results = store.search(&sparse_vec(0, 1.0), 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].content, "alpha content",
            "nearest neighbor should be doc_a"
        );
    }

    #[tokio::test]
    async fn search_ordering_respects_distance() {
        let conn = connect_sqlite(":memory:").unwrap();
        let store = SqliteStore {
            conn: Mutex::new(conn),
        };

        let chunks = vec![
            Chunk {
                title: "far".into(),
                content: "far away".into(),
                url: "f".into(),
                tokens: 2,
            },
            Chunk {
                title: "near".into(),
                content: "very near".into(),
                url: "n".into(),
                tokens: 2,
            },
            Chunk {
                title: "mid".into(),
                content: "in between".into(),
                url: "m".into(),
                tokens: 2,
            },
        ];

        let mut near = vec![0.0f32; 768];
        near[0] = 0.9;
        near[1] = 0.1;

        let mut mid = vec![0.0f32; 768];
        mid[0] = 0.5;
        mid[1] = 0.5;

        let far = sparse_vec(1, 1.0);
        let embeddings = vec![far, near, mid];

        store.insert_chunks(&chunks, &embeddings).await.unwrap();

        let results = store.search(&sparse_vec(0, 1.0), 3).await.unwrap();
        assert_eq!(results[0].content, "very near");
        assert_eq!(results[1].content, "in between");
        assert_eq!(results[2].content, "far away");
    }

    #[tokio::test]
    async fn search_limit_is_respected() {
        let conn = connect_sqlite(":memory:").unwrap();
        let store = SqliteStore {
            conn: Mutex::new(conn),
        };

        let chunks: Vec<Chunk> = (0..10)
            .map(|i| Chunk {
                title: format!("doc_{i}"),
                content: format!("content {i}"),
                url: format!("u{i}"),
                tokens: 1,
            })
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..10).map(|i| sparse_vec(i, 1.0)).collect();

        store.insert_chunks(&chunks, &embeddings).await.unwrap();

        let results = store.search(&sparse_vec(0, 1.0), 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn document_ids_stay_in_sync() {
        let conn = connect_sqlite(":memory:").unwrap();
        let store = SqliteStore {
            conn: Mutex::new(conn),
        };

        let chunks = vec![
            Chunk {
                title: "first".into(),
                content: "aaa".into(),
                url: "1".into(),
                tokens: 1,
            },
            Chunk {
                title: "second".into(),
                content: "bbb".into(),
                url: "2".into(),
                tokens: 1,
            },
        ];
        let embeddings = vec![sparse_vec(0, 1.0), sparse_vec(1, 1.0)];

        store.insert_chunks(&chunks, &embeddings).await.unwrap();

        let orphans: i64 = store
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM vec_documents v
                 LEFT JOIN documents d ON d.id = v.document_id
                 WHERE d.id IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(orphans, 0, "no vec_documents rows should be orphaned");
    }

    /// In-memory DB with 32-dim embeddings (matches the tiny test model).
    fn connect_32() -> Connection {
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::ffi::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> i32,
            >(
                sqlite_vec::sqlite3_vec_init as *const ()
            )));
        }
        let conn = Connection::open(":memory:").unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT,
                url TEXT,
                content TEXT,
                tokens INTEGER
            )",
        )
        .unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
                document_id INTEGER PRIMARY KEY,
                embedding float[32]
            )",
        )
        .unwrap();
        conn
    }

    #[tokio::test]
    async fn smoke_full_sqlite_pipeline() {
        use crate::EmbeddingModel;
        use std::path::Path;

        let dir = Path::new("tiny-nomic");
        if !dir.join("model.safetensors").exists() {
            panic!("Tiny model fixtures not found. Run: cargo run --bin gen-tiny-model");
        }

        let dev = device(true).unwrap();
        let mut model = EmbeddingModel::load_local(dir, &dev).unwrap();

        let chunks = vec![
            Chunk {
                title: "tea".into(),
                content: "a b c d".into(),
                url: "t".into(),
                tokens: 4,
            },
            Chunk {
                title: "coffee".into(),
                content: "x y z".into(),
                url: "c".into(),
                tokens: 3,
            },
            Chunk {
                title: "water".into(),
                content: "1 2 3".into(),
                url: "w".into(),
                tokens: 3,
            },
        ];

        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = model.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 32);

        let conn = connect_32();
        let store = SqliteStore {
            conn: Mutex::new(conn),
        };
        store.insert_chunks(&chunks, &embeddings).await.unwrap();

        let query_emb = model.embed_one("a b c d").unwrap();
        let results = store.search(&query_emb, 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].content, "a b c d",
            "nearest neighbor should be itself"
        );
    }
}
