use anyhow::Result;
use candle_rag_example::{
    Command, device,
    rag_pg::{PgStore, connect_pg},
    rag_sqlite::{SqliteStore, connect_sqlite},
    run,
};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[command(subcommand)]
    command: Command,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let dev = device(args.cpu)?;
    dotenvy::dotenv().ok();

    let (backend, db_path) = match &args.command {
        Command::Ingest {
            backend, db_path, ..
        } => (backend.as_str(), db_path.clone()),
        Command::Query {
            backend, db_path, ..
        } => (backend.as_str(), db_path.clone()),
    };

    match backend {
        "pg" => {
            let store = PgStore {
                pool: connect_pg().await?,
            };
            run(&store, args.command, &dev).await
        }
        _ => {
            let store = SqliteStore {
                conn: std::sync::Mutex::new(connect_sqlite(
                    &db_path.expect("default provided by clap"),
                )?),
            };
            run(&store, args.command, &dev).await
        }
    }
}
