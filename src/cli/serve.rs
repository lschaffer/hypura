use std::path::PathBuf;
use std::sync::Arc;

use hypura::server::manager::ModelManager;
use hypura::server::registry::ModelRegistry;
use hypura::server::routes::{self, AppState};
use hypura::telemetry::metrics::TelemetryEmitter;

pub fn run(
    model: Option<&str>,
    host: &str,
    port: u16,
    context: u32,
    models_dir: Option<PathBuf>,
    ollama_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(model, host, port, context, models_dir, ollama_dir))
}

async fn run_async(
    model: Option<&str>,
    host: &str,
    port: u16,
    context: u32,
    models_dir: Option<PathBuf>,
    ollama_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let registry = ModelRegistry::new(models_dir, ollama_dir);
    let manager = ModelManager::new(registry, context, model)?;
    let telemetry = Arc::new(TelemetryEmitter::new(256));

    let state = Arc::new(AppState {
        manager: Arc::new(std::sync::Mutex::new(manager)),
        telemetry,
    });

    let app = routes::router(state.clone());
    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    println!();
    println!("Hypura Server running in dynamic multi-model mode");
    println!("  Endpoint: http://{bind_addr}");
    println!("  Ollama-compatible API: /api/tags, /api/show, /api/chat, /api/generate");

    let count = {
        let mgr = state.manager.lock().unwrap();
        mgr.registry.models.len()
    };
    println!("  Discovered {count} models (Local + Ollama).");
    if let Some(m) = model {
        println!("  Pre-warmed model: {m}");
    }
    println!();

    axum::serve(listener, app).await?;
    Ok(())
}
