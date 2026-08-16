use std::path::PathBuf;

use hypura::server::registry::ModelRegistry;

pub fn run(
    models_dir: Option<PathBuf>,
    ollama_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let registry = ModelRegistry::new(models_dir, ollama_dir);

    println!();
    println!("Hypura Discovered Models");
    println!("─────────────────────────────────────────────────────────────────────────────");

    let mut list: Vec<_> = registry.models.values().collect();
    list.sort_by(|a, b| a.name.cmp(&b.name));
    list.dedup_by(|a, b| a.name == b.name);

    if list.is_empty() {
        println!("  No models found.");
        println!();
        println!("  Place .gguf files in the current directory or specify --models-dir.");
        println!("  Ollama models are automatically discovered from ~/.ollama/models.");
        println!();
        return Ok(());
    }

    println!(
        "  {:<36} {:<10} {:<12} {:<8} {:<8} {:<10}",
        "NAME", "SIZE", "FAMILY", "PARAMS", "QUANT", "SOURCE"
    );
    println!("  {}", "─".repeat(88));

    for model in list {
        let size_str = format_bytes(model.size);
        let param_str = format_params(model.metadata.parameter_count);
        let quant_str = model
            .metadata
            .quantization
            .as_deref()
            .unwrap_or("unknown");

        println!(
            "  {:<36} {:<10} {:<12} {:<8} {:<8} {:<10}",
            model.name, size_str, model.metadata.architecture, param_str, quant_str, model.source
        );
    }

    println!();
    println!("To start serving:");
    println!("  hypura serve                    (dynamic daemon mode)");
    println!("  hypura serve <MODEL_NAME>       (pre-warmed mode)");
    println!();

    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0} MB", bytes as f64 / (1 << 20) as f64)
    } else {
        format!("{bytes} B")
    }
}

fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1e6)
    } else {
        format!("{params}")
    }
}
