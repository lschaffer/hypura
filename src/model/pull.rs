use std::path::{Path, PathBuf};
use std::time::Duration;
use anyhow::{Context, Result};
use tokio::sync::mpsc;
use crate::server::ollama_types::PullResponseChunk;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullSpec {
    pub alias: String,
    pub filename: String,
    pub download_url: String,
    pub expected_size_bytes: Option<u64>,
}

/// Detect default models directory with priority:
/// 1. `HYPURA_MODELS_DIR` environment variable
/// 2. `/Volumes/RealtekData/models` if exists
/// 3. `~/.hypura/models` if exists
/// 4. `./models` in current working directory
pub fn detect_default_models_dir() -> PathBuf {
    if let Ok(env_val) = std::env::var("HYPURA_MODELS_DIR") {
        let p = PathBuf::from(env_val);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }

    let realtek = PathBuf::from("/Volumes/RealtekData/models");
    if realtek.is_dir() {
        return realtek;
    }

    if let Ok(home) = std::env::var("HOME") {
        let hypura_dir = PathBuf::from(home).join(".hypura/models");
        if hypura_dir.is_dir() {
            return hypura_dir;
        }
    }

    PathBuf::from("models")
}

/// Resolve a user-supplied model query (short alias, HuggingFace repo/tag, or URL)
/// into a concrete `PullSpec`.
pub fn resolve_pull_spec(query: &str) -> Result<PullSpec> {
    let q = query.trim();
    if q.is_empty() {
        anyhow::bail!("Model name or URL cannot be empty");
    }

    // 1. Direct HTTP/HTTPS URL
    if q.starts_with("http://") || q.starts_with("https://") {
        let clean_url = q.split('?').next().unwrap_or(q);
        let raw_filename = clean_url.split('/').next_back().unwrap_or("model.gguf");
        let filename = if raw_filename.is_empty() {
            "model.gguf"
        } else {
            raw_filename
        }.to_string();

        let alias = filename
            .strip_suffix(".gguf")
            .unwrap_or(&filename)
            .to_lowercase();
        return Ok(PullSpec {
            alias,
            filename,
            download_url: q.to_string(),
            expected_size_bytes: None,
        });
    }

    let q_lower = q.to_lowercase();

    // 2. Predefined high-performance aliases for 2026 / 2025 models
    match q_lower.as_str() {
        // Gemma 4 26B (Mixture / Active-4B routing)
        "gemma-4:26b" | "gemma4:26b" | "gemma-4-26b" | "gemma4-26b" | "gemma-4-26b-a4b" | "gemma4:26b-a4b" => {
            return Ok(PullSpec {
                alias: "gemma-4-26b-a4b".into(),
                filename: "google_gemma-4-26B-A4B-it-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF/resolve/main/google_gemma-4-26B-A4B-it-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(15_800_000_000),
            });
        }
        // Gemma 4 31B (Dense flagship)
        "gemma-4:31b" | "gemma4:31b" | "gemma-4-31b" | "gemma4-31b" => {
            return Ok(PullSpec {
                alias: "gemma-4-31b".into(),
                filename: "google_gemma-4-31B-it-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/bartowski/google_gemma-4-31B-it-GGUF/resolve/main/google_gemma-4-31B-it-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(19_200_000_000),
            });
        }
        // Qwen 3.8 27B
        "qwen3.8:27b" | "qwen3.8-27b" | "qwen-3.8:27b" | "qwen-3.8-27b" => {
            return Ok(PullSpec {
                alias: "qwen3.8-27b".into(),
                filename: "Qwen3.8-27B-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(16_200_000_000),
            });
        }
        // Qwen 2.5 Coder 32B Instruct
        "qwen2.5-coder:32b" | "qwen2.5-coder-32b" | "qwen-coder:32b" | "qwen-2.5-coder:32b" => {
            return Ok(PullSpec {
                alias: "qwen2.5-coder-32b-q3km".into(),
                filename: "qwen2.5-coder-32b-instruct-q3_k_m.gguf".into(),
                download_url: "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q3_k_m.gguf".into(),
                expected_size_bytes: Some(15_400_000_000),
            });
        }
        // Mistral Small 24B Instruct
        "mistral-small:24b" | "mistral-small-24b" | "mistral-small-2501" | "mistral-24b" => {
            return Ok(PullSpec {
                alias: "mistral-small-24b-q4km".into(),
                filename: "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF/resolve/main/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(14_300_000_000),
            });
        }
        // DeepSeek Coder V2 Lite MoE (16B total, 2.4B active)
        "deepseek-coder-v2-lite" | "deepseek-coder:16b" | "deepseek-coder-lite" => {
            return Ok(PullSpec {
                alias: "deepseek-coder-v2-lite-q4km".into(),
                filename: "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf".into(),
                download_url: "https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf".into(),
                expected_size_bytes: Some(9_500_000_000),
            });
        }
        _ => {}
    }

    // 3. Hugging Face pattern: owner/repo[:file_or_quant] or hf.co/owner/repo[:file_or_quant]
    let stripped = q
        .strip_prefix("hf.co/")
        .or_else(|| q.strip_prefix("huggingface.co/"))
        .unwrap_or(q);

    if stripped.contains('/') {
        let (repo_part, tag_part) = match stripped.split_once(':') {
            Some((r, t)) => (r, Some(t)),
            None => (stripped, None),
        };

        let parts: Vec<&str> = repo_part.split('/').collect();
        if parts.len() == 2 {
            let owner = parts[0];
            let repo = parts[1];
            let filename = build_hf_fallback_filename(repo, tag_part);

            let download_url = format!(
                "https://huggingface.co/{}/{}/resolve/main/{}",
                owner, repo, filename
            );
            let alias = repo.to_lowercase();
            return Ok(PullSpec {
                alias,
                filename,
                download_url,
                expected_size_bytes: None,
            });
        }
    }

    anyhow::bail!(
        "Unknown model alias or repository '{query}'. Supported aliases: gemma-4:26b, gemma-4:31b, qwen3.8:27b, qwen2.5-coder:32b, mistral-small:24b, deepseek-coder-v2-lite, or specify HuggingFace 'owner/repo:quant' / Ollama 'model:tag'."
    )
}

/// Strip trailing `-GGUF` / `_GGUF` (case-insensitive) from a Hugging Face repo name
/// before appending `-<quant>.gguf` so `unsloth/Qwen3.8-27B-GGUF:Q4_K_M` resolves to
/// `Qwen3.8-27B-Q4_K_M.gguf` instead of `Qwen3.8-27B-GGUF-Q4_K_M.gguf`.
fn build_hf_fallback_filename(repo: &str, tag_part: Option<&str>) -> String {
    if let Some(tag) = tag_part {
        if tag.to_lowercase().ends_with(".gguf") {
            return tag.to_string();
        }
    }
    let base = strip_gguf_suffix(repo);
    let quant = tag_part.unwrap_or("Q4_K_M");
    format!("{}-{}.gguf", base, quant)
}

fn strip_gguf_suffix(name: &str) -> &str {
    for suffix in ["-GGUF", "-gguf", "_GGUF", "_gguf", "-Gguf"] {
        if let Some(stripped) = name.strip_suffix(suffix) {
            return stripped;
        }
    }
    name
}

/// Async resolver that supports:
/// 1. Direct HTTP/HTTPS URLs
/// 2. Built-in short aliases
/// 3. Hugging Face GGUF repos (with live tree API filename matching)
/// 4. Any Ollama registry model (`model`, `model:tag`, `namespace/model:tag`, `ollama.com/...`)
pub async fn resolve_pull_spec_async(query: &str) -> Result<PullSpec> {
    let q = query.trim();
    if q.is_empty() {
        anyhow::bail!("Model name or URL cannot be empty");
    }

    // 1. Direct URL
    if q.starts_with("http://") || q.starts_with("https://") {
        return resolve_pull_spec(q);
    }

    // 2. Explicit Ollama registry prefix
    if let Some(ollama_ref) = q
        .strip_prefix("ollama.com/")
        .or_else(|| q.strip_prefix("registry.ollama.ai/"))
    {
        return resolve_ollama_registry(ollama_ref).await;
    }

    // 3. Explicit Hugging Face prefix
    if let Some(hf_ref) = q
        .strip_prefix("hf.co/")
        .or_else(|| q.strip_prefix("huggingface.co/"))
    {
        let (repo_part, tag_part) = match hf_ref.split_once(':') {
            Some((r, t)) => (r, Some(t)),
            None => (hf_ref, None),
        };
        let parts: Vec<&str> = repo_part.split('/').collect();
        if parts.len() == 2 {
            return resolve_huggingface_async(parts[0], parts[1], tag_part).await;
        }
        anyhow::bail!("Invalid Hugging Face reference '{query}'. Expected 'hf.co/owner/repo[:quant]'.");
    }

    // 4. Built-in aliases (if no '/' and matches sync alias table)
    if !q.contains('/') {
        if let Ok(spec) = resolve_pull_spec(q) {
            return Ok(spec);
        }
        // Bare model name (e.g. `qwen2.5:3b`, `llama3.2`, `inkling-2-mini:26b-a5b-nvfp4`) -> Ollama registry
        return resolve_ollama_registry(q).await;
    }

    // 5. Contains '/' (`owner/repo[:tag]`): could be Hugging Face OR Ollama community namespace
    let (repo_part, tag_part) = match q.split_once(':') {
        Some((r, t)) => (r, Some(t)),
        None => (q, None),
    };
    let parts: Vec<&str> = repo_part.split('/').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid model reference '{query}'.");
    }
    let (owner, repo) = (parts[0], parts[1]);

    let looks_like_hf = repo.to_lowercase().contains("gguf")
        || tag_part.map(|t| t.to_lowercase().ends_with(".gguf")).unwrap_or(false)
        || tag_part.is_none();

    if looks_like_hf {
        // Try Hugging Face API first; if repo not found on HF, fall back to Ollama registry
        match resolve_huggingface_async(owner, repo, tag_part).await {
            Ok(spec) => Ok(spec),
            Err(hf_err) => resolve_ollama_registry(q).await.map_err(|_| hf_err),
        }
    } else {
        // Try Ollama community namespace first (e.g. `lklimek/gemma4-text:26b-a4b-it-qat`), then HF
        match resolve_ollama_registry(q).await {
            Ok(spec) => Ok(spec),
            Err(ollama_err) => resolve_huggingface_async(owner, repo, tag_part)
                .await
                .map_err(|_| ollama_err),
        }
    }
}

/// Resolve a Hugging Face repository using the HF tree API to pick the exact `.gguf` file,
/// falling back to the constructed filename if the API is unreachable.
async fn resolve_huggingface_async(
    owner: &str,
    repo: &str,
    tag_part: Option<&str>,
) -> Result<PullSpec> {
    // If user already gave an exact .gguf filename, no API lookup needed
    if let Some(tag) = tag_part {
        if tag.to_lowercase().ends_with(".gguf") {
            let download_url = format!(
                "https://huggingface.co/{}/{}/resolve/main/{}",
                owner, repo, tag
            );
            return Ok(PullSpec {
                alias: repo.to_lowercase(),
                filename: tag.to_string(),
                download_url,
                expected_size_bytes: None,
            });
        }
    }

    let api_url = format!(
        "https://huggingface.co/api/models/{}/{}/tree/main",
        owner, repo
    );
    let output = tokio::process::Command::new("curl")
        .arg("-sS")
        .arg("-f")
        .arg("-L")
        .arg("--max-time")
        .arg("15")
        .arg(&api_url)
        .output()
        .await;

    if let Ok(out) = output {
        if out.status.success() {
            let body = String::from_utf8_lossy(&out.stdout);
            let quant = tag_part.unwrap_or("Q4_K_M");
            if let Some((path, size)) = find_best_hf_gguf_file(&body, quant) {
                let filename = path.rsplit('/').next().unwrap_or(&path).to_string();
                let download_url = format!(
                    "https://huggingface.co/{}/{}/resolve/main/{}",
                    owner, repo, path
                );
                return Ok(PullSpec {
                    alias: repo.to_lowercase(),
                    filename,
                    download_url,
                    expected_size_bytes: size,
                });
            }
            anyhow::bail!(
                "No .gguf file matching '{}' found in Hugging Face repo '{}/{}'. Try specifying the exact .gguf filename as '{}/{}:<filename>.gguf'.",
                quant, owner, repo, owner, repo
            );
        }
    }

    // Fallback to constructed filename (with -GGUF suffix stripped)
    let filename = build_hf_fallback_filename(repo, tag_part);
    let download_url = format!(
        "https://huggingface.co/{}/{}/resolve/main/{}",
        owner, repo, filename
    );
    Ok(PullSpec {
        alias: repo.to_lowercase(),
        filename,
        download_url,
        expected_size_bytes: None,
    })
}

/// Given JSON from `https://huggingface.co/api/models/{owner}/{repo}/tree/main`,
/// find the best matching `.gguf` file for the requested quant (e.g. `Q4_K_M`).
fn find_best_hf_gguf_file(tree_json: &str, quant: &str) -> Option<(String, Option<u64>)> {
    let entries: serde_json::Value = serde_json::from_str(tree_json).ok()?;
    let arr = entries.as_array()?;
    let quant_lower = quant.to_lowercase();

    let mut gguf_files: Vec<(String, Option<u64>)> = Vec::new();
    for entry in arr {
        if entry.get("type").and_then(|t| t.as_str()) != Some("file") {
            continue;
        }
        let path = match entry.get("path").and_then(|p| p.as_str()) {
            Some(p) if p.to_lowercase().ends_with(".gguf") => p.to_string(),
            _ => continue,
        };
        // Skip mmproj vision projector files and non-first shards
        let lower = path.to_lowercase();
        if lower.contains("mmproj") {
            continue;
        }
        let stem = path.strip_suffix(".gguf").unwrap_or(&path);
        if crate::server::registry::is_non_first_shard(stem) {
            continue;
        }
        let size = entry.get("size").and_then(|s| s.as_u64());
        gguf_files.push((path, size));
    }

    // 1. Exact quant suffix match (e.g. `-q4_k_m.gguf` or `.q4_k_m.gguf`),
    //    preferring the shortest filename so `Q4_K_M` picks `Model-Q4_K_M.gguf`
    //    over `Model-UD-Q4_K_M.gguf` unless `UD-Q4_K_M` was explicitly requested.
    let mut exact_matches: Vec<&(String, Option<u64>)> = gguf_files
        .iter()
        .filter(|(p, _)| {
            let lower = p.to_lowercase();
            lower.ends_with(&format!("-{}.gguf", quant_lower))
                || lower.ends_with(&format!(".{}.gguf", quant_lower))
                || lower.ends_with(&format!("_{}.gguf", quant_lower))
        })
        .collect();
    exact_matches.sort_by_key(|(p, _)| p.len());
    if let Some(found) = exact_matches.first() {
        return Some((*found).clone());
    }

    // 2. Substring match on quant (e.g. `UD-Q4_K_M`, `Q4_K_M-00001-of-...`)
    let mut sub_matches: Vec<&(String, Option<u64>)> = gguf_files
        .iter()
        .filter(|(p, _)| p.to_lowercase().contains(&quant_lower))
        .collect();
    sub_matches.sort_by_key(|(p, _)| p.len());
    if let Some(found) = sub_matches.first() {
        return Some((*found).clone());
    }

    None
}

/// Parse an Ollama reference (`model`, `model:tag`, `namespace/model:tag`) into
/// `(namespace, model, tag)`.
fn parse_ollama_ref(reference: &str) -> (String, String, String) {
    let trimmed = reference.trim().trim_start_matches("library/");
    let (name_part, tag) = match reference.trim().split_once(':') {
        Some((n, t)) if !t.is_empty() => (n, t.to_string()),
        _ => (reference.trim(), "latest".to_string()),
    };

    if let Some((ns, model)) = name_part.split_once('/') {
        (ns.to_string(), model.to_string(), tag)
    } else {
        let model = trimmed.split(':').next().unwrap_or(name_part);
        ("library".to_string(), model.to_string(), tag)
    }
}

/// Resolve any Ollama registry model (`https://registry.ollama.ai/v2/{ns}/{model}/manifests/{tag}`)
/// into a concrete `PullSpec` pointing to its GGUF blob.
async fn resolve_ollama_registry(query: &str) -> Result<PullSpec> {
    let (namespace, model, tag) = parse_ollama_ref(query);
    let manifest_url = format!(
        "https://registry.ollama.ai/v2/{}/{}/manifests/{}",
        namespace, model, tag
    );

    let output = tokio::process::Command::new("curl")
        .arg("-sS")
        .arg("-L")
        .arg("--max-time")
        .arg("20")
        .arg("-H")
        .arg("Accept: application/vnd.docker.distribution.manifest.v2+json")
        .arg("-w")
        .arg("\n%{http_code}")
        .arg(&manifest_url)
        .output()
        .await
        .with_context(|| format!("Failed to query Ollama registry for '{query}'"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to connect to Ollama registry for '{query}': {}", stderr.trim());
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let (body, status_str) = raw.rsplit_once('\n').unwrap_or((&raw, "000"));
    let http_status: u16 = status_str.trim().parse().unwrap_or(0);

    if http_status == 404 {
        anyhow::bail!(
            "Model '{}' not found in Ollama registry ({}). Check the model/tag name or specify a Hugging Face repo 'owner/repo:quant'.",
            query, manifest_url
        );
    }
    if http_status < 200 || http_status >= 300 {
        anyhow::bail!(
            "Ollama registry returned HTTP {} for '{}'.",
            http_status, query
        );
    }

    parse_ollama_manifest_json(body, query, &namespace, &model, &tag)
}

/// Parse Ollama registry manifest JSON into a `PullSpec` for the GGUF model blob.
fn parse_ollama_manifest_json(
    manifest_json: &str,
    query: &str,
    namespace: &str,
    model: &str,
    tag: &str,
) -> Result<PullSpec> {
    let json: serde_json::Value = serde_json::from_str(manifest_json)
        .with_context(|| format!("Invalid manifest JSON from Ollama registry for '{query}'"))?;

    let layers = json
        .get("layers")
        .and_then(|l| l.as_array())
        .ok_or_else(|| anyhow::anyhow!("Ollama manifest for '{}' has no layers", query))?;

    let mut has_per_tensor_layers = false;
    for layer in layers {
        let media_type = layer
            .get("mediaType")
            .and_then(|m| m.as_str())
            .unwrap_or("");
        if media_type == "application/vnd.ollama.image.model" {
            let digest = layer
                .get("digest")
                .and_then(|d| d.as_str())
                .ok_or_else(|| anyhow::anyhow!("Ollama model layer missing digest"))?;
            let size = layer.get("size").and_then(|s| s.as_u64());

            let alias = if namespace == "library" {
                format!("{}-{}", model, tag)
            } else {
                format!("{}_{}-{}", namespace, model, tag)
            }
            .replace(['/', ':'], "-")
            .to_lowercase();

            let filename = format!("{}.gguf", alias);
            let download_url = format!(
                "https://registry.ollama.ai/v2/{}/{}/blobs/{}",
                namespace, model, digest
            );

            return Ok(PullSpec {
                alias,
                filename,
                download_url,
                expected_size_bytes: size,
            });
        }
        if media_type == "application/vnd.ollama.image.tensor" {
            has_per_tensor_layers = true;
        }
    }

    if has_per_tensor_layers {
        anyhow::bail!(
            "Model '{}' in the Ollama registry uses Ollama's per-tensor format (not GGUF). Hypura requires GGUF models — pull a GGUF quant from Hugging Face instead (e.g. 'hypura pull owner/repo-GGUF:Q4_K_M').",
            query
        );
    }

    anyhow::bail!(
        "Ollama manifest for '{}' does not contain a GGUF model layer (application/vnd.ollama.image.model).",
        query
    )
}

/// Execute download via system `curl` with progress streaming, resume, and atomic completion.
pub async fn download_model_with_progress(
    spec: &PullSpec,
    models_dir: &Path,
    force: bool,
    progress_tx: Option<mpsc::Sender<PullResponseChunk>>,
) -> Result<PathBuf> {
    let target_dir = models_dir.join(&spec.alias);
    tokio::fs::create_dir_all(&target_dir).await.with_context(|| {
        format!("Failed to create target directory {}", target_dir.display())
    })?;

    let final_path = target_dir.join(&spec.filename);
    let part_path = target_dir.join(format!("{}.part", spec.filename));

    // If final file already exists and force is false, return immediately
    if final_path.exists() && !force {
        if let Some(ref tx) = progress_tx {
            let size = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
            let _ = tx
                .send(PullResponseChunk {
                    status: "already downloaded".into(),
                    digest: None,
                    total: Some(size),
                    completed: Some(size),
                })
                .await;
            let _ = tx
                .send(PullResponseChunk {
                    status: "success".into(),
                    digest: None,
                    total: Some(size),
                    completed: Some(size),
                })
                .await;
        }
        return Ok(final_path);
    }

    if let Some(ref tx) = progress_tx {
        let _ = tx
            .send(PullResponseChunk {
                status: "pulling manifest".into(),
                digest: None,
                total: spec.expected_size_bytes,
                completed: Some(0),
            })
            .await;
    }

    // Try to get total size via curl HEAD request
    let total_bytes = match get_remote_content_length(&spec.download_url).await {
        Ok(len) if len > 0 => Some(len),
        _ => spec.expected_size_bytes,
    };

    if let Some(ref tx) = progress_tx {
        let completed = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
        let _ = tx
            .send(PullResponseChunk {
                status: format!("downloading {}", spec.filename),
                digest: None,
                total: total_bytes,
                completed: Some(completed),
            })
            .await;
    }

    // Spawn curl with resume (-C -), follow-redirects (-L), silent but show error (-sS)
    let mut cmd = tokio::process::Command::new("curl");
    cmd.arg("-L")
        .arg("-C")
        .arg("-")
        .arg("-f")
        .arg("-sS")
        .arg("-o")
        .arg(&part_path)
        .arg(&spec.download_url);

    let mut child = cmd.spawn().with_context(|| {
        format!(
            "Failed to launch curl process for download from {}",
            spec.download_url
        )
    })?;

    // Monitor file size while curl runs in background
    let part_clone = part_path.clone();
    let tx_clone = progress_tx.clone();
    let monitor_handle = tokio::spawn(async move {
        let mut last_completed = 0u64;
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if let Ok(meta) = tokio::fs::metadata(&part_clone).await {
                let current = meta.len();
                if current != last_completed {
                    last_completed = current;
                    if let Some(ref tx) = tx_clone {
                        if tx
                            .send(PullResponseChunk {
                                status: "downloading".into(),
                                digest: None,
                                total: total_bytes,
                                completed: Some(current),
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        }
    });

    let status = child.wait().await?;
    monitor_handle.abort();

    if !status.success() {
        anyhow::bail!(
            "Download failed: curl exited with status {:?}. Partial download kept at {}",
            status.code(),
            part_path.display()
        );
    }

    // Atomic rename .part -> final .gguf
    tokio::fs::rename(&part_path, &final_path)
        .await
        .with_context(|| {
            format!(
                "Failed to rename {} to {}",
                part_path.display(),
                final_path.display()
            )
        })?;

    let final_size = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);

    if let Some(ref tx) = progress_tx {
        let _ = tx
            .send(PullResponseChunk {
                status: "verifying sha256 digest".into(),
                digest: None,
                total: Some(final_size),
                completed: Some(final_size),
            })
            .await;
        let _ = tx
            .send(PullResponseChunk {
                status: "success".into(),
                digest: None,
                total: Some(final_size),
                completed: Some(final_size),
            })
            .await;
    }

    Ok(final_path)
}

/// Helper: query HTTP Content-Length header via curl HEAD request
async fn get_remote_content_length(url: &str) -> Result<u64> {
    let output = tokio::process::Command::new("curl")
        .arg("-sI")
        .arg("-L")
        .arg(url)
        .output()
        .await?;

    if !output.status.success() {
        anyhow::bail!("HEAD request failed");
    }

    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines().rev() {
        let line_lower = line.to_lowercase();
        if line_lower.starts_with("content-length:") {
            if let Some(val_str) = line.split(':').nth(1) {
                if let Ok(num) = val_str.trim().parse::<u64>() {
                    return Ok(num);
                }
            }
        }
    }

    anyhow::bail!("Content-Length header not found");
}

/// Delete a model and its associated files (including all shards if split).
/// Returns the model display name and total bytes freed.
pub fn delete_model_files(
    query: &str,
    models_dir: Option<&Path>,
    registry: &mut crate::server::registry::ModelRegistry,
) -> Result<(String, u64)> {
    // If a custom models_dir was provided, set it on registry and refresh
    if let Some(dir) = models_dir {
        registry.set_custom_dir(Some(dir.to_path_buf()));
    } else {
        registry.refresh();
    }

    let model_entry = registry
        .resolve_model(query)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", query))?;

    let main_path = model_entry.path.clone();
    let model_name = model_entry.name.clone();
    let mut total_freed = 0u64;

    if !main_path.exists() {
        anyhow::bail!("Model file does not exist at {}", main_path.display());
    }

    // Check if sharded model (e.g. *-00001-of-00004.gguf)
    let parent_dir = main_path.parent();
    let filename_str = main_path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    if let Some(pos) = filename_str.find("-00001-of-") {
        let prefix = &filename_str[..pos];
        if let Some(dir) = parent_dir {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with(prefix) && name.ends_with(".gguf") {
                            if let Ok(meta) = std::fs::metadata(&p) {
                                total_freed += meta.len();
                            }
                            let _ = std::fs::remove_file(&p);
                        }
                    }
                }
            }
        }
    } else {
        if let Ok(meta) = std::fs::metadata(&main_path) {
            total_freed += meta.len();
        }
        std::fs::remove_file(&main_path)
            .with_context(|| format!("Failed to delete {}", main_path.display()))?;
    }

    // If parent directory is empty or only had this model, attempt to clean up empty dir
    if let Some(dir) = parent_dir {
        if let Some(custom_root) = models_dir.or(registry.custom_dir.as_deref()) {
            if dir != custom_root {
                let _ = std::fs::remove_dir(dir); // only succeeds if empty
            }
        }
    }

    // Refresh registry to update tags and loaded state
    registry.refresh();

    Ok((model_name, total_freed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_pull_spec_aliases() {
        let gemma = resolve_pull_spec("gemma-4:26b").unwrap();
        assert_eq!(gemma.alias, "gemma-4-26b-a4b");
        assert!(gemma.filename.contains("gemma-4-26B-A4B"));
        assert!(gemma.download_url.contains("bartowski"));

        let qwen = resolve_pull_spec("qwen3.8:27b").unwrap();
        assert_eq!(qwen.alias, "qwen3.8-27b");
        assert!(qwen.filename.contains("Qwen3.8-27B-Q4_K_M"));

        let coder = resolve_pull_spec("qwen2.5-coder:32b").unwrap();
        assert_eq!(coder.alias, "qwen2.5-coder-32b-q3km");
        assert!(coder.filename.contains("q3_k_m"));

        let mistral = resolve_pull_spec("mistral-small:24b").unwrap();
        assert_eq!(mistral.alias, "mistral-small-24b-q4km");
    }

    #[test]
    fn test_resolve_pull_spec_direct_url() {
        let spec = resolve_pull_spec("https://example.com/models/test-model-q4.gguf").unwrap();
        assert_eq!(spec.filename, "test-model-q4.gguf");
        assert_eq!(spec.alias, "test-model-q4");
    }

    #[test]
    fn test_resolve_pull_spec_huggingface_repo() {
        let spec = resolve_pull_spec("unsloth/Qwen3.8-27B-GGUF:Q4_K_M").unwrap();
        assert_eq!(spec.alias, "qwen3.8-27b-gguf");
        assert_eq!(spec.filename, "Qwen3.8-27B-Q4_K_M.gguf");
        assert!(spec.download_url.contains("unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf"));
    }

    #[test]
    fn test_parse_ollama_ref() {
        assert_eq!(
            parse_ollama_ref("qwen2.5:3b"),
            ("library".into(), "qwen2.5".into(), "3b".into())
        );
        assert_eq!(
            parse_ollama_ref("llama3.2"),
            ("library".into(), "llama3.2".into(), "latest".into())
        );
        assert_eq!(
            parse_ollama_ref("lklimek/gemma4-text:26b-a4b-it-qat"),
            ("lklimek".into(), "gemma4-text".into(), "26b-a4b-it-qat".into())
        );
    }

    #[test]
    fn test_parse_ollama_manifest_gguf_and_per_tensor() {
        let gguf_manifest = r#"{
            "schemaVersion": 2,
            "layers": [
                {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc123", "size": 1929903008},
                {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:def456", "size": 128}
            ]
        }"#;
        let spec = parse_ollama_manifest_json(gguf_manifest, "qwen2.5:3b", "library", "qwen2.5", "3b").unwrap();
        assert_eq!(spec.alias, "qwen2.5-3b");
        assert_eq!(spec.filename, "qwen2.5-3b.gguf");
        assert_eq!(spec.expected_size_bytes, Some(1929903008));
        assert_eq!(
            spec.download_url,
            "https://registry.ollama.ai/v2/library/qwen2.5/blobs/sha256:abc123"
        );

        let tensor_manifest = r#"{
            "schemaVersion": 2,
            "layers": [
                {"mediaType": "application/vnd.ollama.image.tensor", "digest": "sha256:111", "size": 9320}
            ]
        }"#;
        let err = parse_ollama_manifest_json(
            tensor_manifest,
            "inkling-2-mini:26b-a5b-nvfp4",
            "library",
            "inkling-2-mini",
            "26b-a5b-nvfp4",
        )
        .unwrap_err();
        assert!(err.to_string().contains("per-tensor format"));
    }

    #[test]
    fn test_find_best_hf_gguf_file() {
        let tree = r#"[
            {"type": "file", "path": "mmproj-F16.gguf", "size": 500000000},
            {"type": "file", "path": "Inkling-2-Mini-UD-Q4_K_M.gguf", "size": 16500000000},
            {"type": "file", "path": "Inkling-2-Mini-Q4_K_M.gguf", "size": 16200000000},
            {"type": "file", "path": "Inkling-2-Mini-Q8_0-00002-of-00002.gguf", "size": 12000000000}
        ]"#;
        let (path, size) = find_best_hf_gguf_file(tree, "Q4_K_M").unwrap();
        assert_eq!(path, "Inkling-2-Mini-Q4_K_M.gguf");
        assert_eq!(size, Some(16200000000));

        let (ud_path, _) = find_best_hf_gguf_file(tree, "UD-Q4_K_M").unwrap();
        assert_eq!(ud_path, "Inkling-2-Mini-UD-Q4_K_M.gguf");

        assert!(find_best_hf_gguf_file(tree, "Q8_0").is_none());
    }
}
