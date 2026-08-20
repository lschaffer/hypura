use std::time::Instant;

use axum::body::Body;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::compute::inference::{GeneratedToken, GenerationResult};
use crate::server::ollama_types::*;

/// Convert a token channel into an NDJSON streaming body for `/api/generate`.
pub fn ndjson_generate_stream(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> Body {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        // Stream token chunks
        while let Some(token) = token_rx.recv().await {
            let chunk = GenerateResponseChunk {
                model: model_name.clone(),
                created_at: now_rfc3339(),
                response: token.text,
                done: false,
                done_reason: None,
                total_duration: None,
                load_duration: None,
                prompt_eval_count: None,
                prompt_eval_duration: None,
                eval_count: None,
                eval_duration: None,
            };
            let mut line = serde_json::to_string(&chunk).unwrap_or_default();
            line.push('\n');
            if tx.send(Ok(line)).await.is_err() {
                return;
            }
        }

        // Final chunk with timing
        let total_ns = request_start.elapsed().as_nanos() as u64;
        let result = result_rx.await.ok();
        let final_chunk = GenerateResponseChunk {
            model: model_name,
            created_at: now_rfc3339(),
            response: String::new(),
            done: true,
            done_reason: Some("stop".into()),
            total_duration: Some(total_ns),
            load_duration: Some(load_duration_ns),
            prompt_eval_count: result.as_ref().map(|r| r.prompt_tokens),
            prompt_eval_duration: result
                .as_ref()
                .map(|r| (r.prompt_eval_ms * 1_000_000.0) as u64),
            eval_count: result.as_ref().map(|r| r.tokens_generated),
            eval_duration: result.as_ref().map(|r| {
                if r.tok_per_sec_avg > 0.0 {
                    (r.tokens_generated as f64 / r.tok_per_sec_avg * 1e9) as u64
                } else {
                    0
                }
            }),
        };
        let mut line = serde_json::to_string(&final_chunk).unwrap_or_default();
        line.push('\n');
        let _ = tx.send(Ok(line)).await;
    });

    Body::from_stream(ReceiverStream::new(rx))
}

/// Convert a token channel into an NDJSON streaming body for `/api/chat`.
pub fn ndjson_chat_stream(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> Body {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut full_response = String::new();
        let mut prefix_buffer = String::new();
        let mut is_tool_call = false;
        let mut prefix_checked = false;

        while let Some(token) = token_rx.recv().await {
            full_response.push_str(&token.text);

            if !prefix_checked {
                prefix_buffer.push_str(&token.text);
                let trimmed = prefix_buffer.trim_start();

                // 1. Detect tool calls or reasoning channels early to prevent streaming raw JSON/reasoning to UI
                if trimmed.starts_with("<tool_call")
                    || trimmed.starts_with("<|tool_call")
                    || trimmed.starts_with("```tool_call")
                    || trimmed.starts_with("<atem:function_calls")
                    || trimmed.starts_with("to=functions.")
                    || (trimmed.starts_with("to=") && !trimmed.starts_with("to=user"))
                    || (trimmed.starts_with("<|start|>assistant to=")
                        && !trimmed.starts_with("<|start|>assistant to=user"))
                {
                    is_tool_call = true;
                    prefix_checked = true;
                    continue;
                }

                // 2. Strip Harmony channel headers like `to=user<|message|>` or `to=user\n`
                if let Some(msg_idx) = trimmed.find("<|message|>") {
                    let user_text = &trimmed[msg_idx + "<|message|>".len()..];
                    if !user_text.is_empty() {
                        let chunk = make_chat_chunk(&model_name, user_text.to_string(), false);
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    prefix_checked = true;
                    continue;
                }

                if trimmed.starts_with("to=user\n") {
                    let user_text = &trimmed["to=user\n".len()..];
                    if !user_text.is_empty() {
                        let chunk = make_chat_chunk(&model_name, user_text.to_string(), false);
                        if tx.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    prefix_checked = true;
                    continue;
                }

                // If not matching any known protocol prefix, flush buffer and stream normally
                if prefix_buffer.len() > 30
                    || (!trimmed.starts_with("to=")
                        && !trimmed.starts_with("<|")
                        && !trimmed.starts_with("<tool"))
                {
                    let chunk = make_chat_chunk(&model_name, prefix_buffer.clone(), false);
                    if tx.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                    prefix_checked = true;
                }
                continue;
            }

            if !is_tool_call {
                let chunk = make_chat_chunk(&model_name, token.text, false);
                if tx.send(Ok(chunk)).await.is_err() {
                    return;
                }
            }
        }

        let total_ns = request_start.elapsed().as_nanos() as u64;
        let result = result_rx.await.ok();
        tracing::info!("STREAM CHAT RAW RESPONSE: {:?}", full_response);
        let (cleaned_content, tool_calls) = crate::server::chat::parse_tool_calls(&full_response);
        tracing::info!("STREAM PARSED TOOL CALLS: {:?}, CLEANED CONTENT: {:?}", tool_calls, cleaned_content);
        let final_chunk = ChatResponseChunk {
            model: model_name,
            created_at: now_rfc3339(),
            message: ChatMessage {
                role: "assistant".into(),
                content: if is_tool_call { String::new() } else { cleaned_content },
                tool_calls,
            },
            done: true,
            done_reason: Some("stop".into()),
            total_duration: Some(total_ns),
            load_duration: Some(load_duration_ns),
            prompt_eval_count: result.as_ref().map(|r| r.prompt_tokens),
            prompt_eval_duration: result
                .as_ref()
                .map(|r| (r.prompt_eval_ms * 1_000_000.0) as u64),
            eval_count: result.as_ref().map(|r| r.tokens_generated),
            eval_duration: result.as_ref().map(|r| {
                if r.tok_per_sec_avg > 0.0 {
                    (r.tokens_generated as f64 / r.tok_per_sec_avg * 1e9) as u64
                } else {
                    0
                }
            }),
        };
        let mut line = serde_json::to_string(&final_chunk).unwrap_or_default();
        line.push('\n');
        let _ = tx.send(Ok(line)).await;
    });

    Body::from_stream(ReceiverStream::new(rx))
}

fn make_chat_chunk(model_name: &str, content: String, done: bool) -> String {
    let chunk = ChatResponseChunk {
        model: model_name.to_string(),
        created_at: now_rfc3339(),
        message: ChatMessage {
            role: "assistant".into(),
            content,
            tool_calls: None,
        },
        done,
        done_reason: None,
        total_duration: None,
        load_duration: None,
        prompt_eval_count: None,
        prompt_eval_duration: None,
        eval_count: None,
        eval_duration: None,
    };
    let mut line = serde_json::to_string(&chunk).unwrap_or_default();
    line.push('\n');
    line
}
