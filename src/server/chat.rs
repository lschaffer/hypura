use crate::server::ollama_types::{ChatMessage, FunctionCall, ToolCall};

/// Format chat messages (and optional tool definitions) into a prompt string
/// customized for the underlying model architecture (e.g. GPT-OSS / Harmony vs ChatML).
pub fn format_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
    arch: Option<&str>,
) -> String {
    let is_gpt_oss = arch.map_or(false, |a| {
        let l = a.to_lowercase();
        l == "gptoss" || l == "gpt-oss" || l == "openai_moe"
    });

    if is_gpt_oss {
        return format_gptoss_chat_prompt(messages, tools);
    }

    let mut prompt = String::new();

    // Default ChatML format
    let system_msg = messages.iter().find(|m| m.role == "system");
    let base_system = system_msg.map(|m| m.content.as_str());

    let tools_array = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(val),
        _ => None,
    });

    if let Some(tools_json) = tools_array {
        let clean_tools: Vec<&serde_json::Value> = match tools_json {
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|t| t.get("function").unwrap_or(t))
                .collect(),
            _ => vec![tools_json],
        };
        let tools_str = serde_json::to_string_pretty(&clean_tools).unwrap_or_default();
        prompt.push_str("<|im_start|>system\n");
        if let Some(sys) = base_system {
            prompt.push_str(sys);
            prompt.push_str("\n\n");
        }
        prompt.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n");
        prompt.push_str("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        prompt.push_str(&tools_str);
        prompt.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"arg_name\": \"arg_value\"}}\n</tool_call>\nWhen no tool call is needed, respond directly with standard text.<|im_end|>\n");
    } else if let Some(sys) = base_system {
        prompt.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }

    for msg in messages {
        if msg.role == "system" {
            continue;
        }
        prompt.push_str(&format!("<|im_start|>{}\n", msg.role));
        if !msg.content.is_empty() {
            prompt.push_str(&msg.content);
        }
        if let Some(ref tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                let call_obj = serde_json::json!({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                });
                prompt.push_str(&format!(
                    "<tool_call>\n{}\n</tool_call>",
                    serde_json::to_string(&call_obj).unwrap_or_default()
                ));
            }
        }
        prompt.push_str("<|im_end|>\n");
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Specialized prompt formatting for OpenAI Harmony / GPT-OSS models.
fn format_gptoss_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
) -> String {
    let mut prompt = String::new();

    let tools_array = tools.and_then(|val| match val {
        serde_json::Value::Array(arr) if !arr.is_empty() => Some(val),
        _ => None,
    });

    let system_msg = messages.iter().find(|m| m.role == "system");
    let base_system = system_msg
        .map(|m| m.content.as_str())
        .unwrap_or("You are a helpful assistant.");

    if let Some(tools_json) = tools_array {
        let tools_str = serde_json::to_string_pretty(tools_json).unwrap_or_default();
        prompt.push_str("<|start|>system<|message|>");
        prompt.push_str(base_system);
        prompt.push_str("\n\nYou have access to the following functions:\n```json\n");
        prompt.push_str(&tools_str);
        prompt.push_str("\n```\n\nTo call a function, respond strictly in this format:\n\
            to=functions.<function_name><|channel|>commentary<|message|>{\"arg_name\": \"arg_value\"}<|call|>\n\
            When no function call is needed, respond with standard text.<|end|>\n");
    } else if let Some(sys) = system_msg {
        prompt.push_str(&format!("<|start|>system<|message|>{}<|end|>\n", sys.content));
    }

    for msg in messages {
        if msg.role == "system" {
            continue;
        }

        match msg.role.as_str() {
            "user" => {
                prompt.push_str(&format!("<|start|>user<|message|>{}<|end|>\n", msg.content));
            }
            "assistant" => {
                if let Some(ref tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        let args_str =
                            serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                        prompt.push_str(&format!(
                            "<|start|>assistant to=functions.{}<|channel|>commentary<|message|>{args_str}<|call|>\n",
                            tc.function.name
                        ));
                    }
                }
                if !msg.content.is_empty() {
                    prompt.push_str(&format!(
                        "<|start|>assistant<|message|>{}<|return|>\n",
                        msg.content
                    ));
                }
            }
            "tool" => {
                prompt.push_str(&format!("<|start|>tool<|message|>{}<|end|>\n", msg.content));
            }
            other => {
                prompt.push_str(&format!("<|start|>{other}<|message|>{}<|end|>\n", msg.content));
            }
        }
    }

    prompt.push_str("<|start|>assistant");
    prompt
}

/// Parse tool calls and clean message content from the model's raw generated text.
pub fn parse_tool_calls(raw_output: &str) -> (String, Option<Vec<ToolCall>>) {
    let mut tool_calls = Vec::new();
    let mut cleaned_text = raw_output.to_string();

    // 1. Try parsing GPT-OSS / Harmony tool call syntax:
    // e.g. "to=functions.<name><|channel|>commentary<|message|>{...}<|call|>"
    // or "<|start|>assistant to=functions.<name>..."
    while let Some(to_idx) = cleaned_text.find("to=functions.") {
        let after_to = &cleaned_text[to_idx + "to=functions.".len()..];
        let name_end = after_to
            .find(|c: char| c == '<' || c == ' ' || c == '\n' || c == '{')
            .unwrap_or(after_to.len());
        let func_name = after_to[..name_end].trim().to_string();

        let remainder = &after_to[name_end..];
        let args_start_offset = if let Some(msg_idx) = remainder.find("<|message|>") {
            Some(msg_idx + "<|message|>".len())
        } else {
            remainder.find('{')
        };

        if !func_name.is_empty() {
            if let Some(args_offset) = args_start_offset {
                let args_text = &remainder[args_offset..];
                let end_tag_offset = args_text
                    .find("<|call|>")
                    .or_else(|| args_text.find("<|end|>"))
                    .or_else(|| args_text.find("<|return|>"));

                let (raw_args, full_match_len) = if let Some(end_o) = end_tag_offset {
                    (
                        &args_text[..end_o],
                        to_idx + "to=functions.".len() + name_end + args_offset + end_o + 8,
                    )
                } else if let Some(json_end) = find_matching_brace(args_text) {
                    (
                        &args_text[..json_end],
                        to_idx + "to=functions.".len() + name_end + args_offset + json_end,
                    )
                } else {
                    (args_text.trim(), cleaned_text.len())
                };

                if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw_args.trim()) {
                    if val.is_object() {
                        tool_calls.push(ToolCall {
                            id: Some(format!("call_{}", uuid_short())),
                            call_type: Some("function".into()),
                            function: FunctionCall {
                                name: func_name,
                                arguments: val,
                            },
                        });
                        let end_pos = full_match_len.min(cleaned_text.len());
                        cleaned_text.replace_range(to_idx..end_pos, "");
                        continue;
                    }
                }
            }
        }
        break;
    }

    // 2. Try parsing Gemma / standard call syntax: <|tool_call>call:name{...}<tool_call|> or call:name{...}
    let gemma_markers = [
        ("<|tool_call>call:", "<tool_call|>"),
        ("<tool_call>call:", "</tool_call>"),
        ("<|tool_call>call:", "<|end|>"),
    ];

    for (start_tag, end_tag) in gemma_markers {
        while let Some(start_idx) = cleaned_text.find(start_tag) {
            let after_start = &cleaned_text[start_idx + start_tag.len()..];
            if let Some(end_idx) = after_start.find(end_tag) {
                let call_str = &after_start[..end_idx];
                if let Some((name, args_val)) = parse_gemma_call(call_str) {
                    tool_calls.push(ToolCall {
                        id: Some(format!("call_{}", uuid_short())),
                        call_type: Some("function".into()),
                        function: FunctionCall {
                            name,
                            arguments: args_val,
                        },
                    });
                }
                let full_end_idx = start_idx + start_tag.len() + end_idx + end_tag.len();
                cleaned_text.replace_range(start_idx..full_end_idx, "");
            } else {
                break;
            }
        }
    }

    // Fallback: search for un-tagged `call:func_name{...}` anywhere in text
    while let Some(call_idx) = cleaned_text.find("call:") {
        let after_call = &cleaned_text[call_idx + 5..];
        if let Some(end_brace) = find_matching_brace(after_call) {
            let call_str = &after_call[..end_brace];
            if let Some((name, args_val)) = parse_gemma_call(call_str) {
                tool_calls.push(ToolCall {
                    id: Some(format!("call_{}", uuid_short())),
                    call_type: Some("function".into()),
                    function: FunctionCall {
                        name,
                        arguments: args_val,
                    },
                });
                cleaned_text.replace_range(call_idx..call_idx + 5 + end_brace, "");
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // 3. Try parsing Qwen 3.8 XML syntax:
    // <tool_call>
    // <function=name>
    // <parameter=key>value</parameter>
    // </function>
    // </tool_call>
    while let Some(start_idx) = cleaned_text.find("<function=") {
        let after_func = &cleaned_text[start_idx + "<function=".len()..];
        if let Some(func_name_end) = after_func.find('>') {
            let func_name = after_func[..func_name_end].trim().to_string();
            let remainder = &after_func[func_name_end + 1..];
            
            let func_end_idx = remainder.find("</function>").unwrap_or(remainder.len());
            let func_body = &remainder[..func_end_idx];

            let mut arguments = serde_json::Map::new();
            let mut param_search = func_body;
            while let Some(p_start) = param_search.find("<parameter=") {
                let after_p = &param_search[p_start + "<parameter=".len()..];
                if let Some(p_name_end) = after_p.find('>') {
                    let p_name = after_p[..p_name_end].trim().to_string();
                    let p_val_after = &after_p[p_name_end + 1..];
                    if let Some(p_val_end) = p_val_after.find("</parameter>") {
                        let p_val_raw = p_val_after[..p_val_end].trim();
                        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(p_val_raw) {
                            arguments.insert(p_name, json_val);
                        } else {
                            arguments.insert(p_name, serde_json::Value::String(p_val_raw.to_string()));
                        }
                        param_search = &p_val_after[p_val_end + "</parameter>".len()..];
                        continue;
                    }
                }
                break;
            }

            tool_calls.push(ToolCall {
                id: Some(format!("call_{}", uuid_short())),
                call_type: Some("function".into()),
                function: FunctionCall {
                    name: func_name,
                    arguments: serde_json::Value::Object(arguments),
                },
            });

            // Clean up surrounding tags if present
            let full_start = if let Some(tc_open) = cleaned_text[..start_idx].rfind("<tool_call>") {
                let between = &cleaned_text[tc_open + "<tool_call>".len()..start_idx];
                if between.trim().is_empty() {
                    tc_open
                } else {
                    start_idx
                }
            } else {
                start_idx
            };
            let end_offset = start_idx + "<function=".len() + func_name_end + 1 + func_end_idx + "</function>".len();
            let full_end = if let Some(tc_close) = cleaned_text[end_offset..].find("</tool_call>") {
                end_offset + tc_close + "</tool_call>".len()
            } else {
                end_offset
            };
            cleaned_text.replace_range(full_start..full_end.min(cleaned_text.len()), "");
        } else {
            break;
        }
    }

    // 4. Try parsing Standard XML / JSON syntax: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    let json_markers = [
        ("<tool_call>", "</tool_call>"),
        ("<toolcall>", "</toolcall>"),
        ("```tool_call", "```"),
        ("<|tool_call>", "<tool_call|>"),
        ("[TOOL_CALLS]", "[/TOOL_CALLS]"),
    ];

    for (start_tag, end_tag) in json_markers {
        while let Some(start_idx) = cleaned_text.find(start_tag) {
            let after_start = &cleaned_text[start_idx + start_tag.len()..];
            if let Some(end_idx) = after_start.find(end_tag) {
                let inner_json = after_start[..end_idx].trim();
                if let Some(tc) = parse_json_tool_call(inner_json) {
                    tool_calls.push(tc);
                } else if let Ok(val) = serde_json::from_str::<serde_json::Value>(inner_json) {
                    if let Some(arr) = val.as_array() {
                        for item in arr {
                            if let Some(tc) = parse_json_tool_call(&item.to_string()) {
                                tool_calls.push(tc);
                            }
                        }
                    }
                }
                let full_end_idx = start_idx + start_tag.len() + end_idx + end_tag.len();
                cleaned_text.replace_range(start_idx..full_end_idx, "");
            } else {
                break;
            }
        }
    }

    // 4. Try parsing un-tagged [TOOL_CALLS] [...] syntax (e.g. Mistral v3 without end tag)
    if let Some(tc_idx) = cleaned_text.find("[TOOL_CALLS]") {
        let after_tc = &cleaned_text[tc_idx + "[TOOL_CALLS]".len()..];
        let trimmed_after = after_tc.trim_start();
        if trimmed_after.starts_with('[') || trimmed_after.starts_with('{') {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed_after) {
                if let Some(arr) = val.as_array() {
                    for item in arr {
                        if let Some(tc) = parse_json_tool_call(&item.to_string()) {
                            tool_calls.push(tc);
                        }
                    }
                } else if let Some(tc) = parse_json_tool_call(&val.to_string()) {
                    tool_calls.push(tc);
                }
                cleaned_text.replace_range(tc_idx.., "");
            }
        }
    }

    // 6. Clean up thought channels if present: <|channel>thought\n...<channel|> or <thought>...</thought> or <think>...</think>
    cleaned_text = strip_channel_tags(&cleaned_text);

    let trimmed = cleaned_text.trim().to_string();
    let final_tool_calls = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    (trimmed, final_tool_calls)
}

/// Helper: parse Gemma 4 `function_name{param:<|"|>val<|"|>}` or `function_name{param: "val"}`
fn parse_gemma_call(call_str: &str) -> Option<(String, serde_json::Value)> {
    let brace_idx = call_str.find('{')?;
    let func_name = call_str[..brace_idx].trim().to_string();
    if func_name.is_empty() {
        return None;
    }

    let args_slice = call_str[brace_idx..].trim();
    // Normalize Gemma 4 custom quotes <|"|> into standard double quotes "
    let normalized = args_slice.replace("<|\"|>", "\"");

    // Attempt direct JSON parse
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&normalized) {
        if val.is_object() {
            return Some((func_name, val));
        }
    }

    // If direct parse fails, try quoting unquoted keys: {key: "val"} -> {"key": "val"}
    let fixed_json = quote_unquoted_keys(&normalized);
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&fixed_json) {
        if val.is_object() {
            return Some((func_name, val));
        }
    }

    // Fallback: manual key-value parser for simple {key: "val", key2: 123}
    let manual_map = parse_simple_kv(&normalized);
    Some((func_name, serde_json::Value::Object(manual_map)))
}

/// Helper: parse a JSON block containing tool call object
fn parse_json_tool_call(json_str: &str) -> Option<ToolCall> {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
        // Direct object: {"name": "...", "arguments": ...}
        if let Some(name) = val.get("name").and_then(|n| n.as_str()) {
            let arguments = match val.get("arguments") {
                Some(serde_json::Value::String(s)) => {
                    serde_json::from_str(s).unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
                }
                Some(obj @ serde_json::Value::Object(_)) => obj.clone(),
                _ => serde_json::Value::Object(serde_json::Map::new()),
            };
            return Some(ToolCall {
                id: Some(format!("call_{}", uuid_short())),
                call_type: Some("function".into()),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            });
        }

        // Nested in "function": {"function": {"name": "...", "arguments": ...}}
        if let Some(func_obj) = val.get("function") {
            if let Some(name) = func_obj.get("name").and_then(|n| n.as_str()) {
                let arguments = match func_obj.get("arguments") {
                    Some(serde_json::Value::String(s)) => {
                        serde_json::from_str(s).unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
                    }
                    Some(obj @ serde_json::Value::Object(_)) => obj.clone(),
                    _ => serde_json::Value::Object(serde_json::Map::new()),
                };
                return Some(ToolCall {
                    id: Some(format!("call_{}", uuid_short())),
                    call_type: Some("function".into()),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments,
                    },
                });
            }
        }
    }
    None
}

/// Convert unquoted keys `{foo: "bar"}` to `{"foo": "bar"}`
fn quote_unquoted_keys(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 16);
    let mut chars = input.chars().peekable();
    let mut in_string = false;
    let mut escape = false;

    while let Some(c) = chars.next() {
        if in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            out.push(c);
        } else {
            if c == '"' {
                in_string = true;
                out.push(c);
            } else if c.is_alphabetic() || c == '_' {
                // Potential unquoted key
                let mut ident = String::new();
                ident.push(c);
                while let Some(&next_c) = chars.peek() {
                    if next_c.is_alphanumeric() || next_c == '_' {
                        ident.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                // Check if followed by ':'
                let mut ws = String::new();
                while let Some(&next_c) = chars.peek() {
                    if next_c.is_whitespace() {
                        ws.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                if let Some(&':') = chars.peek() {
                    out.push('"');
                    out.push_str(&ident);
                    out.push('"');
                    out.push_str(&ws);
                } else {
                    out.push_str(&ident);
                    out.push_str(&ws);
                }
            } else {
                out.push(c);
            }
        }
    }
    out
}

/// Fallback manual key-value parser
fn parse_simple_kv(input: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    let trimmed = input.trim_matches(|c| c == '{' || c == '}');
    for pair in trimmed.split(',') {
        if let Some((k, v)) = pair.split_once(':') {
            let key = k.trim().trim_matches('"').to_string();
            let raw_val = v.trim().trim_matches('"');
            if let Ok(num) = raw_val.parse::<i64>() {
                map.insert(key, serde_json::Value::Number(num.into()));
            } else if let Ok(b) = raw_val.parse::<bool>() {
                map.insert(key, serde_json::Value::Bool(b));
            } else {
                map.insert(key, serde_json::Value::String(raw_val.to_string()));
            }
        }
    }
    map
}

/// Helper: find matching closing brace `}` accounting for strings and nesting
fn find_matching_brace(s: &str) -> Option<usize> {
    let start = s.find('{')?;
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;

    for (i, c) in s[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(start + i + 1);
            }
        }
    }
    None
}

/// Strip thought channels and special control tags from model output
fn strip_channel_tags(text: &str) -> String {
    let mut result = text.to_string();

    let channel_patterns = [
        ("to=self<|message|>", "<|eom|>"),
        ("to=self<|message|>", "<|eot|>"),
        ("to=self<|message|>", "<|start|>"),
        ("<|start|>assistant to=self<|message|>", "<|eom|>"),
        ("<|start|>assistant to=self<|message|>", "<|eot|>"),
        ("<|start|>assistant to=self<|message|>", "<|start|>"),
        ("<|channel>thought\n", "<channel|>"),
        ("<|channel>thought", "<channel|>"),
        ("<|channel|>thought\n", "<|end|>"),
        ("<|channel|>thought", "<|end|>"),
        ("<|channel|>thought\n", "<channel|>"),
        ("<|channel|>thought", "<channel|>"),
        ("<|channel|>analysis\n", "<|end|>"),
        ("<|channel|>analysis", "<|end|>"),
        ("<|channel|>commentary\n", "<|end|>"),
        ("<|channel|>commentary", "<|end|>"),
        ("<thought>", "</thought>"),
        ("<think>", "</think>"),
    ];

    for (start_tag, end_tag) in channel_patterns {
        while let Some(start_idx) = result.find(start_tag) {
            let after_start = &result[start_idx + start_tag.len()..];
            if let Some(end_idx) = after_start.find(end_tag) {
                let full_end_idx = start_idx + start_tag.len() + end_idx + end_tag.len();
                result.replace_range(start_idx..full_end_idx, "");
            } else {
                result.replace_range(start_idx..start_idx + start_tag.len(), "");
            }
        }
    }

    // Strip remaining to=<recipient><|message|> or to=<recipient>\n headers
    while let Some(to_idx) = result.find("to=") {
        let after_to = &result[to_idx..];
        if let Some(msg_idx) = after_to.find("<|message|>") {
            let full_end = to_idx + msg_idx + "<|message|>".len();
            result.replace_range(to_idx..full_end, "");
        } else if let Some(nl_idx) = after_to.find('\n') {
            let full_end = to_idx + nl_idx + 1;
            result.replace_range(to_idx..full_end, "");
        } else {
            result.replace_range(to_idx..result.len(), "");
        }
    }

    // Clean stray tokens
    let stray_tokens = [
        "<|channel|>final<|message|>",
        "<|channel|>final",
        "<|start|>assistant",
        "<|start|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|message|>",
        "<|header_start|>assistant<|header_end|>",
        "<|header_start|>",
        "<|header_end|>",
        "<|eot|>",
        "<|im_start|>assistant",
        "<|im_start|>",
        "<|im_end|>",
        "to=user",
        "to=self",
        "<atem:function_calls>",
        "</atem:function_calls>",
    ];

    for tok in stray_tokens {
        result = result.replace(tok, "");
    }

    result
}



fn uuid_short() -> String {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{:08x}", nanos)
}









#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gptoss_tool_call() {
        let raw = "<|channel|>thought\nNeed station data\n<|end|><|start|>assistant to=functions.get_station_data<|channel|>commentary<|message|>{\"station_id\": 104, \"sensor\": \"temperature\"}<|call|>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_station_data");
        assert_eq!(
            tcs[0].function.arguments["station_id"],
            serde_json::json!(104)
        );
        assert_eq!(
            tcs[0].function.arguments["sensor"],
            serde_json::json!("temperature")
        );
    }

    #[test]
    fn test_parse_gemma4_tool_call() {
        let raw = "<|channel>thought\nNeed to get weather for Paris\n<channel|><|tool_call>call:get_current_weather{location:<|\"|>Paris<|\"|>,unit:<|\"|>celsius<|\"|>}<tool_call|>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_current_weather");
        assert_eq!(
            tcs[0].function.arguments["location"],
            serde_json::json!("Paris")
        );
        assert_eq!(
            tcs[0].function.arguments["unit"],
            serde_json::json!("celsius")
        );
    }

    #[test]
    fn test_parse_json_tool_call() {
        let raw = "Let me check that for you.\n<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}\n</tool_call>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "Let me check that for you.");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs[0].function.name, "calculator");
        assert_eq!(tcs[0].function.arguments["expression"], serde_json::json!("2+2"));
    }

    #[test]
    fn test_parse_mistral_tool_call() {
        let raw = "[TOOL_CALLS] [{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Tokyo\"}}]";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_current_weather");
        assert_eq!(tcs[0].function.arguments["location"], serde_json::json!("Tokyo"));
    }

    #[test]
    fn test_parse_qwen38_xml_tool_call() {
        let raw = "<think>\nThinking about weather...\n</think>\n<tool_call>\n<function=get_forecast>\n<parameter=location>\nGraz, Austria\n</parameter>\n<parameter=hours>\n12\n</parameter>\n</function>\n</tool_call>";
        let (content, tool_calls) = parse_tool_calls(raw);
        assert_eq!(content, "");
        assert!(tool_calls.is_some());
        let tcs = tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_forecast");
        assert_eq!(tcs[0].function.arguments["location"], serde_json::json!("Graz, Austria"));
        assert_eq!(tcs[0].function.arguments["hours"], serde_json::json!(12));
    }

    #[test]
    fn test_format_chat_prompt_with_tools() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "What is 2+2?".into(),
            tool_calls: None,
        }];
        let tools = serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Evaluate math expression"
                }
            }
        ]);
        let prompt = format_chat_prompt(&messages, Some(&tools), None);
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("\"name\": \"calc\""));
        assert!(prompt.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"));

        let gpt_prompt = format_chat_prompt(&messages, Some(&tools), Some("gptoss"));
        assert!(gpt_prompt.contains("<|start|>system<|message|>"));
        assert!(gpt_prompt.contains("to=functions.<function_name>"));
        assert!(gpt_prompt.contains("<|start|>user<|message|>What is 2+2?<|end|>"));
        assert!(gpt_prompt.ends_with("<|start|>assistant"));
    }
}
