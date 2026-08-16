use crate::server::ollama_types::{ChatMessage, FunctionCall, ToolCall};

/// Format chat messages (and optional tool definitions) into a ChatML prompt string.
pub fn format_chat_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
) -> String {
    let mut prompt = String::new();

    // If tools are provided, prepend them into a system instruction
    if let Some(tools_val) = tools {
        let tools_array = match tools_val {
            serde_json::Value::Array(arr) if !arr.is_empty() => Some(tools_val),
            _ => None,
        };

        if let Some(tools_json) = tools_array {
            let tools_str = serde_json::to_string_pretty(tools_json).unwrap_or_default();
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str("You have access to the following tools:\n");
            prompt.push_str(&tools_str);
            prompt.push_str(
                "\n\nTo use a tool, respond with a tool call in one of the following formats:\n\
                <|tool_call>call:function_name{arg_name:arg_value}<tool_call|>\n\
                or\n\
                <tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"arg_name\": \"arg_value\"}}\n</tool_call>\n\
                Only output tool calls when necessary. When no tool is needed, respond with standard text.\
                <|im_end|>\n",
            );
        }
    }

    for msg in messages {
        prompt.push_str(&format!("<|im_start|>{}\n", msg.role));
        if !msg.content.is_empty() {
            prompt.push_str(&msg.content);
        }
        if let Some(ref tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                let args_str = serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                prompt.push_str(&format!(
                    "<|tool_call>call:{}{}<tool_call|>",
                    tc.function.name, args_str
                ));
            }
        }
        prompt.push_str("<|im_end|>\n");
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Parse tool calls and clean message content from the model's raw generated text.
pub fn parse_tool_calls(raw_output: &str) -> (String, Option<Vec<ToolCall>>) {
    let mut tool_calls = Vec::new();
    let mut cleaned_text = raw_output.to_string();

    // 1. Try parsing Gemma 4 syntax: <|tool_call>call:name{...}<tool_call|> or <tool_call>call:...
    let gemma_markers = [
        ("<|tool_call>call:", "<tool_call|>"),
        ("<tool_call>call:", "</tool_call>"),
        ("call:", "<tool_call|>"),
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

    // 2. Try parsing Standard XML / JSON syntax: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    let json_markers = [
        ("<tool_call>", "</tool_call>"),
        ("<toolcall>", "</toolcall>"),
        ("```tool_call", "```"),
        ("<|tool_call>", "<tool_call|>"),
    ];

    for (start_tag, end_tag) in json_markers {
        while let Some(start_idx) = cleaned_text.find(start_tag) {
            let after_start = &cleaned_text[start_idx + start_tag.len()..];
            if let Some(end_idx) = after_start.find(end_tag) {
                let inner_json = after_start[..end_idx].trim();
                if let Some(tc) = parse_json_tool_call(inner_json) {
                    tool_calls.push(tc);
                }
                let full_end_idx = start_idx + start_tag.len() + end_idx + end_tag.len();
                cleaned_text.replace_range(start_idx..full_end_idx, "");
            } else {
                break;
            }
        }
    }

    // 3. Clean up thought channels if present: <|channel>thought\n...<channel|> or <thought>...</thought>
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

/// Strip `<|channel>thought\n...<channel|>` or `<thought>...</thought>` tags
fn strip_channel_tags(text: &str) -> String {
    let mut result = text.to_string();

    let channel_patterns = [
        ("<|channel>thought\n", "<channel|>"),
        ("<|channel>thought", "<channel|>"),
        ("<thought>", "</thought>"),
    ];

    for (start_tag, end_tag) in channel_patterns {
        while let Some(start_idx) = result.find(start_tag) {
            let after_start = &result[start_idx + start_tag.len()..];
            if let Some(end_idx) = after_start.find(end_tag) {
                let full_end_idx = start_idx + start_tag.len() + end_idx + end_tag.len();
                result.replace_range(start_idx..full_end_idx, "");
            } else {
                // If opening tag exists without closing tag, remove opening tag
                result.replace_range(start_idx..start_idx + start_tag.len(), "");
            }
        }
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
        let prompt = format_chat_prompt(&messages, Some(&tools));
        assert!(prompt.contains("You have access to the following tools:"));
        assert!(prompt.contains("\"name\": \"calc\""));
        assert!(prompt.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"));
    }
}
