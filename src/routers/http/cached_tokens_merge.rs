//! Prefiller cached tokens reporting for PD disaggregation.
//!
//! Port of vllm-ascend#11630 ([BugFix] Report prefiller cached tokens in PD proxy).
//!
//! In PD disaggregated mode with prefix caching enabled, prefix cache hits happen
//! on the prefiller, but the response returned to the client comes from the
//! decoder, whose reported `cached_tokens` does not include the prefiller's
//! prefix cache hits. These utilities extract the cached token count from the
//! prefiller's response and inject it into the decode response — both the
//! non-streaming JSON body and the final usage chunk of a streaming SSE
//! response (the empty-`choices` chunk sent when `stream_options.include_usage`
//! is enabled, or the nested `response.usage` of a Responses API
//! `response.completed` event).

use bytes::Bytes;
use futures::stream::{self, StreamExt};
use serde_json::{json, Value};
use tracing::debug;

/// Extract the prefiller's cached token count from a prefill response JSON.
///
/// Supports both the OpenAI Chat/Completions format
/// (`usage.prompt_tokens_details.cached_tokens`) and the Responses API format
/// (`usage.input_tokens_details.cached_tokens`). Returns `None` when the
/// prefiller did not report cached tokens.
pub fn extract_prefill_cached_tokens(prefill_json: &Value) -> Option<u64> {
    let usage = prefill_json.get("usage")?;
    for key in ["prompt_tokens_details", "input_tokens_details"] {
        if let Some(cached) = usage
            .get(key)
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
        {
            return Some(cached);
        }
    }
    None
}

/// Inject the prefiller's cached token count into a usage object, creating or
/// replacing the details object as needed. Matches the details field name to
/// the API format of the usage block.
fn inject_into_usage(usage: &mut Value, cached_tokens: u64) -> bool {
    let Some(obj) = usage.as_object_mut() else {
        return false;
    };
    for key in ["prompt_tokens_details", "input_tokens_details"] {
        if let Some(details) = obj.get_mut(key) {
            if details.is_object() {
                details["cached_tokens"] = json!(cached_tokens);
            } else {
                *details = json!({ "cached_tokens": cached_tokens });
            }
            return true;
        }
    }
    if obj.contains_key("prompt_tokens") {
        obj.insert(
            "prompt_tokens_details".to_string(),
            json!({ "cached_tokens": cached_tokens }),
        );
        true
    } else if obj.contains_key("input_tokens") {
        obj.insert(
            "input_tokens_details".to_string(),
            json!({ "cached_tokens": cached_tokens }),
        );
        true
    } else {
        false
    }
}

/// Inject the prefiller's cached token count into a full decode response JSON.
///
/// Handles the top-level `usage` of Chat/Completions and Responses API
/// responses, as well as the usage nested under `response` in Responses API
/// streaming `response.completed` events. Returns whether an injection
/// happened.
pub fn inject_cached_tokens_in_json(decode_json: &mut Value, cached_tokens: u64) -> bool {
    if let Some(usage) = decode_json.get_mut("usage") {
        if inject_into_usage(usage, cached_tokens) {
            return true;
        }
    }
    if let Some(usage) = decode_json
        .get_mut("response")
        .and_then(|response| response.get_mut("usage"))
    {
        if inject_into_usage(usage, cached_tokens) {
            return true;
        }
    }
    false
}

/// Inject into a single streaming chunk. Content chunks (non-empty `choices`)
/// are skipped; the final usage chunk (empty or absent `choices`) is injected.
fn inject_cached_tokens_in_chunk(chunk_json: &mut Value, cached_tokens: u64) -> bool {
    if let Some(choices) = chunk_json.get("choices").and_then(Value::as_array) {
        if !choices.is_empty() {
            return false;
        }
    }
    inject_cached_tokens_in_json(chunk_json, cached_tokens)
}

/// Try to inject the prefiller's cached token count into a full (non-streaming)
/// JSON response body. The body is passed through unchanged when it is not
/// JSON or has no usable usage block. Returns whether the body was modified.
pub fn inject_cached_tokens_in_body(body: &mut Bytes, cached_tokens: u64) -> bool {
    let Ok(mut decode_json) = serde_json::from_slice::<Value>(&*body) else {
        return false;
    };
    if !inject_cached_tokens_in_json(&mut decode_json, cached_tokens) {
        return false;
    }
    match serde_json::to_vec(&decode_json) {
        Ok(encoded) => {
            *body = Bytes::from(encoded);
            true
        }
        Err(_) => false,
    }
}

/// Stateful transformer that injects the prefiller's cached token count into
/// the final usage chunk of an SSE stream coming from the decode server.
///
/// SSE `data:` lines are JSON-parsed individually; lines that are not valid
/// JSON (e.g. `[DONE]`) or that are not modified are forwarded byte-for-byte.
pub struct CachedTokensSseInjector {
    cached_tokens: Option<u64>,
    buffer: Vec<u8>,
}

impl CachedTokensSseInjector {
    pub fn new(cached_tokens: Option<u64>) -> Self {
        Self {
            cached_tokens,
            buffer: Vec::new(),
        }
    }

    /// Process an incoming byte chunk from the decode stream and return the
    /// bytes to forward to the client. Lines are only transformed once they
    /// are fully received (newline-terminated).
    pub fn process(&mut self, chunk: &[u8]) -> Bytes {
        self.buffer.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=pos).collect();
            process_sse_line(&line, self.cached_tokens, &mut out);
        }
        Bytes::from(out)
    }

    /// Flush any buffered partial line when the decode stream ends.
    pub fn finish(&mut self) -> Bytes {
        let remaining = std::mem::take(&mut self.buffer);
        let mut out = Vec::new();
        if !remaining.is_empty() {
            process_sse_line(&remaining, self.cached_tokens, &mut out);
        }
        Bytes::from(out)
    }
}

fn process_sse_line(line: &[u8], cached_tokens: Option<u64>, out: &mut Vec<u8>) {
    let Some(ct) = cached_tokens else {
        out.extend_from_slice(line);
        return;
    };
    let (body, newline) = match line.split_last() {
        Some((&b'\n', rest)) => (rest, &b"\n"[..]),
        _ => (line, &b""[..]),
    };
    let body = match body.split_last() {
        Some((&b'\r', rest)) => rest,
        _ => body,
    };
    if !body.starts_with(b"data:") {
        out.extend_from_slice(line);
        return;
    }
    let payload = body.strip_prefix(b"data:").unwrap_or(body);
    let payload = payload.strip_prefix(b" ").unwrap_or(payload);
    let Ok(mut chunk_json) = serde_json::from_slice::<Value>(payload) else {
        out.extend_from_slice(line);
        return;
    };
    if !inject_cached_tokens_in_chunk(&mut chunk_json, ct) {
        out.extend_from_slice(line);
        return;
    }
    let Ok(encoded) = serde_json::to_vec(&chunk_json) else {
        out.extend_from_slice(line);
        return;
    };
    debug!(
        "Injected prefiller cached_tokens={} into streaming usage chunk",
        ct
    );
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(&encoded);
    out.extend_from_slice(newline);
}

/// Wrap a decode response byte stream, injecting the prefiller's cached token
/// count into the final usage chunk of the SSE stream. Error items are
/// forwarded unchanged.
pub fn inject_cached_tokens_stream<S, E>(
    decode_stream: S,
    cached_tokens: u64,
) -> impl futures::Stream<Item = Result<Bytes, E>> + Send + 'static
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let injector = CachedTokensSseInjector::new(Some(cached_tokens));
    let decode_stream = Box::pin(decode_stream);
    stream::unfold(
        (decode_stream, injector, false),
        |(mut decode_stream, mut injector, finished)| async move {
            if finished {
                return None;
            }
            match decode_stream.next().await {
                Some(Ok(chunk)) => {
                    let out = injector.process(&chunk);
                    Some((Ok(out), (decode_stream, injector, finished)))
                }
                Some(Err(e)) => Some((Err(e), (decode_stream, injector, finished))),
                None => {
                    let out = injector.finish();
                    Some((Ok(out), (decode_stream, injector, true)))
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run_injector(input: &str, cached_tokens: Option<u64>) -> String {
        let mut injector = CachedTokensSseInjector::new(cached_tokens);
        let mut out = injector.process(input.as_bytes()).to_vec();
        out.extend_from_slice(&injector.finish());
        String::from_utf8(out).unwrap()
    }

    fn data_lines(output: &str) -> Vec<Value> {
        output
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|payload| *payload != "[DONE]")
            .map(|payload| serde_json::from_str::<Value>(payload).unwrap())
            .collect()
    }

    #[test]
    fn test_extract_prefill_cached_tokens_chat_format() {
        let prefill = json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
            "usage": {
                "prompt_tokens": 256,
                "completion_tokens": 1,
                "total_tokens": 257,
                "prompt_tokens_details": {"cached_tokens": 128}
            }
        });
        assert_eq!(extract_prefill_cached_tokens(&prefill), Some(128));
    }

    #[test]
    fn test_extract_prefill_cached_tokens_responses_format() {
        let prefill = json!({
            "id": "resp_1",
            "usage": {
                "input_tokens": 256,
                "output_tokens": 1,
                "total_tokens": 257,
                "input_tokens_details": {"cached_tokens": 64}
            }
        });
        assert_eq!(extract_prefill_cached_tokens(&prefill), Some(64));
    }

    #[test]
    fn test_extract_prefill_cached_tokens_missing_or_invalid() {
        assert_eq!(extract_prefill_cached_tokens(&json!({})), None);
        assert_eq!(
            extract_prefill_cached_tokens(&json!({"usage": {"prompt_tokens": 5}})),
            None
        );
        assert_eq!(
            extract_prefill_cached_tokens(&json!({
                "usage": {"prompt_tokens_details": {"cached_tokens": "many"}}
            })),
            None
        );
    }

    #[test]
    fn test_inject_creates_prompt_tokens_details() {
        let mut decode = json!({
            "choices": [{"index": 0, "message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
        });
        assert!(inject_cached_tokens_in_json(&mut decode, 128));
        assert_eq!(
            decode["usage"]["prompt_tokens_details"]["cached_tokens"],
            128
        );
    }

    #[test]
    fn test_inject_creates_input_tokens_details() {
        let mut decode = json!({
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        });
        assert!(inject_cached_tokens_in_json(&mut decode, 64));
        assert_eq!(decode["usage"]["input_tokens_details"]["cached_tokens"], 64);
    }

    #[test]
    fn test_inject_replaces_existing_details() {
        let mut decode = json!({
            "usage": {
                "prompt_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 0}
            }
        });
        assert!(inject_cached_tokens_in_json(&mut decode, 128));
        assert_eq!(
            decode["usage"]["prompt_tokens_details"]["cached_tokens"],
            128
        );
    }

    #[test]
    fn test_inject_responses_streaming_completed_event() {
        let mut event = json!({
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0}
                }
            }
        });
        assert!(inject_cached_tokens_in_json(&mut event, 64));
        assert_eq!(
            event["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            64
        );
    }

    #[test]
    fn test_inject_returns_false_without_usage() {
        let mut decode = json!({"choices": []});
        assert!(!inject_cached_tokens_in_json(&mut decode, 128));
        let mut null_usage = json!({"usage": null});
        assert!(!inject_cached_tokens_in_json(&mut null_usage, 128));
    }

    #[test]
    fn test_inject_chunk_skips_content_chunks() {
        let mut chunk = json!({
            "id": "1",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}],
            "usage": {"prompt_tokens": 10}
        });
        assert!(!inject_cached_tokens_in_chunk(&mut chunk, 128));
        assert!(chunk["usage"].get("prompt_tokens_details").is_none());
    }

    #[test]
    fn test_sse_injector_injects_final_usage_chunk() {
        let content_line = "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}";
        let usage_line = "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}";
        let input = format!("{content_line}\n\n{usage_line}\n\ndata: [DONE]\n\n");

        let output = run_injector(&input, Some(128));

        assert!(output.contains("[DONE]"));
        let lines = data_lines(&output);
        let usage_chunk = lines
            .iter()
            .find(|v| v.get("usage").is_some())
            .expect("usage chunk present");
        assert_eq!(
            usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"],
            128
        );
        assert_eq!(usage_chunk["usage"]["prompt_tokens"], 10);
        let content_chunk = lines
            .iter()
            .find(|v| {
                v.get("choices")
                    .and_then(Value::as_array)
                    .is_some_and(|a| !a.is_empty())
            })
            .expect("content chunk present");
        assert_eq!(content_chunk["choices"][0]["delta"]["content"], "Hi");
    }

    #[test]
    fn test_sse_injector_handles_arbitrary_chunk_boundaries() {
        let input = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}\n\n",
            "data: [DONE]\n\n",
        );
        let single_shot = run_injector(input, Some(128));

        let mut injector = CachedTokensSseInjector::new(Some(128));
        let mut split = Vec::new();
        for byte in input.bytes() {
            split.extend_from_slice(&injector.process(&[byte]));
        }
        split.extend_from_slice(&injector.finish());
        assert_eq!(String::from_utf8(split).unwrap(), single_shot);

        let lines = data_lines(&single_shot);
        let usage_chunk = lines.iter().find(|v| v.get("usage").is_some()).unwrap();
        assert_eq!(
            usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"],
            128
        );
    }

    #[test]
    fn test_sse_injector_passthrough_without_usage_chunk() {
        let input = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let output = run_injector(input, Some(128));
        assert_eq!(output, input);
    }

    #[test]
    fn test_sse_injector_passthrough_when_disabled() {
        let input = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10}}\n\n",
            "data: [DONE]\n\n",
        );
        let output = run_injector(input, None);
        assert_eq!(output, input);
    }

    #[test]
    fn test_sse_injector_responses_api_completed_event() {
        let input = concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"total_tokens\":12,\"input_tokens_details\":{\"cached_tokens\":0}}}}\n\n",
            "data: [DONE]\n\n",
        );
        let output = run_injector(input, Some(64));

        let lines = data_lines(&output);
        let completed = lines
            .iter()
            .find(|v| v["type"] == "response.completed")
            .expect("response.completed event present");
        assert_eq!(
            completed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            64
        );
        assert!(output.contains("event: response.completed\n"));
        assert!(output.contains("data: [DONE]"));
    }

    #[test]
    fn test_sse_injector_preserves_non_data_lines() {
        let input = concat!(
            ": keep-alive comment\n\n",
            "event: ping\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3}}\n",
            "\n",
            "data: [DONE]\n\n",
        );
        let output = run_injector(input, Some(7));
        assert!(output.contains(": keep-alive comment\n"));
        assert!(output.contains("event: ping\n"));
        assert!(output.contains("data: [DONE]"));
        let lines = data_lines(&output);
        assert_eq!(
            lines[0]["usage"]["prompt_tokens_details"]["cached_tokens"],
            7
        );
    }

    #[test]
    fn test_sse_injector_tolerates_crlf() {
        let input =
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5}}\r\n\r\ndata: [DONE]\r\n\r\n";
        let output = run_injector(input, Some(9));
        let lines = data_lines(&output);
        assert_eq!(
            lines[0]["usage"]["prompt_tokens_details"]["cached_tokens"],
            9
        );
        assert!(output.contains("data: [DONE]"));
    }

    #[test]
    fn test_inject_body_json_modified() {
        let mut body = Bytes::from(
            "{\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}"
                .to_string(),
        );
        assert!(inject_cached_tokens_in_body(&mut body, 128));
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            parsed["usage"]["prompt_tokens_details"]["cached_tokens"],
            128
        );
    }

    #[test]
    fn test_inject_body_non_json_passthrough() {
        let original = Bytes::from_static(b"not json");
        let mut body = original.clone();
        assert!(!inject_cached_tokens_in_body(&mut body, 128));
        assert_eq!(body, original);
    }

    #[test]
    fn test_inject_cached_tokens_stream_end_to_end() {
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            )),
            Ok(Bytes::from_static(
                b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}\n\n",
            )),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let stream = inject_cached_tokens_stream(futures::stream::iter(chunks), 128);
        let mut total = Vec::new();
        for chunk in futures::executor::block_on_stream(Box::pin(stream)) {
            total.extend_from_slice(&chunk.unwrap());
        }
        let output = String::from_utf8(total).unwrap();

        assert!(output.contains("data: [DONE]"));
        let lines = data_lines(&output);
        let usage_chunk = lines.iter().find(|v| v.get("usage").is_some()).unwrap();
        assert_eq!(
            usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"],
            128
        );
    }
}
