#!/usr/bin/env python3
"""
Adapter that lets Claude Code talk to a USAi gateway.

Claude Code speaks the Anthropic Messages API (POST /v1/messages), while USAi
only speaks the OpenAI-style "chat/completions" API. This proxy exposes an
Anthropic Messages endpoint for Claude Code to connect to: it accepts POST
/v1/messages, translates each request into a USAi chat/completions call, and
translates the reply back into Anthropic Messages responses (including the
streaming SSE event sequence). It also serves /v1/models so Claude Code's
`/model` picker can list the available Claude models.

Only the features USAi supports are translated: text, system prompts, images,
tools/tool-use, and streaming. Anthropic-only features that USAi's chat API has
no equivalent for -- extended thinking, documents, web search, citations, and
prompt caching -- are dropped.

Requires the USAI_BASE_URL and USAI_API_KEY environment variables.

Stdlib only. Run:  uv run usai-claude-proxy   (or: python3 main.py)
Then point Claude Code at http://127.0.0.1:7878 via ANTHROPIC_BASE_URL.
"""

import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("USAI_BASE_URL")
API_KEY = os.environ.get("USAI_API_KEY")
LISTEN_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("PROXY_PORT", "7878"))

# Anthropic model ids -> USAi model slugs. Claude Code sends the hyphenated
# Anthropic names; USAi expects its own underscore slugs. Unknown ids pass
# through unchanged so a manually selected USAi slug still works.
MODEL_MAP = {
    "claude-opus-4-8": "claude_4_8_opus",
    "claude-opus-4-7": "claude_4_7_opus",
    "claude-opus-4-5": "claude_4_5_opus",
    "claude-sonnet-4-6": "claude_4_6_sonnet",
    "claude-sonnet-4-5": "claude_4_5_sonnet",
    "claude-haiku-4-5": "claude_4_5_haiku",
}
# Reverse map for advertising Anthropic-style ids on /v1/models.
USAI_TO_ANTHROPIC = {v: k for k, v in MODEL_MAP.items()}
DISPLAY_NAMES = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-5": "Claude Opus 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}

STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def log(*a):
    print("[proxy]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Request translation: Anthropic Messages  ->  Chat Completions
# ---------------------------------------------------------------------------
def _system_to_text(system):
    """Anthropic `system` (string or text blocks) -> plain string."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _image_part(source):
    """Anthropic image source -> chat image_url part, or None if unusable."""
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64":
        media = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    if source.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    return None


def _stringify_tool_result_content(content):
    """Tool-result content -> text. Non-text blocks are flattened to a note."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content)
    parts = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            parts.append(f"[non-text tool result omitted: {block.get('type', 'unknown')}]")
    return "\n".join(parts)


def _convert_message(msg, out):
    """Convert one Anthropic message into one or more chat messages."""
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if isinstance(content, str):
        out.append({"role": role, "content": content})
        return

    # content is a list of blocks. Group text/image parts into a single chat
    # message, while tool_use/tool_result blocks become their own messages.
    parts = []
    tool_calls = []

    def flush_parts():
        if not parts:
            return
        # Collapse a pure-text part list into a plain string.
        if all(p.get("type") == "text" for p in parts):
            text = "".join(p["text"] for p in parts)
            out.append({"role": role, "content": text})
        else:
            out.append({"role": role, "content": list(parts)})
        parts.clear()

    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            parts.append({"type": "text", "text": str(block)})
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            part = _image_part(block.get("source"))
            if part:
                parts.append(part)
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
        elif btype == "tool_result":
            # A tool_result closes any pending assistant/user content first.
            flush_parts()
            out.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id"),
                "content": _stringify_tool_result_content(block.get("content", "")),
            })
        elif btype in ("thinking", "redacted_thinking"):
            continue  # USAi chat API has no thinking equivalent
        else:
            parts.append({"type": "text", "text": f"[unsupported block omitted: {btype}]"})

    if tool_calls:
        # tool_use blocks come from the assistant; attach any leading text.
        text = "".join(p["text"] for p in parts if p.get("type") == "text")
        parts.clear()
        out.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
    else:
        flush_parts()


def _convert_tools(tools):
    """Anthropic tool definitions -> chat tool definitions (client tools only)."""
    chat_tools = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        # Server tools (web_search, bash, code_execution, ...) carry a `type`
        # and no input_schema; USAi can't run them, so skip.
        if "input_schema" not in t:
            continue
        chat_tools.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })
    return chat_tools


def _convert_tool_choice(tc):
    if not isinstance(tc, dict):
        return None
    kind = tc.get("type")
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    if kind == "any":
        return "required"
    if kind == "tool" and tc.get("name"):
        return {"type": "function", "function": {"name": tc["name"]}}
    return None


def messages_to_chat(body):
    """Translate an Anthropic Messages request into a chat/completions dict."""
    messages = []
    system = body.get("system")
    if system:
        text = _system_to_text(system)
        if text:
            messages.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        _convert_message(msg, messages)

    model = body.get("model", "")
    chat = {
        "model": MODEL_MAP.get(model, model),
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }

    if body.get("max_tokens") is not None:
        chat["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        chat["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        chat["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        chat["stop"] = body["stop_sequences"]

    tools = _convert_tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
    tool_choice = _convert_tool_choice(body.get("tool_choice"))
    if tool_choice is not None:
        chat["tool_choice"] = tool_choice

    return chat


# ---------------------------------------------------------------------------
# Upstream calls
# ---------------------------------------------------------------------------
def upstream_request(chat_body, api_key, stream):
    url = UPSTREAM.rstrip("/") + "/chat/completions"
    data = json.dumps(chat_body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "text/event-stream" if stream else "application/json")
    return urllib.request.urlopen(req, timeout=600)


# ---------------------------------------------------------------------------
# Response translation: Chat Completions  ->  Anthropic Messages
# ---------------------------------------------------------------------------
def _usage(usage):
    return {
        "input_tokens": (usage or {}).get("prompt_tokens", 0),
        "output_tokens": (usage or {}).get("completion_tokens", 0),
    }


def build_message_object(msg_id, model, text, tool_calls, finish_reason, usage):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for tc in tool_calls:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex),
            "name": fn.get("name", ""),
            "input": args,
        })
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": STOP_REASON.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": _usage(usage),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _api_key(self):
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        xkey = self.headers.get("x-api-key")
        if xkey:
            return xkey
        return API_KEY

    def _route(self):
        """Path without query string or trailing slash, e.g. `/v1/messages`."""
        return self.path.split("?", 1)[0].rstrip("/")

    # -- routing --------------------------------------------------------
    def do_HEAD(self):
        # Claude Code sends a `HEAD /` connectivity probe on startup.
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self._route().endswith("/models"):
            self._proxy_models()
            return
        self._send_json(200, {"status": "ok"})

    def do_POST(self):
        if not self._route().endswith("/messages"):
            self._send_json(404, {"type": "error",
                                  "error": {"type": "not_found_error", "message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            self._send_json(400, {"type": "error",
                                  "error": {"type": "invalid_request_error", "message": f"bad json: {e}"}})
            return

        stream = bool(body.get("stream", False))
        model = body.get("model", "")
        chat_body = messages_to_chat(body)

        try:
            up = upstream_request(chat_body, self._api_key(), stream)
        except urllib.error.HTTPError as e:
            self._send_raw(e.code, e.read(), "application/json")
            return
        except Exception as e:
            log("upstream exception", repr(e))
            self._send_json(502, {"type": "error",
                                  "error": {"type": "api_error", "message": f"upstream: {e}"}})
            return

        if stream:
            self._stream(up, model)
        else:
            self._non_stream(up, model)

    # -- model discovery ------------------------------------------------
    def _proxy_models(self):
        try:
            url = UPSTREAM.rstrip("/") + "/models"
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {self._api_key()}")
            with urllib.request.urlopen(req, timeout=60) as r:
                upstream = json.loads(r.read())
        except urllib.error.HTTPError as e:
            self._send_raw(e.code, e.read(), "application/json")
            return
        except Exception as e:
            log("models error", repr(e))
            self._send_json(200, {"data": []})
            return

        data = upstream.get("data", []) if isinstance(upstream, dict) else []
        models = []
        for m in data:
            slug = m.get("id") if isinstance(m, dict) else None
            if not slug or slug not in USAI_TO_ANTHROPIC:
                continue
            anthropic_id = USAI_TO_ANTHROPIC[slug]
            models.append({
                "type": "model",
                "id": anthropic_id,
                "display_name": DISPLAY_NAMES.get(anthropic_id, anthropic_id),
                "created_at": "2025-01-01T00:00:00Z",
            })
        self._send_json(200, {"data": models, "has_more": False, "first_id": None, "last_id": None})

    # -- non streaming --------------------------------------------------
    def _non_stream(self, up, model):
        data = json.loads(up.read())
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        obj = build_message_object(
            "msg_" + uuid.uuid4().hex,
            model,
            msg.get("content") or "",
            msg.get("tool_calls") or [],
            choice.get("finish_reason"),
            data.get("usage"),
        )
        self._send_json(200, obj)

    # -- streaming ------------------------------------------------------
    def _stream(self, up, model):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # No Content-Length is sent for the event stream, so signal end-of-
        # response by closing the connection when it completes.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        msg_id = "msg_" + uuid.uuid4().hex

        def emit(event, obj):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(obj)}\n\n".encode("utf-8"))
            self.wfile.flush()

        # Anthropic block indices are contiguous and assigned as blocks open.
        text_index = None            # index of the open text block, if any
        tools = {}                   # chat tool_call index -> {block_index, ...}
        next_index = 0
        finish_reason = None
        usage = None

        try:
            emit("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })

            for line in up:
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:"):].strip()
                if payload == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta", {})

                # text
                dc = delta.get("content")
                if dc:
                    if text_index is None:
                        text_index = next_index
                        next_index += 1
                        emit("content_block_start", {
                            "type": "content_block_start", "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    emit("content_block_delta", {
                        "type": "content_block_delta", "index": text_index,
                        "delta": {"type": "text_delta", "text": dc},
                    })

                # tool calls
                for tcd in delta.get("tool_calls") or []:
                    idx = tcd.get("index", 0)
                    slot = tools.get(idx)
                    fn = tcd.get("function", {})
                    if slot is None:
                        # First chunk for this tool call: open a tool_use block.
                        block_index = next_index
                        next_index += 1
                        slot = {"block_index": block_index}
                        tools[idx] = slot
                        emit("content_block_start", {
                            "type": "content_block_start", "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tcd.get("id") or ("toolu_" + uuid.uuid4().hex),
                                "name": fn.get("name", ""),
                                "input": {},
                            },
                        })
                    args = fn.get("arguments")
                    if args:
                        emit("content_block_delta", {
                            "type": "content_block_delta", "index": slot["block_index"],
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        })

            # close open blocks
            if text_index is not None:
                emit("content_block_stop", {"type": "content_block_stop", "index": text_index})
            for idx in sorted(tools, key=lambda i: tools[i]["block_index"]):
                emit("content_block_stop",
                     {"type": "content_block_stop", "index": tools[idx]["block_index"]})

            emit("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": STOP_REASON.get(finish_reason, "end_turn"),
                          "stop_sequence": None},
                "usage": {"output_tokens": (usage or {}).get("completion_tokens", 0)},
            })
            emit("message_stop", {"type": "message_stop"})
        except (BrokenPipeError, ConnectionResetError):
            log("client disconnected during stream")
        finally:
            up.close()

    # -- helpers --------------------------------------------------------
    def _send_json(self, code, obj):
        self._send_raw(code, json.dumps(obj).encode(), "application/json")

    def _send_raw(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    if not UPSTREAM:
        sys.exit("error: USAI_BASE_URL environment variable is required")
    if not API_KEY:
        sys.exit("error: USAI_API_KEY environment variable is required")
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    log(f"listening on http://{LISTEN_HOST}:{LISTEN_PORT}  -> {UPSTREAM}")
    log(f"point Claude Code at http://{LISTEN_HOST}:{LISTEN_PORT} via ANTHROPIC_BASE_URL")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")


if __name__ == "__main__":
    main()
