# usai-claude-proxy

A small local proxy that lets [Claude Code](https://www.anthropic.com/claude-code)
talk to a **USAi** LLM gateway.

## Why this exists

- Claude Code speaks the **Anthropic Messages API** (`POST /v1/messages`).
- USAi only exposes the OpenAI-style **Chat Completions API**
  (`/chat/completions`); it has no `/v1/messages` endpoint.

This proxy bridges the gap. It listens on `127.0.0.1:7878`, accepts the
Anthropic Messages requests Claude Code sends, translates them into USAi
`chat/completions` calls, and translates the reply back into Anthropic Messages
responses (including the streaming SSE event sequence). It also serves a
`/v1/models` endpoint so Claude Code's `/model` picker can list the Claude
models.

Only the features USAi supports are translated: text, system prompts, images,
tools/tool-use, and streaming. Anthropic-only features that USAi's chat API has
no equivalent for -- **extended thinking**, documents, web search, citations,
and prompt caching -- are dropped.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) (`brew install uv`)
- `USAI_BASE_URL` and `USAI_API_KEY` set in your environment (both required)

The proxy itself is pure Python stdlib (no third-party dependencies).

## Starting the proxy

```bash
cd ~/code/usai-claude-proxy
export USAI_BASE_URL=https://<your-usai-host>/api/v1
export USAI_API_KEY=<your-key>
uv run usai-claude-proxy
```

It prints:

```
[proxy] listening on http://127.0.0.1:7878  -> https://<your-usai-host>/api/v1
[proxy] point Claude Code at http://127.0.0.1:7878 via ANTHROPIC_BASE_URL
```

To run it in the background:

```bash
cd ~/code/usai-claude-proxy
nohup uv run usai-claude-proxy >/tmp/usai-claude-proxy.log 2>&1 &
```

Stop a background instance with:

```bash
pkill -f usai-claude-proxy
```

Quick health check:

```bash
curl -s http://127.0.0.1:7878/         # -> {"status": "ok"}
```

### Configuration (env vars)

| Variable        | Default       | Purpose                            |
| --------------- | ------------- | ---------------------------------- |
| `USAI_BASE_URL` | (required)    | Upstream USAi base URL             |
| `USAI_API_KEY`  | (required)    | Bearer token sent to USAi          |
| `PROXY_HOST`    | `127.0.0.1`   | Address the proxy binds to         |
| `PROXY_PORT`    | `7878`        | Port the proxy binds to            |

The proxy exits with an error if `USAI_BASE_URL` or `USAI_API_KEY` is missing.

The API key resolution prefers a client-supplied `Authorization: Bearer` or
`x-api-key` header (which is what Claude Code sends) and only falls back to the
proxy's own `USAI_API_KEY` environment variable when no such header is present.

## Claude Code configuration

Point Claude Code at the proxy with environment variables:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:7878
export ANTHROPIC_AUTH_TOKEN=$USAI_API_KEY

# Optional: list the USAi Claude models in the `/model` picker.
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1

# Recommended: USAi's chat API can't represent Claude Code's pre-release beta
# capabilities, so disabling them avoids occasional 400 errors.
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
```

Notes:

- `ANTHROPIC_BASE_URL` points at **the proxy**, not USAi directly.
- `ANTHROPIC_AUTH_TOKEN` is forwarded to USAi as the `Authorization: Bearer`
  header.
- **Extended thinking is unavailable** through USAi. The proxy strips `thinking`
  configuration from requests and never emits thinking blocks.

## Model mapping

Claude Code sends hyphenated Anthropic model ids; USAi expects its own
underscore slugs. The proxy maps between them, and `/v1/models` advertises the
Anthropic-style ids so the picker shows familiar names. Only Anthropic models
are supported.

| Anthropic id (Claude Code) | USAi slug           | Display name       |
| -------------------------- | ------------------- | ------------------ |
| `claude-opus-4-8`          | `claude_4_8_opus`   | Claude Opus 4.8    |
| `claude-opus-4-7`          | `claude_4_7_opus`   | Claude Opus 4.7    |
| `claude-opus-4-5`          | `claude_4_5_opus`   | Claude Opus 4.5    |
| `claude-sonnet-4-6`        | `claude_4_6_sonnet` | Claude Sonnet 4.6  |
| `claude-sonnet-4-5`        | `claude_4_5_sonnet` | Claude Sonnet 4.5  |
| `claude-haiku-4-5`         | `claude_4_5_haiku`  | Claude Haiku 4.5   |

Unknown model ids pass through unchanged, so a manually selected USAi slug still
works.

## Everyday usage

1. Start the proxy (see above). It must be running whenever you use Claude Code.
2. Set the `ANTHROPIC_*` environment variables.
3. Run `claude` as normal.

## How the translation works

`main.py`:

- Converts the Anthropic `system` field and `messages` content blocks into chat
  `messages`: `text` stays text, `image` becomes an `image_url` data URI, an
  assistant `tool_use` block becomes `tool_calls`, and a user `tool_result`
  block becomes a `tool` message. Non-text tool-result content is flattened to
  text with a bracketed note, and `thinking` blocks are dropped.
- Converts Anthropic `tools` (`{name, description, input_schema}`) into chat
  `tools` (`{type: function, function: {...}}`), skipping server tools USAi
  can't run.
- Re-emits streamed chat deltas as the Anthropic SSE event sequence:
  `message_start`, then per content block a `content_block_start` /
  `content_block_delta` (`text_delta` or `input_json_delta`) /
  `content_block_stop`, then `message_delta` and `message_stop`.
- Serves `/v1/models` by fetching USAi's list, keeping the Claude models, and
  reshaping them to Anthropic-style ids and display names.
