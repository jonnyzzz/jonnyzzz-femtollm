# femtollm

Minimal LLM proxy router with protocol conversion and fallback support. Routes requests to vLLM and other OpenAI-compatible backends.

## Features

- **Model-based routing** — regex patterns match model names to backends
- **Protocol conversion** — Anthropic Messages API to OpenAI Chat Completions
- **Fallback** — multiple backends per model, tries in order on 5xx errors
- **Streaming** — SSE passthrough for OpenAI streaming responses
- **Zero dependencies** — pure Go, single static binary

## Usage

```bash
# Build
go build ./cmd/femtollm

# Run
cp config.example.json config.json
# Edit config.json with your backends
./femtollm -config config.json
```

## Configuration

```json
{
  "listen": ":8080",
  "backends": [
    {
      "name": "vllm-local",
      "pattern": "(?i)qwen.*coder",
      "url": "http://localhost:8000",
      "model": "Qwen/Qwen3-Coder-FP8"
    },
    {
      "name": "fallback",
      "pattern": ".*",
      "url": "http://localhost:8000"
    }
  ]
}
```

## Endpoints

| Endpoint | Protocol | Description |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | Chat completions (streaming + non-streaming) |
| `POST /v1/messages` | Anthropic | Messages API (converted to OpenAI internally) |
| `GET /v1/models` | OpenAI | List advertised models |
| `GET /health` | — | Health check |

## Deploy with stevedore

```bash
stevedore repo add femtollm git@github.com:jonnyzzz/jonnyzzz-femtollm.git --branch main
stevedore deploy sync femtollm && stevedore deploy up femtollm
```

Place `config.json` in the stevedore data volume (`${STEVEDORE_DATA}/config.json`).
