# flense

A lightweight reverse proxy that cuts your AI API costs by compressing large payloads before they reach expensive frontier models, with no changes to your application code.

---

## The Problem

Frontier AI models (Claude Opus, GPT-4, Gemini Ultra) are billed per token. Most large requests are padded with stuff the model does not need to see in full: entire source files, verbose docs, repetitive boilerplate. You pay full price for all of it.

## How Flense Helps

Flense sits between your application and the AI API. It intercepts outgoing requests, compresses heavy payloads using static code analysis, and forwards an optimised version to the frontier model. Your app sees no difference. Just a cheaper bill.

```
Your App  ->  localhost:2912 (flense)  ->  api.anthropic.com
```

No model calls. No quality trade-off for compression. Just fewer tokens.

---

## Quick Start

```bash
pip install flense
flense start
```

Then point your SDK at flense with the provider prefix:

```python
# Anthropic
client = Anthropic(base_url="http://localhost:2912/anthropic")

# OpenAI
client = OpenAI(base_url="http://localhost:2912/openai")
```

---

## How It Works

### Bulk-Reader
When a payload contains large code files or documents, flense compresses them using [Tree-sitter](https://tree-sitter.github.io/tree-sitter/), a fast, error-tolerant code parser that supports 100+ languages.

Instead of sending the full file, flense extracts:
- Class and function signatures
- Method names, parameters, return types
- Line number anchors so the model knows where everything came from

Function bodies, comments, and docstrings are stripped. The frontier model gets a skeleton. Enough to reason accurately, at a fraction of the token cost.

**No secondary model. No tokens spent on compression. Tree-sitter runs locally.**

Fallback chain: Tree-sitter -> Universal Ctags -> regex heuristics

### Code-Writer Bypass
For repetitive code generation tasks (scaffolding tests, mapping schemas), flense can route the request directly to a cheaper cloud model and write the output straight to disk, bypassing the frontier model entirely.

Triggered explicitly via a request header:
```python
headers={"X-Flense-Strategy": "code-writer"}
```

---

## Telemetry

Flense appends savings data to every response header. Readable by your app, or visible in the terminal dashboard:

```
X-Flense-Tokens-Saved: 18400
X-Flense-Est-Savings: $0.552
X-Flense-Compression-Time: 42ms
X-Flense-Strategy: bulk-reader
```

Run `flense tui` for a live terminal dashboard showing real-time savings across your session.

---

## Configuration

```toml
# flense.toml — all fields optional

[server]
port = 2912
headless = false

[compression]
threshold = 5000      # compress payloads above this token count
strategy = "auto"     # auto | ast | ctags | passthrough

[providers.anthropic]
upstream = "https://api.anthropic.com"

[providers.openai]
upstream = "https://api.openai.com"

[code_writer]
model = "claude-haiku-4-5"
fallback = "gpt-4o-mini"
output_dir = "./generated"
```

---

## Provider Support

Flense supports **Anthropic** and **OpenAI** at v1. Route by URL prefix — each provider gets its own adapter with correct token counting and pricing:

```
localhost:2912/anthropic/v1/messages         →  api.anthropic.com
localhost:2912/openai/v1/chat/completions    →  api.openai.com
```

Adding a new provider is a single adapter file and route registration — nothing else changes.

---

## License

Copyright (c) 2026 Anand. All rights reserved. See [LICENSE](LICENSE).
