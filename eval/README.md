# Flense Eval Harness

Measures whether flense's compression keeps the information a model needs to answer
a question — the project's core open question, made testable.

## Running it

```bash
# Free, offline: does the needed info survive compression?
python eval/harness.py

# Real model: also score a live model's answer (costs money, needs a key)
ANTHROPIC_API_KEY=... python eval/harness.py --real
OPENAI_API_KEY=...    python eval/harness.py --real --provider openai --model gpt-4o-mini
```

## What it measures

Each case under `cases/` is a snippet of code + a question + `must_retain`: the
strings the answer depends on. The harness runs the **real** flense pipeline over
the code, then checks whether those strings survive in the compressed context.

- **retention mode** (default): fraction of `must_retain` strings still present in
  the compressed context, plus token savings. No API key, no network, no cost.
- **real mode** (`--real`): additionally sends the compressed context + question to
  a live model and scores the model's answer against `must_retain`.

## Representative result (retention mode)

```
category        saved   retention
api-surface      20%      100%
structure        25%      100%
deep-body        41%        0%   <- compression drops info these questions need
```

## How to read it

- **Structure / API-surface questions** ("what methods exist?", "what does this
  function take?") retain 100% of the needed info while saving tokens — compression
  is essentially free here.
- **Deep-body questions** ("why is there a bug in this function?") lose the detail,
  because flense strips function bodies down to signatures. This is the failure mode
  to respect, and it is exactly why the classifier is conservative and why
  `X-Flense-Strategy: passthrough` exists.

The takeaway is a **quality/cost curve**, not a yes/no: use compression freely for
exploratory, structural context; disable it when the task needs verbatim bodies.

## Adding cases

Drop a JSON file in `cases/`:

```json
{
  "id": "unique_name",
  "category": "structure | api-surface | deep-body | ...",
  "needs_body": false,
  "language": "python",
  "question": "...",
  "must_retain": ["strings the answer depends on"],
  "code": "...source code..."
}
```
