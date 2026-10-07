#!/usr/bin/env python3
"""Flense compression eval harness.

Answers the project's core open question: *when flense compresses code context,
is the information the model needs still there?*

Two modes:

  retention (default, free, offline)
      For each case, run the REAL flense compression pipeline over the code,
      then check whether the strings the question depends on (``must_retain``)
      survive in the compressed context. Reports retention and token savings
      per category. No API key, no network, no cost.

  real (--real, costs money, needs a key)
      Additionally send the compressed context + question to a live model and
      score the model's ANSWER against ``must_retain``. Compares answer quality
      with compressed vs. full context. Requires ANTHROPIC_API_KEY (or
      OPENAI_API_KEY with --provider openai).

Usage:
    python eval/harness.py                 # retention mode
    python eval/harness.py --real          # real-model mode (Anthropic)
    python eval/harness.py --real --provider openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the flense package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flense.compression import compress_payload  # noqa: E402

CASES_DIR = Path(__file__).resolve().parent / "cases"


@dataclass
class CaseResult:
    case_id: str
    category: str
    needs_body: bool
    was_compressed: bool
    tokens_before: int
    tokens_after: int
    retained: int
    total_signals: int
    model_score: float | None = None  # real mode only

    @property
    def retention(self) -> float:
        return self.retained / self.total_signals if self.total_signals else 1.0

    @property
    def savings_pct(self) -> float:
        if not self.tokens_before:
            return 0.0
        return 100.0 * (self.tokens_before - self.tokens_after) / self.tokens_before


def load_cases() -> list[dict]:
    cases = []
    for path in sorted(CASES_DIR.glob("*.json")):
        with open(path) as f:
            cases.append(json.load(f))
    return cases


def _build_body(case: dict) -> bytes:
    """Build a provider-style request body with the case's code in a fence."""
    lang = case.get("language", "")
    code = case["code"]
    content = (
        f"File: src/module.{lang or 'txt'}\n"
        f"```{lang}\n{code}```\n"
        f"Question: {case['question']}"
    )
    return json.dumps(
        {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": content}],
        }
    ).encode()


def _compressed_context(case: dict) -> tuple[str, object]:
    """Run the real flense pipeline; return (compressed_text, result)."""
    body = _build_body(case)
    # Force AST strategy and a low threshold so compression is exercised.
    result = compress_payload(
        body=body,
        provider="anthropic",
        threshold=50,
        header_strategy="ast",
    )
    payload = json.loads(result.compressed_body)
    text = payload["messages"][0]["content"]
    return text, result


def run_retention(case: dict) -> CaseResult:
    text, result = _compressed_context(case)
    must_retain = case.get("must_retain", [])
    retained = sum(1 for s in must_retain if s in text)
    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        needs_body=case.get("needs_body", False),
        was_compressed=result.was_compressed,
        tokens_before=result.tokens_before,
        tokens_after=result.tokens_after,
        retained=retained,
        total_signals=len(must_retain),
    )


def run_real(case: dict, provider: str, model: str) -> CaseResult:
    """Retention + a live model answer scored against must_retain."""
    import httpx

    base = run_retention(case)
    text, _ = _compressed_context(case)
    prompt = (
        f"{text}\n\nAnswer the question above using only the context provided. "
        f"Be concise."
    )

    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY not set")
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        answer = resp.json()["content"][0]["text"]
    else:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set")
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=60.0,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]

    must = case.get("must_retain", [])
    # Crude but real: how many expected facts appear in the model's answer.
    hits = sum(1 for s in must if s.split("(")[0].strip() in answer)
    base.model_score = hits / len(must) if must else 1.0
    return base


def print_report(results: list[CaseResult], real: bool) -> None:
    print()
    header = f"{'case':<28}{'category':<14}{'compressed':<12}{'saved':>7}{'retention':>11}"
    if real:
        header += f"{'answer':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = (
            f"{r.case_id:<28}{r.category:<14}"
            f"{('yes' if r.was_compressed else 'no'):<12}"
            f"{r.savings_pct:>6.0f}%{r.retention:>10.0%}"
        )
        if real:
            line += f"{(r.model_score or 0):>8.0%}"
        print(line)

    print()
    # Aggregate by category.
    cats: dict[str, list[CaseResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    print("By category:")
    for cat, rs in sorted(cats.items()):
        avg_sav = sum(r.savings_pct for r in rs) / len(rs)
        avg_ret = sum(r.retention for r in rs) / len(rs)
        note = ""
        if any(r.needs_body for r in rs) and avg_ret < 1.0:
            note = "  <- compression drops info these questions need"
        print(f"  {cat:<14} saved {avg_sav:>4.0f}%   retention {avg_ret:>4.0%}{note}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Flense compression eval harness")
    ap.add_argument("--real", action="store_true", help="also query a live model")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()

    cases = load_cases()
    if not cases:
        print("No cases found in", CASES_DIR)
        return 1

    results = [
        run_real(c, args.provider, args.model) if args.real else run_retention(c)
        for c in cases
    ]
    print_report(results, real=args.real)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
