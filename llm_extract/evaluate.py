"""Field-level evaluation of an extractor against a gold JSONL file.

python -m llm_extract.evaluate --provider baseline
python -m llm_extract.evaluate --provider anthropic   # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .core import extract
from .schemas import SupportTicket

FREE_TEXT = {"summary"}  # judged by humans / LLM-as-judge, not exact match


def _norm(v):
    if hasattr(v, "value"):
        v = v.value
    return v.strip().lower() if isinstance(v, str) else v


def run(provider, path: str | Path, max_repairs: int = 2) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    hits: dict[str, list[int]] = defaultdict(list)
    attempts, valid, failures = [], 0, []
    for row in rows:
        res = extract(row["text"], SupportTicket, provider, max_repairs=max_repairs)
        attempts.append(res.attempts)
        if not res.ok:
            failures.append({"id": row["id"], "error": res.errors[-1]})
            for f in row["gold"]:
                if f not in FREE_TEXT:
                    hits[f].append(0)
            continue
        valid += 1
        pred = res.value.model_dump()
        for f, gold in row["gold"].items():
            if f not in FREE_TEXT:
                hits[f].append(int(_norm(pred.get(f)) == _norm(gold)))
    per_field = {f: mean(v) for f, v in sorted(hits.items())}
    return {
        "n": len(rows),
        "valid_rate": valid / len(rows),
        "mean_attempts": mean(attempts),
        "field_accuracy": per_field,
        "macro_accuracy": mean(per_field.values()),
        "failures": failures,
    }


def _provider(name: str):
    if name == "baseline":
        from .baseline import KeywordBaseline

        return KeywordBaseline()
    if name == "anthropic":
        from .providers import AnthropicProvider

        return AnthropicProvider()
    if name == "openai":
        from .providers import OpenAIProvider

        return OpenAIProvider()
    raise SystemExit(f"unknown provider {name}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="baseline", choices=["baseline", "anthropic", "openai"])
    p.add_argument("--data", default=str(Path(__file__).parents[1] / "data" / "tickets.jsonl"))
    p.add_argument("--max-repairs", type=int, default=2)
    args = p.parse_args(argv)
    report = run(_provider(args.provider), args.data, args.max_repairs)
    print(
        f"provider={args.provider}  n={report['n']}  valid={report['valid_rate']:.0%}  "
        f"attempts/item={report['mean_attempts']:.2f}\n"
    )
    print("| field | accuracy |\n|---|---|")
    for f, acc in report["field_accuracy"].items():
        print(f"| {f} | {acc:.2f} |")
    print(f"| **macro** | **{report['macro_accuracy']:.2f}** |")


if __name__ == "__main__":
    main()
