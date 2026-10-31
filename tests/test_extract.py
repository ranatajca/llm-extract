import json
from pathlib import Path

import pytest

from llm_extract import ExtractionFailed, ScriptedProvider, extract, parse_json
from llm_extract.baseline import KeywordBaseline
from llm_extract.evaluate import run
from llm_extract.schemas import Category, Priority, SupportTicket

GOOD = {
    "category": "billing",
    "priority": "high",
    "sentiment": "negative",
    "product": None,
    "order_id": "PX-48213",
    "refund_requested": True,
    "summary": "Customer was charged twice and wants a refund.",
}
TEXT = "I was charged twice. Order PX-48213. Refund please."


def test_parse_json_handles_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": 1} hope that helps') == {"a": 1}


def test_valid_first_try():
    res = extract(TEXT, SupportTicket, ScriptedProvider([json.dumps(GOOD)]))
    assert res.ok and res.attempts == 1
    assert res.value.category is Category.billing


def test_enum_casing_is_normalised_without_a_repair():
    reply = json.dumps({**GOOD, "category": "Billing", "priority": "HIGH"})
    res = extract(TEXT, SupportTicket, ScriptedProvider([reply]))
    assert res.attempts == 1 and res.value.priority is Priority.high


def test_repair_loop_feeds_back_precise_errors():
    bad = json.dumps({**GOOD, "order_id": "48213", "priority": "critical"})
    provider = ScriptedProvider([bad, json.dumps(GOOD)])
    res = extract(TEXT, SupportTicket, provider)
    assert res.ok and res.attempts == 2
    feedback = provider.calls[1][-1]["content"]
    assert "order_id" in feedback and "priority" in feedback
    assert provider.calls[1][-2] == {"role": "assistant", "content": bad}


def test_unparseable_then_fixed():
    res = extract(TEXT, SupportTicket, ScriptedProvider(["not json at all", json.dumps(GOOD)]))
    assert res.ok and "Could not parse JSON" in res.errors[0]


def test_gives_up_after_max_repairs():
    provider = ScriptedProvider(["nope"] * 3)
    res = extract(TEXT, SupportTicket, provider, max_repairs=2)
    assert not res.ok and res.attempts == 3
    with pytest.raises(ExtractionFailed):
        extract(
            TEXT,
            SupportTicket,
            ScriptedProvider(["nope"] * 2),
            max_repairs=1,
            raise_on_failure=True,
        )


def test_baseline_eval_runs_end_to_end():
    report = run(KeywordBaseline(), Path(__file__).parents[1] / "data" / "tickets.jsonl")
    assert report["n"] == 16 and report["valid_rate"] == 1.0
    assert 0.4 < report["macro_accuracy"] <= 1.0
