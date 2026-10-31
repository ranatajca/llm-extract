"""A keyword/regex 'provider' that speaks the same JSON protocol as an LLM.

It gives the eval harness an offline floor to beat and lets CI run end to end
without API keys. It only reads the text between <<< and >>> in the prompt.
"""

from __future__ import annotations

import json
import re

from .providers import Message

_RULES = {
    "category": [
        ("billing", r"charg|invoice|refund|billed|payment|subscription|price"),
        ("shipping", r"deliver|shipping|package|courier|tracking|arriv"),
        ("bug", r"crash|error|bug|broken|not working|fails|freez"),
        ("account", r"password|log ?in|sign ?in|account|2fa|locked"),
        ("feature_request", r"would be (great|nice)|please add|feature|wish|could you add"),
    ],
    "sentiment": [
        ("negative", r"angry|terrible|unacceptable|frustrat|worst|disappoint|ridiculous"),
        ("positive", r"love|great|thanks so much|awesome|amazing|happy"),
    ],
}
_PRODUCTS = ["mobile app", "web dashboard", "api", "desktop app", "smart speaker", "headphones"]


def _first(rules, text, default):
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    return default


class KeywordBaseline:
    def complete(self, messages: list[Message]) -> str:
        prompt = messages[1]["content"]
        text = prompt.split("<<<", 1)[-1].rsplit(">>>", 1)[0].strip()
        low = text.lower()
        urgent = re.search(r"\b(down|outage|asap|immediately|urgent)\b", low)
        high = re.search(r"\b(twice|double|cannot|can't|again)\b", low)
        order = re.search(r"\b([A-Z]{2}-\d{4,8})\b", text)
        product = next((p for p in _PRODUCTS if p in low), None)
        return json.dumps(
            {
                "category": _first(_RULES["category"], low, "other"),
                "priority": "urgent" if urgent else "high" if high else "medium",
                "sentiment": _first(_RULES["sentiment"], low, "neutral"),
                "product": product,
                "order_id": order.group(1) if order else None,
                "refund_requested": bool(re.search(r"refund|money back", low)),
                "summary": text.split(".")[0][:140],
            }
        )
