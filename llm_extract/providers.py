"""Model providers behind a one-method interface, so the extraction loop is testable."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class Provider(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


class AnthropicProvider:
    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        import anthropic  # lazy: only needed when used

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("LLM_EXTRACT_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def complete(self, messages: list[Message]) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=chat
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAIProvider:
    def __init__(self, model: str | None = None):
        import openai  # lazy

        self.client = openai.OpenAI()
        self.model = model or os.getenv("LLM_EXTRACT_MODEL", "gpt-4o-mini")

    def complete(self, messages: list[Message]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content or ""


class ScriptedProvider:
    """Replays canned replies in order. Used in tests to exercise the repair loop."""

    def __init__(self, replies: Iterable[str]):
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        if not self.replies:
            raise RuntimeError("ScriptedProvider ran out of replies")
        return self.replies.pop(0)
