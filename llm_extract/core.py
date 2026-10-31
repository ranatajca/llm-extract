"""Schema-first extraction: prompt -> parse -> validate -> repair loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .providers import Message, Provider

T = TypeVar("T", bound=BaseModel)

SYSTEM = (
    "You extract structured data from text. Reply with a single JSON object that "
    "validates against the given JSON Schema. Use null for fields the text does not "
    "support. Never invent values. No prose, no markdown."
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass
class Extraction(Generic[T]):
    value: T | None
    attempts: int
    errors: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None


class ExtractionFailed(RuntimeError):
    def __init__(self, result: Extraction):
        super().__init__(f"extraction failed after {result.attempts} attempts: {result.errors[-1]}")
        self.result = result


def parse_json(reply: str) -> dict:
    """Tolerates code fences and leading/trailing chatter around one JSON object."""
    text = _FENCE.sub("", reply.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def build_messages(text: str, schema: type[BaseModel]) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"JSON Schema:\n{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
                f"Text:\n<<<\n{text}\n>>>"
            ),
        },
    ]


def format_error(err: Exception) -> str:
    if isinstance(err, ValidationError):
        parts = [
            f"- {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in err.errors()
        ]
        return "Validation errors:\n" + "\n".join(parts)
    return f"Could not parse JSON: {err}"


def extract(
    text: str,
    schema: type[T],
    provider: Provider,
    max_repairs: int = 2,
    raise_on_failure: bool = False,
) -> Extraction[T]:
    """Ask the model for JSON, validate it, and feed precise errors back for repair.

    Each repair turn shows the model its previous answer plus the exact pydantic
    errors, which fixes most failures (wrong enum casing, missing fields, stray
    prose) in one extra call instead of a blind retry.
    """
    messages = build_messages(text, schema)
    result: Extraction[T] = Extraction(value=None, attempts=0)
    for _ in range(max_repairs + 1):
        result.attempts += 1
        reply = provider.complete(messages)
        result.raw.append(reply)
        try:
            result.value = schema.model_validate(parse_json(reply))
            return result
        except (json.JSONDecodeError, ValidationError) as err:
            msg = format_error(err)
            result.errors.append(msg)
            messages = [
                *messages,
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"{msg}\n\nReturn the corrected JSON object only."},
            ]
    if raise_on_failure:
        raise ExtractionFailed(result)
    return result
