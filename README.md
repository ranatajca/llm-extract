# llm-extract

Turn messy text into **validated, typed objects** with an LLM, and measure how well it works.

```python
from llm_extract import extract, AnthropicProvider
from llm_extract.schemas import SupportTicket

res = extract(
    "I was charged twice this month. Order PX-48213, please refund the duplicate.",
    SupportTicket,
    AnthropicProvider(),
)
res.value  # example output
# SupportTicket(category=<Category.billing>, priority=<Priority.high>, sentiment=<Sentiment.negative>,
#               order_id='PX-48213', refund_requested=True, summary='Customer was double charged...')
res.attempts  # 1
```

## Why

`json.loads(model_output)` works in demos and fails in production: code fences, chatty preambles, `"High"` instead of `"high"`, an order ID in the wrong format, a missing required field. Blind retries waste calls on the same mistake.

This library makes the **schema the contract**:

1. **Prompt** with the Pydantic model's JSON Schema.
2. **Parse** tolerantly (strips fences and surrounding prose).
3. **Validate** with Pydantic, normalising cheap mistakes like enum casing *before* they count as failures.
4. **Repair**: on failure, send the model its own answer plus the *exact* field-level errors (`order_id: String should match pattern '^[A-Z]{2}-\d{4,8}$'`) and ask for a fix. Bounded by `max_repairs`.
5. **Measure**: field-level accuracy, valid-output rate and attempts per item against a gold set.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q                                          # runs offline
python -m llm_extract.evaluate --provider baseline # offline keyword baseline

pip install -e ".[anthropic]" && export ANTHROPIC_API_KEY=...
python -m llm_extract.evaluate --provider anthropic
```

## Evaluation

`data/tickets.jsonl` holds 16 hand-labelled support tickets. `summary` is free text and is excluded from exact-match scoring.

Offline keyword/regex baseline, the floor an LLM has to beat:

| field | accuracy |
|---|---|
| category | 0.62 |
| order_id | 1.00 |
| priority | 0.56 |
| product | 1.00 |
| refund_requested | 1.00 |
| sentiment | 0.62 |
| **macro** | **0.80** |

The baseline nails the pattern-shaped fields (IDs, product names, "refund") and struggles where meaning matters: priority ("data leak" is urgent with no urgent keyword), sentiment and sarcasm, and category ambiguity (a wrong-colour item counts as shipping, not billing). Those are the fields where an LLM earns its cost. Run the `anthropic` or `openai` provider to fill in the comparison on your own key.

## Design notes

- **Providers are one method** (`complete(messages) -> str`), so the loop is unit-tested with `ScriptedProvider`, which replays canned replies. The tests cover fences, chatter, enum casing, a pattern violation fixed in one repair, unparseable output and exhausting the retry budget.
- **Normalise, don't fail**: `field_validator(mode="before")` lower-cases enums so casing never costs an API call.
- **Precise feedback beats blind retries**: the repair turn names the failing field and rule, which is usually enough to fix it in one extra call.
- **Bring your own schema**: any Pydantic v2 model works. `schemas.py` is just an example.

## Layout

```
llm_extract/
  core.py       extract(), tolerant JSON parsing, repair loop
  providers.py  Anthropic, OpenAI, Scripted (tests)
  schemas.py    example SupportTicket schema
  baseline.py   offline keyword baseline that speaks the same protocol
  evaluate.py   field-level eval harness + CLI
data/tickets.jsonl
tests/
```

MIT licensed.
