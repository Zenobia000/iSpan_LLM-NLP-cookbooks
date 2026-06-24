# OpenAI SDK 2026 Migration Audit

Date: 2026-06-24

## Teaching Position

The course should teach **Responses API first** for new OpenAI examples:

- Use `client.responses.create()` as the default SDK entrypoint.
- Use `input` and `output_text` for basic text generation.
- Use hosted tools such as `file_search` / `web_search` through Responses.
- Use `previous_response_id` for simple continuation and Conversations API for longer-lived server-side conversation objects.
- Keep Chat Completions only as legacy literacy and migration material.

Chat Completions is still available for compatibility, but it should not be the main teaching surface for new 2026 project code. Assistants API should be removed from mainline examples except in migration notes because it is deprecated and scheduled for shutdown on 2026-08-26.

## Multimodal Model Currency (verified 2026)

`01-uncontrollability/04-multimodal.ipynb` was also refreshed to current model families:

- **Image generation**: `gpt-image-1` → `gpt-image-2` (flagship, Apr 2026: reasoning-powered, 4K, multilingual text rendering). Family also includes `gpt-image-1.5`, `gpt-image-1`, `gpt-image-1-mini`; DALL·E 2/3 are previous-generation.
- **Text-to-speech**: `gpt-4o-mini-tts` (steerable, current); `tts-1` / `tts-1-hd` kept only as the early fixed-style comparison.
- **Realtime voice/text**: `gpt-realtime` (GA) / `gpt-realtime-mini` via WebSocket / WebRTC / SIP — added a runnable text→text example (`AsyncOpenAI().realtime.connect`, session/response events); switch `output_modalities` to `["audio"]` for voice, prefer WebRTC for production agents. Model controlled by `OPENAI_REALTIME_MODEL` (default `gpt-realtime`).
- **Speech-to-text**: `gpt-4o-transcribe` / `gpt-4o-mini-transcribe` (lower WER than the original Whisper).

## Completed in This Pass

**Status: migration complete.** 30 notebooks now use the Responses API as the main OpenAI
entrypoint and are enrolled in the static guardrails (`MODERN_OPENAI_NOTEBOOKS`):

- `01-uncontrollability`: 01-openai-api-intro, 02-sampling-and-uncertainty, 04-multimodal
- `02-intent-convergence`: 01-prompt-basics, 02-cot-reasoning, 03-prompt-chaining, 04-prompt-integration-usecase
- `03-structured-output`: 01-json-mode, 02-function-calling-basics, 03-structured-extraction, 04-classification-gradio
- `04-knowledge-rag`: 01-embedding, 02-vanilla-rag, 03-similarity-and-relevance, 04-vector-db-rag, 05-dynamic-few-shot, 06-advanced-rag, 07-pdf-parsing
- `05-agent-harness`: 01-function-calling-agents, 02-react-loop, 04-function-calling-rag, 05-shop-guardrails, 06-prompt-injection, 07-plugin-tools, 08-chatbot, 09-responses-api
- `06-multi-agent`: 03-cross-model-review
- `07-calibration-eval`: 01-rag-evaluation, 02-feedback-loop

**Migration techniques used**: text helpers → `client.responses.create(input=..., max_output_tokens=...)` + `output_text`;
JSON → `text={"format": {"type": "json_object"}}` / `text_format=` via `client.responses.parse` (`output_parsed`);
tool calling → top-level `{"type":"function","name":...}` schema (runtime-normalized from legacy nested form),
`response.output` `function_call` items, `function_call_output` continuation, with `max_depth` recursion guards;
streaming → `response.output_text.delta` events; multimodal → `input_text`/`input_image`/`input_file` content parts;
model controlled by `OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")`.

**Deliberate exceptions (Responses-first does not apply / would teach wrong code):**

- `01-uncontrollability/03-reasoning-thinking.ipynb` — OpenAI surface migrated to Responses (o-series via `reasoning={"effort": ...}`); the **DeepSeek R1** section keeps its OpenAI-**compatible** Chat Completions endpoint because DeepSeek does not implement the Responses API and `reasoning_content` only exists there. Multi-provider notebook, not enrolled.
- `06-multi-agent/01-openai-agents-sdk.ipynb` — built on the **OpenAI Agents SDK** (`Agent`/`Runner`); the planner call was moved to Responses, but Agent model pins are SDK-idiomatic, so it is not enrolled in the OpenAI-only guardrail.
- `08-capstone/01-capstone-overview.ipynb` — skeleton with `NotImplementedError` TODOs (no live OpenAI call), so it is not enrolled; its guidance references the Responses API.
- `05-agent-harness/03-langchain-agents.ipynb` — uses LangChain `create_agent`; its incidental OpenAI helper was moved to Responses but the notebook teaches the LangChain path, not enrolled.

Legacy Chat Completions / `seed` / `n` reproducibility lessons that the Responses API cannot replicate were re-expressed (e.g. multi-sample via loops, reproducibility via low temperature + logging).

### Original completed list (first pass)

- `prompt-engineering/01-uncontrollability/01-openai-api-intro.ipynb`
- `prompt-engineering/01-uncontrollability/02-sampling-and-uncertainty.ipynb`
- `prompt-engineering/02-intent-convergence/01-prompt-basics.ipynb`
- `prompt-engineering/03-structured-output/01-json-mode.ipynb`
- `prompt-engineering/03-structured-output/02-function-calling-basics.ipynb`
- `prompt-engineering/05-agent-harness/09-responses-api.ipynb`

Guardrails:

- `tests/test_openai_2026_static.py`
- `tests/test_openai_responses_notebook.py`

Verification command:

```bash
python -m pytest tests/test_openai_2026_static.py tests/test_openai_responses_notebook.py tests/test_model_baseline.py
```

## Remaining Legacy Hotspots — RESOLVED

All items below were migrated in the completion pass (see the enrolled list above), except the
deliberate exceptions already noted. The original backlog is kept for traceability.

### Highest Priority (done)

These shape the learner's mental model and were migrated:

- `prompt-engineering/01-uncontrollability/03-reasoning-thinking.ipynb`
- `prompt-engineering/01-uncontrollability/04-multimodal.ipynb`
- `prompt-engineering/02-intent-convergence/02-cot-reasoning.ipynb`
- `prompt-engineering/02-intent-convergence/03-prompt-chaining.ipynb`
- `prompt-engineering/02-intent-convergence/04-prompt-integration-usecase.ipynb`

### Structured Output

Move from `client.chat.completions.parse(...)` examples to Responses structured output examples. Keep one explicit migration comparison if useful.

- `prompt-engineering/03-structured-output/03-structured-extraction.ipynb`
- `prompt-engineering/03-structured-output/04-classification-gradio.ipynb`
- `prompt-engineering/07-calibration-eval/01-rag-evaluation.ipynb`

### RAG

For basic RAG, keep local retrieval concepts but send generation through Responses. For hosted retrieval examples, prefer `file_search` + vector stores.

- `prompt-engineering/04-knowledge-rag/01-embedding.ipynb`
- `prompt-engineering/04-knowledge-rag/02-vanilla-rag.ipynb`
- `prompt-engineering/04-knowledge-rag/03-similarity-and-relevance.ipynb`
- `prompt-engineering/04-knowledge-rag/04-vector-db-rag.ipynb`
- `prompt-engineering/04-knowledge-rag/05-dynamic-few-shot.ipynb`
- `prompt-engineering/04-knowledge-rag/06-advanced-rag.ipynb`
- `prompt-engineering/04-knowledge-rag/07-pdf-parsing.ipynb`

### Agent and Tool Calling

New OpenAI-native examples should use Responses tools/function calling. LangChain examples should use LangChain's `create_agent` / `init_chat_model` path, not OpenAI Chat Completions helpers.

- `prompt-engineering/05-agent-harness/01-function-calling-agents.ipynb`
- `prompt-engineering/05-agent-harness/02-react-loop.ipynb`
- `prompt-engineering/05-agent-harness/03-langchain-agents.ipynb`
- `prompt-engineering/05-agent-harness/04-function-calling-rag.ipynb`
- `prompt-engineering/05-agent-harness/05-shop-guardrails.ipynb`
- `prompt-engineering/05-agent-harness/06-prompt-injection.ipynb`
- `prompt-engineering/05-agent-harness/07-plugin-tools.ipynb`
- `prompt-engineering/05-agent-harness/08-chatbot.ipynb`

### Later Passes

- `prompt-engineering/06-multi-agent/01-openai-agents-sdk.ipynb`
- `prompt-engineering/06-multi-agent/03-cross-model-review.ipynb`
- `prompt-engineering/07-calibration-eval/02-feedback-loop.ipynb`
- `prompt-engineering/07-calibration-eval/03-fine-tuning-synthetic-data.ipynb`
- `prompt-engineering/08-capstone/01-capstone-overview.ipynb`

## Migration Rules

1. Default model should be controlled by `OPENAI_MODEL`, not hard-coded to `gpt-4o` or `gpt-4o-mini`.
2. Use a current default only in one place per notebook, for example `os.getenv("OPENAI_MODEL", "gpt-5.4-mini")`.
3. Replace basic generation helpers:

```python
resp = client.responses.create(
    model=OPENAI_MODEL,
    input=messages,
    temperature=temperature,
    max_output_tokens=max_output_tokens,
)
return resp.output_text
```

4. Use `response.output` when teaching tool calls, structured outputs, reasoning traces, or streaming events.
5. Keep Chat Completions examples only inside clearly marked migration/legacy sections.
6. Do not teach Assistants API as a build path; only mention it when explaining migration to Responses.
