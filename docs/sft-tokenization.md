# SFT Tokenization and Assistant-Only Loss

Kiln native SFT has one Qwen3.5-4B tokenization and label contract. Inline
examples, server-local JSONL, and named datasets all reach this same path after
message-schema validation.

Row admission is fail-closed by default and uses the same tokenization check
through every transport. The optional explicit skip policy and stable
kept/rejected row hashes are specified in
[SFT Ingestion, Invalid Rows, and Row Identity](sft-ingestion.md).

## Conversation rendering

SFT validates the source-pinned model `chat_template.jinja`, selects TRL 1.8's
prefix-preserving Qwen3.5 thinking training variant, and renders with
`add_generation_prompt=false`. A training sequence ends after the final
serialized message. Kiln does not append the open assistant/thinking prefix
used to start inference.

OpenAI-shaped assistant tool calls are accepted with
`function.arguments` as a JSON string. Kiln parses that string into the object
the Qwen template expects before rendering. A tool-call-only assistant message
may use `content: null` or omit `content`; both normalize to empty content. Tool
responses retain `role: "tool"`, `name`, and `tool_call_id` through ingestion,
although the current Qwen template does not serialize the latter two fields.

## Assistant-only labels

The mask contract is `kiln.qwen35-assistant-only.v1`:

- System, user, and tool-response turns are masked from loss.
- The assistant role header `<|im_start|>assistant\n` is masked from loss.
- Assistant thinking text, answer text, and Qwen tool-call XML are supervised.
- The closing `<|im_end|>` and its trailing newline are supervised.
- There is no appended generation prompt in an SFT sequence.

The checked-in fixture uses the standard Hugging Face label representation:

```text
labels[i] = input_ids[i]  when assistant_mask[i] == 1
labels[i] = -100          otherwise
```

The causal loss predicts token `i` from the preceding position. Excluding the
assistant role header is deliberate and matches TRL; including the message
terminator ensures the model is trained to stop. Kiln carries TRL's generation
boundaries through rendering as internal markers and removes them before
tokenization; it does not infer loss spans by searching message text for Qwen
delimiter strings. An internal-marker collision fails the row closed.

## HF and TRL equivalence

The official Qwen3.5-4B template at the pinned revision does not contain
Hugging Face `{% generation %}` blocks. Asking Transformers for an assistant
mask from that template directly produces an all-zero mask, so it is not a
valid `assistant_only_loss` template by itself.

TRL 1.8 detects this template and selects its bundled
`qwen3_5_think_training.jinja`. That template is prefix-preserving, always
serializes the reasoning block on completed assistant turns, starts the
`{% generation %}` span after the assistant role header, and ends it after the
message terminator. The oracle runs that exact source-pinned template through
Transformers and uses `return_assistant_tokens_mask` as the label mask. Kiln
uses the same template with the two Jinja extension tags adapted to internal
span markers that are removed after rendering. The effective-template digest
is over the marker-free template. The checked-in real-tokenizer test proves
rendered text, token IDs, masks, and labels still match exactly.

The source-pinned fixture covers plain, explicit thinking, tool-call,
tool-response, multi-turn, and adversarial delimiter-literal examples. The
latter proves message text containing Qwen role and terminator strings cannot
move assistant loss boundaries. It records exact rendered text, token IDs,
assistant masks, and `-100` labels together with the model revision, artifact
hashes, Transformers and TRL source hashes, and dependency versions:

- `crates/kiln-train/tests/fixtures/qwen35_sft_oracle_v1.json`
- `scripts/qualification/qwen35_sft_oracle.py`

Regenerate and compare it against a staged copy of the pinned model artifacts:

```bash
uv run \
  --with transformers==5.13.1 \
  --with tokenizers==0.22.2 \
  --with jinja2==3.1.6 \
  python scripts/qualification/qwen35_sft_oracle.py \
  --model-path /path/to/Qwen3.5-4B \
  --check

KILN_QWEN35_MODEL_PATH=/path/to/Qwen3.5-4B \
  cargo test --locked -p kiln-train \
  --test qwen35_sft_oracle -- --ignored
```

The Python contract tests validate the checked-in fixture without installing
Transformers. The ignored Rust integration test is the real-tokenizer gate and
must be run on a machine with the exact staged artifacts. Neither test requires
a GPU.
