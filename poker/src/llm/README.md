# `poker/src/llm/`

LLM fine-tuning for the legacy [`poker/`](../..) MVP.

## Module

- [`train_llm.py`](train_llm.py) — `PokerLLMTrainer` and
  `PokerLLMDataPreparator`. Fine-tunes a Hugging Face causal LM
  (Mistral-7B / Llama-2-7B / Mixtral / Gemma) on PokerBench's
  prompt/label JSONs. Supports:

  - **LoRA / PEFT** with configurable `r`, `alpha`, `target_modules`.
  - **8-bit loading** via bitsandbytes when available (falls back to
    full precision otherwise — see BUG_AUDIT item 5).
  - **Instruction templates** — Alpaca and ChatML.
  - **Push-to-Hub** and evaluation cadence.

## Usage

Prepare data only:

```bash
cd poker
python src/llm/train_llm.py --data-dir data/processed --prepare-only
```

Full fine-tune (GPU strongly recommended):

```bash
python src/llm/train_llm.py \
    --data-dir data/processed \
    --output-dir data/models/llm_poker \
    --model-name mistralai/Mistral-7B-v0.1 \
    --use-lora --epochs 3 --batch-size 4
```

Config-driven:

```bash
python src/llm/train_llm.py --config configs/llm_config.yaml
```

See [`../../configs/llm_config.yaml`](../../configs/llm_config.yaml)
for the full field reference.

## Known issues (from BUG_AUDIT)

- **Item 4** — fixed. `evaluation_strategy` was renamed to
  `eval_strategy` in modern `transformers`; the trainer now
  introspects `TrainingArguments`' signature and forwards whichever
  kwarg exists.
- **Item 5** — fixed. Unconditional `load_in_8bit=True` used to blow
  up without `bitsandbytes`; now attempted only when the import
  succeeds.
- **Item J** — open. Wide `except Exception: continue` blocks in
  `PokerLLMDataPreparator.prepare_dataset` silently drop rows. Add a
  per-exception-class counter for schema drift visibility.

## Related

- Canonical stack equivalent:
  [`../../../poker_predictor/llm/`](../../../poker_predictor/llm/) —
  a PEP 723 UV script (`train_sft_job.py`) designed for HF Jobs, plus
  a chat-JSONL preparer (`prepare_sft.py`) and a
  transformers/llama.cpp inference wrapper (`infer.py`).
