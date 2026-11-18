# Qwen3-8B Attention Tracker: Deepset + BIPIA

This project reimplements the detection math in **Attention Tracker: Detecting Prompt Injection Attacks in LLMs** (NAACL Findings 2025) for `Qwen/Qwen3-8B`. It is a research prototype, not a production security boundary. Code is original and uses the official BIPIA builder package.

## Experiment flow

1. Load Qwen3-8B in eager-attention, non-thinking mode.
2. Collect per-head last-input-token attention to instruction and data on 30 **synthetic** normal/ignore-attack pairs; the benchmark test set is not involved.
3. Select important heads with `mean_normal - mean_attack - k*(std_normal + std_attack) > 0`, default `k=4` (population std).
4. Score the **Deepset test split** and BIPIA's paired clean/poisoned cases with the same selected heads. Lower Focus Score means more suspicious.
5. Measure AUROC from negative Focus Score; no rejection threshold or binary decision is used.
6. Optionally generate original model answers for BIPIA.

**Focus Score:** for each important head, sum last-token attention on the instruction range, normalize by attention on `(instruction + meaningful data ranges)`, then average over selected heads. This matches the official repository's `process_attn(..., 'normalize_sum')` and head aggregation, while extending the range handling to Qwen3 chat templates and BIPIA task prompts.

## Install

Requires Linux and CUDA GPU. The 8B model in BF16 is roughly 16 GB of weights alone, and eager-attention temporarily constructs quadratic attention matrices. **24+ GB VRAM is recommended** for the default short contexts. Even with forward hooks the entire matrix for an individual layer is materialized; reduce `--max-input-tokens` if out of memory. `max-input-tokens` is a **limit, not silent truncation**.

```bash
# Install torch with a CUDA build appropriate for your GPU first.
pip install -r requirements.txt

mkdir -p external
git clone https://github.com/microsoft/BIPIA.git external/BIPIA
pip install -e external/BIPIA

# Full default: Deepset test + three locally available BIPIA tasks.
python run_experiment.py --benchmark both \
    --bipia-repo external/BIPIA \
    --bipia-tasks email,table,code
```

BIPIA's `qa` (WebQA) and `abstract` (Summarization) task context files may require extra licensed data downloads, as specified in `external/BIPIA/benchmark/README.md`. Set `--bipia-tasks email,table,code,qa,abstract` only after obtaining them.

### Examples

```bash
# Quick smoke run (still loads all 8B model parameters)
python run_experiment.py --benchmark deepset --deepset-test-limit 20

# BIPIA original model response examples (no automatic ASR judge)
python run_experiment.py --benchmark bipia --bipia-tasks email,table,code \
    --generate-bipia-responses --max-new-tokens 128

# Retry head selection if k=4 produces zero important heads
python run_experiment.py --k 3 --force-refit

# More substantial BIPIA evaluation (all available context/attack variants/pairs)
# Requires a suitable GPU and max-input-tokens adapted to real contexts.
python run_experiment.py --benchmark bipia --bipia-max-contexts 0 \
    --bipia-max-variants-per-attack 0 --bipia-max-pairs 0 \
    --max-input-tokens 2048

# Inspect core tests without any GPU or model downloads
pytest -q
```

If you change `--model-id`, use a different `--head-checkpoint`. If you change `--k`, re-run with `--force-refit`. Use a new `--output-dir` for this score-only result format so it stays distinct from older threshold-based results.

## Outputs

- `tracker_qwen3_8b.json`: selected `(layer, head)` pairs and k. Older checkpoints with a threshold can still be loaded; the threshold is ignored.
- `head_candidate_scores.npy`: per-head conservative separation scores.
- `deepset_metrics.json`, `deepset_predictions.jsonl`.
- `bipia_{task}_metrics.json`, `bipia_{task}_predictions.jsonl`.
- `bipia_{task}_skipped.jsonl` if examples exceed the input-token budget.
- `summary.json`.

`*_predictions.jsonl` contains each sample's label, Focus Score, `formatted_text` (the complete chat-template prompt before tokenization, including the assistant generation prefix), and BIPIA task/attack metadata. Optional original model responses are included when `--generate-bipia-responses` is set. `*_metrics.json` contains AUROC and sample counts, including complete BIPIA pair counts.

## Evaluation cautions

- **Deepset** is a labeled detection dataset, not a benchmark of how often a prompt injection truly alters the answer. AUROC evaluates ranking by Focus Score; model ASR cannot be inferred from the labels alone.
- **BIPIA** in the original repository embeds untrusted `context` into its *system prompt*. This program extracts the official instruction, puts it and the user question in the system message, and puts the clean or poisoned context in the user message. Focus Score treats the whole system message as instruction and the context as data. This changes the official prompt layout, so results are not directly comparable with runs using the official BIPIA layout.
- **BIPIA attack presence is not attack success**. AUROC measures score ranking against attack-presence labels. To measure **Attack Success Rate (ASR)**, run BIPIA's official attack-success judge or manually annotate generated responses. This repository does **not** label unjudged outputs as successful attacks.
- Head-selection sentences were authored independently and are **not exactly** the 30 GPT-4-generated samples used in the original paper. The model is also newer. Thus these are new experiments, **not a numerical reproduction** of published results.
- AUROC is threshold-free. BIPIA scores can still differ from Deepset because the tasks and prompt layouts differ.
- The default BIPIA sampling is a small, reproducible **subset**: first random 3 source contexts, 2 variants per attack family, then 40 injected pairs per task. This is not the full benchmark. Increase limits for full coverage, and report skipped input examples.
- A classifier's ROC AUC does not establish that prompt injection is prevented during real-world tool use.

## Sources

- Paper: https://aclanthology.org/2025.findings-naacl.123/
- Original detector and head-selection code: https://github.com/khhung-906/Attention-Tracker
- Deepset: https://huggingface.co/datasets/deepset/prompt-injections
- BIPIA: https://github.com/microsoft/BIPIA
- Qwen3: https://huggingface.co/Qwen/Qwen3-8B
