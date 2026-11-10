# Qwen3-8B Attention Tracker: Deepset + BIPIA

This project reimplements the detection math in **Attention Tracker: Detecting Prompt Injection Attacks in LLMs** (NAACL Findings 2025) for `Qwen/Qwen3-8B`. It is a research prototype, not a production security boundary. Code is original and uses the official BIPIA builder package.

## Experiment flow

1. Load Qwen3-8B in eager-attention, non-thinking mode.
2. Collect per-head last-input-token attention to instruction and data on 30 **synthetic** normal/ignore-attack pairs; the benchmark test set is not involved.
3. Select important heads with `mean_normal - mean_attack - k*(std_normal + std_attack) > 0`, default `k=4` (population std).
4. Calibrate a fixed rejection threshold on **benign Deepset train** examples (5th percentile of focus score by default).
5. Evaluate the **Deepset test split** and BIPIA's paired clean/poisoned cases using the exact same detector and threshold. Lower focus score means more suspicious.
6. Optionally generate baseline answers and guarded answers for BIPIA. The guarded answer is blocked if the detector rejects; otherwise it is the same unguarded answer.

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

# BIPIA defense-vs-baseline response examples (no automatic ASR judge)
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

`--threshold` can override calibration. If you change `--model-id`, use a different `--head-checkpoint`. If you change `--k`, re-run with `--force-refit`. For new experimental configurations it is recommended to write to a new `--output-dir` so result files do not overwrite each other.

## Outputs

- `tracker_qwen3_8b.json`: selected `(layer, head)` pairs, k and threshold.
- `head_candidate_scores.npy`: per-head conservative separation scores.
- `deepset_metrics.json`, `deepset_predictions.jsonl`.
- `bipia_{task}_metrics.json`, `bipia_{task}_predictions.jsonl`.
- `bipia_{task}_skipped.jsonl` if examples exceed the input-token budget.
- `summary.json`.

`*_predictions.jsonl` contains each sample's label, prediction correctness, Focus Score, risk score, and BIPIA task/attack metadata. Optional baseline and defended responses are included when `--generate-bipia-responses` is set.

## Evaluation cautions

- **Deepset** is a labeled detection dataset, not a benchmark of how often a prompt injection truly alters the answer. AUROC and false positive/negative rates are valid; model ASR cannot be inferred from the labels alone.
- **BIPIA** in the original repository embeds untrusted `context` into its *system prompt*. This program preserves its default task prompts and treats the initial system text prior to context as instruction, with the document context and user question as data. This is an extension of the original tracker and should be explicitly documented when comparing across papers.
- **BIPIA attack presence is not attack success**. The `attack_detection_rate_tpr` and `paired_clean_pass_attack_block_rate` are *detector-level* metrics. For genuine baseline vs defended **Attack Success Rate (ASR)**, run BIPIA's official attack-success judge or manually annotate the generated baseline output and guarded output. This repository does **not** label unjudged outputs as successful attacks.
- Head-selection sentences were authored independently and are **not exactly** the 30 GPT-4-generated samples used in the original paper. The model is also newer. Thus these are new experiments, **not a numerical reproduction** of published results.
- The global rejection threshold is calibrated on the benign Deepset train set and *transferred* to BIPIA. AUROC is threshold-free; BIPIA FPR/TPR can suffer from domain shift. A BIPIA-specific threshold calibration should use a separate **BIPIA train** set, not the test set.
- The default BIPIA sampling is a small, reproducible **subset**: first random 3 source contexts, 2 variants per attack family, then 40 injected pairs per task. This is not the full benchmark. Increase limits for full coverage, and report skipped input examples.
- A classifier's ROC AUC does not establish that prompt injection is prevented during real-world tool use.

## Sources

- Paper: https://aclanthology.org/2025.findings-naacl.123/
- Original detector and head-selection code: https://github.com/khhung-906/Attention-Tracker
- Deepset: https://huggingface.co/datasets/deepset/prompt-injections
- BIPIA: https://github.com/microsoft/BIPIA
- Qwen3: https://huggingface.co/Qwen/Qwen3-8B
