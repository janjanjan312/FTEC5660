## Reproducibility Work Report (FTEC5660)

**Student**: (Your Name)  
**Student ID**: (Your ID)  
**Date**: 2026-02-16  

### Project summary

- **Project**: MetaMind — a multi-agent framework for social reasoning / Theory of Mind (ToM).
- **Repo location (this submission)**: `homeworks/Reproducibility Work/MetaMind/`
- **Introduction**: the system runs a multi-stage pipeline (ToM Agent → Domain Agent → Response Agent) with multi-step reasoning and internal selection/refinement.

### Setup notes (env, data, keys, compute)

- **Python**: 3.12.12
- **Install**:

```bash
cd "homeworks/Reproducibility Work/MetaMind"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **LLM provider**: DeepSeek via Volcengine Ark (OpenAI-compatible endpoint).
  - **API key**: API key loaded from environment variables / local `.env`.
  - **Example env**: see `.env.example`.
  - **Controlled change (model swap)**: the upstream project examples reference OpenAI models (e.g., GPT-4). I used DeepSeek (Ark) to make the pipeline runnable with accessible credentials. Therefore, headline numbers in the paper (GPT-4 ToMBench average) are not directly comparable to my measurements.

- **Dataset**: ToMBench (JSONL) shipped as `evaluations.zip` and extracted under `evaluations/`.

### Reproduction target(s) + metric definition

#### Target

Reproduce **ToMBench multiple-choice accuracy** on a clearly-defined subset:

- **Benchmark**: ToMBench
- **Task**: `Ambiguous Story Task`
- **Subset**: first \(N\) examples where \(N \in \{10,20,30\}\)

This is a scoped reproduction (allowed by the assignment), focusing on a single well-defined task subset.

#### Metric

- **Accuracy** \( \% \) = \(\frac{\text{# correct}}{\text{# total}}\times 100\)
- **Correctness rule**: predicted option letter in \(\{A,B,C,D\}\) matches the dataset label.

### Results (my numbers vs reported numbers)

#### My measured results (DeepSeek Ark; temperature=0; hypothesis_count=7)

Command used:

```bash
cd "homeworks/Reproducibility Work/MetaMind"
LLM_TEMPERATURE=0 LLM_MAX_TOKENS=200 TOM_HYPOTHESIS_COUNT=7 python evaluations/tombench/eval_tombench.py --task "Ambiguous Story Task" --max_examples 30 --checkpoints "10,20,30"
```

Checkpoint accuracies (cumulative):

| Subset size | Correct | Accuracy (%) |
|-----------:|--------:|-------------:|
| 10 | 7 | 70.00 |
| 20 | 11 | 55.00 |
| 30 | 15 | 50.00 |

Caveat (evaluation setup): the evaluation script reuses one `MetamindApplication` instance and updates `SocialMemory` for the same `user_id` across questions, so ToMBench items are not strictly independent in this setup. This can contribute to accuracy fluctuations as \(N\) increases.

#### Reference numbers and comparability caveat

The upstream repo/paper reports **overall ToMBench** accuracy numbers under specific model settings (e.g., the project README cites **74.8%** average accuracy for base GPT-4).

In this reproduction, I use a different underlying model/provider (DeepSeek via Ark) and evaluate only a scoped subset (the first 30 items of `Ambiguous Story Task`), so my results are **not directly comparable** to those headline ToMBench averages.

Instead, I treat this as a clearly-defined, measurable subset reproduction target.

### Modification + results after modification

Planned modification (small, isolated, measurable):

- **Change**: `TOM_AGENT_CONFIG["hypothesis_count"]` (**7 → 3**; still overrideable via `TOM_HYPOTHESIS_COUNT`)
- **Why**: this is a key ToM-stage parameter (number of hypotheses generated), and the repo discusses sensitivity to hypothesis count.
- **Measurement**: re-run the same task subset and compare Accuracy before/after.

Results after modification: **TBD** (to be filled once we run the modified setting).

### Debug diary (main blockers + resolutions)

- **Blocker**: dependency install failed under system-managed Python (PEP 668).
  - **Fix**: created a local virtualenv `.venv` and installed `requirements.txt`.
- **Blocker**: ToMBench JSONL files not found at `evaluations/tombench/` after unzip.
  - **Fix**: dataset unzipped into nested path `evaluations/evaluations/tombench/`; evaluation script updated to auto-detect both layouts.
- **Blocker**: `ModuleNotFoundError: No module named 'main'` when running eval script.
  - **Fix**: added project-root `sys.path` injection in `eval_tombench.py` so it can import `main.py` when executed directly.
- **Blocker**: evaluation is slow.
  - **Reason**: multiple LLM calls per example (ToM hypotheses + domain + response + possible revisions).
  - **Mitigation**: use a scoped subset (30 items) and checkpoint reporting.

### Conclusions

TBD after results:

- What reproduced reliably (and under what conditions).
- What was sensitive (e.g., hypothesis count / revision loops).
- Practical recommendations for future users (setup, speed, cost).

