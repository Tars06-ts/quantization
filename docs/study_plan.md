# Study & build plan

Old code → theory → baseline → time-aware → paper ideas.
Tie every reading block to a file or notebook you'll write, so the theory sticks.

---

## Phase 0: Go through the old code (~half a day)

Don't re-run anything. Read the code and the saved outputs in `notebooks/archive/`.

- [ ] **`dit-try`** (30 min): own AdaLN-Zero + attention; architecture refresher
- [ ] **`dit-pretrained`**: how the pipeline is loaded and called
- [ ] **`dit-quant`**: component sweep (read one of the 4 cells; they're copies)
- [ ] **`mse-based-quant`**: hooks + per-layer MSE. Which layers had the highest MSE? (hint: `attn1.to_out`, `ff.net.2`)
- [ ] For each notebook write 2–3 lines: *what I did / what I found / what was wrong*. Compare with README §1; if you disagree, yours wins.

---

## Phase 1: Theory (1–2 days)

All links: `resources/README.md` (⭐ = must read now).

### Day 1: foundations
| Done | Read | Focus on | Unlocks |
|---|---|---|---|
| [ ] | Lilian Weng, *"What are Diffusion Models?"* (blog) | forward process, x_t, ε-prediction, DDPM vs DDIM, CFG | why x_t changes with t |
| [ ] | **DiT** (Peebles & Xie) | §3: patchify, adaLN-Zero, Fig. 3 | `notebooks/01` |
| [ ] | diffusers source: `DiTTransformer2DModel`, `BasicTransformerBlock`, `AdaLayerNormZero`, `DiTPipeline.__call__` | where t enters, 6 modulation chunks, CFG batch doubling | `pipeline.py`, `timestep.py` |
| [ ] | Nagel et al., *"A White Paper on Neural Network Quantization"* | §2 (uniform / symmetric / per-channel), §3 PTQ (range setting: min-max, MSE) | `quantizers.py` |

### Day 2: diffusion quantization
| Done | Read | Focus on | Unlocks |
|---|---|---|---|
| [ ] | **Q-Diffusion** (+ skim PTQ4DM) | origin of time-step-aware calibration (UNet era) | the core idea |
| [ ] | **TDQ** | MLP predicts quant params from timestep, close to candidate direction #2 | novelty check |
| [ ] | **PTQ4DiT** | salient channels, temporal activation variation, calibration | baseline to beat |
| [ ] | **Q-DiT** | granularity allocation, dynamic activation quant | "dynamic" upper bound |
| [ ] | **TQ-DiT**, **TaQ-DiT** | how they group / handle timesteps (TQ-DiT's TGQ = your bucketing baseline!) | closest competitors |
| [ ] | (optional) SmoothQuant | channel-outlier migration | `notebooks/02` outlier question |

For the four DiT papers: abstract, intro, method figure, experimental-setup table. Skip the proofs.
Fill in the table in `docs/problem_statement.md` as you go: **bits, sampler, steps, CFG, FID protocol, numbers**.

### Self-check: answer these without notes before coding
- [ ] Why does the qkv input in block 10 look different at t=900 vs t=50?
- [ ] Per-tensor vs per-channel: which axis for a Linear weight, and why?
- [ ] What does min-max calibration get wrong when there are outliers?
- [ ] What is dynamic quantization, and why is it an upper bound rather than the goal?
- [ ] How does each of the four DiT papers handle time, in one line each?

---

## Phase 2: Baseline (build → validate, alternating)

- [ ] `config.py`, `utils.py`, `pipeline.py` → `scripts/00_sanity_fp16.py` → **notebook 01**
- [ ] `layers.py`, `timestep.py::TimestepTracker` → notebook 01 Q6 (timestep list)
- [ ] `hooks.py::ActivationStats` → `scripts/01_collect_stats.py` → **notebook 02**
      ⚠️ **Go/no-go:** do activations actually move with t? Do they vary more across t than across samples?
- [ ] `quant/quantizers.py` + `tests/test_quantizers.py` → **notebook 03**
- [ ] `quant/qlinear.py` (1 bucket), `quant/calibrate.py`, `eval/metrics.py` → `scripts/02_static_ptq.py` → **notebook 04**
- [ ] `scripts/03_component_sweep.py` → notebook 04 (component questions)

**Baseline done when:** W8A8 static PTQ runs end-to-end, plus error-vs-timestep plot, plus static-vs-dynamic headroom measured at W8A8 / W8A6 / W4A8.

---

## Phase 3: Time-aware quantization

- [ ] **Notebook 05, Part A** (toy). Start as soon as `quantizers.py` works; this is where you *learn* it.
- [ ] `timestep.py::uniform_buckets` → bucket lookup in `QuantLinear` → `calibrate(bucket_of_t=...)`
- [ ] `scripts/04_timestep_buckets.py` → **notebook 05, Part B**
- [ ] `timestep.py::clustered_buckets` → uniform vs clustered comparison

**Time-aware baseline done when:** you have the quality-vs-number-of-buckets curve and know how much of the static→dynamic headroom bucketing recovers.

---

## Phase 4: From baseline to paper idea

Only start this once Phases 2–3 give real numbers. The ideas should come from what the plots show, not from guesses.

- [ ] List every surprise / failure from notebooks 02, 04 and 05 (that's where ideas come from)
- [ ] For each candidate direction (README §9), write: *claim, evidence I have, experiment that would prove it, closest prior work*
- [ ] Brainstorm with Claude / supervisor → pick **one** focused contribution
- [ ] Notebooks 06 / 07 depending on the direction chosen
- [ ] Repeat key results on DiT-XL/2-512, then FID on the survivors (`scripts/07`)

---

## Habits

- **Lab log** (`docs/log.md`): one dated entry per day covering what I did, what I saw, what's next. Becomes the experiment section.
- **`git init` now**, commit at each milestone.
- **Limit plumbing to ~1 h.** If diffusers internals or hooks are stuck after that, ask for help. Spend the time on research questions.
