# dtq — Time-Aware Quantization of Diffusion Transformers

Post-training quantization (PTQ) research on **DiT-XL/2** (256 and 512), centered on one question:

> **In a DiT, the same layer sees different activation distributions at different denoising
> timesteps. Can timestep-aware quantization improve quality at low precision without
> excessive overhead?**

This repo is a rewrite of my old Kaggle notebooks (`notebooks/archive/`) into a small and
reproducible Python package plus numbered experiment scripts, meant to run on an **A3 (H100) GPU**.

> **Status:** skeleton only. Every file has a docstring that says *what goes there*,
> *which notebook it replaces*, and *what to watch out for*. Each function is a stub
> (`raise NotImplementedError`). I'm implementing them myself to re-learn the material.
>
> **Start here: [`docs/study_plan.md`](docs/study_plan.md).**

---

## 1. Where I left off (old notebooks → lessons)

| Notebook | What it did | What was wrong / missing | Goes to |
|---|---|---|---|
| `dit-try` | Toy AdaLN-Zero and MHSA from scratch | Learning exercise only. Don't reimplement DiT | `docs/model_map.md` (understanding) |
| `dit-pretrained` | FP16 baseline (DPM++, 25 steps, CFG 4) + torchao sweep (int8 W-only, W8A8-dynamic, fp8 W-only, W4A8) on 256 & 512 | torchao = black box; no control over scales, can't do per-timestep. Int8 was *slower* than FP16 on T4 | `pipeline.py`, `scripts/00`, `scripts/02` |
| `dit-quant` | Component-wise sweep: adaLN / attn / mlp combos, int8 & fp8 | Same code copy-pasted 4×; quality judged only as "BLACK/OK" | `layers.py`, `policy.py`, `scripts/03` |
| `mse-based-quant` | Per-layer output MSE (FP vs int8), then thresholds → int4 / int8 / skip | MSE at **one dummy timestep (t=50) with random latents**; hooks overwrote outputs every call; raw-MSE thresholds; int4 chosen from int8 error; the result produced **NaNs**; mixed scheduler/steps vs other notebooks; `"block" in name` also caught per-block timestep embedders | `hooks.py`, `timestep.py`, `policy.py`, `scripts/05` |

**Biggest takeaway:** every old experiment ignored the time axis. That gap is the research direction.

---

## 2. Repo structure: what goes where

```
quantization/
├── README.md
├── requirements.txt          deps (install torch separately for your CUDA)
├── pyproject.toml            `pip install -e .` -> `import dtq` works from scripts
├── docs/
│   ├── study_plan.md         START HERE: phased checklist (old code -> theory -> baseline -> time-aware -> paper)
│   ├── problem_statement.md  one-page problem statement + related-work table   (Day 1 AM)
│   └── model_map.md          DiT-XL/2 module names/shapes from print(pipe.transformer)
├── src/dtq/
│   ├── config.py             Config dataclass: model, scheduler, steps, CFG, seeds, bits, buckets
│   ├── utils.py              seeding, CUDA timer, peak memory, json io
│   ├── pipeline.py           load DiTPipeline, swap scheduler, generate(), copy transformer
│   ├── layers.py             module name -> component (qkv/attn_out/fc1/fc2/adaln/embedder), block idx
│   ├── timestep.py           TimestepTracker (which t are we at?), uniform & clustered buckets
│   ├── hooks.py              ActivationStats (per layer × t), OutputErrorProbe (FP vs quant per layer × t)
│   ├── quant/
│   │   ├── quantizers.py     fake int quant: scale/zp, per-tensor/channel/group, minmax/percentile/MSE ranges
│   │   ├── qlinear.py        QuantLinear: W fake-quant + A fake-quant with scale looked up by timestep bucket
│   │   ├── calibrate.py      run calibration trajectories -> scales per (layer, bucket)
│   │   ├── policy.py         which layers get which bits: uniform / component / mixed / budgeted
│   │   └── mx.py             simulated MXFP8 (32-blk, E8M0 scale) & NVFP4 (16-blk, E4M3 scale + FP32 global)
│   └── eval/
│       ├── metrics.py        degenerate check, PSNR, LPIPS (vs FP16), CLIP score, FID
│       └── plots.py          image grids, stat-vs-t, error-vs-t, layer×t heatmap
├── scripts/                  each one = one experiment, writes to results/<name>/
│   ├── 00_sanity_fp16.py       FP16 sampling works; fp16 vs bf16 noise floor
│   ├── 01_collect_stats.py     activation stats across timesteps  (motivating figure)
│   ├── 02_static_ptq.py        W8A16 / W8A8 static / dynamic; error-vs-t plot
│   ├── 03_component_sweep.py   which component is sensitive, and when  (port of dit-quant)
│   ├── 04_timestep_buckets.py  static vs 2/3/4/8 buckets vs per-t vs dynamic; uniform vs clustered
│   ├── 05_mixed_precision.py   time-aware mixed precision under bit budget  (port of mse-based-quant)
│   ├── 06_mx_formats.py        MXFP8 / NVFP4 simulation; does block scaling make time-awareness moot?
│   └── 07_eval_fid.py          FID for the final table (expensive, resumable)
├── tests/
│   └── test_quantizers.py    correctness tests for quant primitives. Write these first
├── notebooks/                QUESTIONNAIRES: question -> empty code cell -> "my answer" cell
│   ├── 01_dit_anatomy.ipynb                 what am I quantizing? (fills docs/model_map.md)
│   ├── 02_temporal_activation_stats.ipynb   do activations really change with t? (motivation)
│   ├── 03_quantizer_sanity.ipynb            are my quantizers correct / well-chosen?
│   ├── 04_static_ptq.ipynb                  where does static PTQ fail; how much headroom?
│   ├── 05_time_aware.ipynb                  learn time-aware PTQ (toy), then justify it on DiT
│   ├── 06_sensitivity_mixed_precision.ipynb temporal sensitivity -> bit allocation
│   ├── 07_mx_formats.ipynb                  does block scaling make time-awareness moot?
│   └── archive/              old Kaggle notebooks (read-only reference)
└── results/                  gitignored outputs; every run saves its config.json here
```

### Dependency graph (build bottom-up)

```
config, utils
   └── pipeline ── layers ── timestep
                     │          │
               quant/quantizers │ ── quant/mx
                     │          │
               quant/qlinear ◄──┘
                     │
         quant/calibrate, quant/policy, hooks
                     │
             eval/metrics, eval/plots
                     │
                  scripts/0x
```

### Code vs notebooks

- **`src/dtq/` + `scripts/`** hold all the logic. Scripts run the experiments (also headless on the GPU VM) and save CSVs and figures to `results/`.
- **`notebooks/`** are for **validation and justification only**. Each one is a list of questions, each followed by an empty code cell and a "my answer" cell. Notebooks import from `dtq` and load from `results/`. They **never define functions**: if you write a helper twice, move it into `src/dtq/`.
- The verdict cells are the raw material for the paper's motivation, ablations and discussion.

### Where the time-aware code lives

Time-awareness isn't a separate module. It's a time axis threaded through the existing pieces:

| Piece of time-aware PTQ | File |
|---|---|
| Know the current timestep inside any layer | `timestep.py::TimestepTracker` |
| Map timestep → bucket (uniform, clustered) | `timestep.py::uniform_buckets / clustered_buckets` |
| Collect statistics per (layer, t) | `hooks.py::ActivationStats` |
| Measure quant error per (layer, t) | `hooks.py::OutputErrorProbe` |
| Store one activation scale per bucket; look it up at forward time | `quant/qlinear.py::QuantLinear` |
| Calibrate scales per bucket | `quant/calibrate.py::calibrate(bucket_of_t=...)` |
| Bits per (layer, bucket) | `quant/policy.py::apply_mixed_precision / sensitivity_to_bits` |
| Run the comparisons | `scripts/04_timestep_buckets.py`, `scripts/05_mixed_precision.py` |
| Learn + justify | `notebooks/05_time_aware.ipynb` (Part A is a toy example to learn on before touching DiT) |

Static PTQ is just the special case `num_buckets = 1`, so the baseline and the method share one code path. That makes the comparison fair.

---

## 3. Build order and milestones

Tick these off as you go.

### Milestone A: baseline harness (Day 1)
- [ ] `config.py`, `utils.py`
- [ ] `pipeline.py` → `scripts/00_sanity_fp16.py` produces a correct image grid
- [ ] `docs/model_map.md` filled in
- [ ] `layers.py` (check: 196 quantizable linears in 28 blocks)
- [ ] `timestep.py::TimestepTracker` (print t at every step, check it matches `pipe.scheduler.timesteps`)
- [ ] `hooks.py::ActivationStats` → `scripts/01_collect_stats.py` → **stat-vs-timestep plot**
- [ ] `docs/problem_statement.md` drafted using that plot

### Milestone B: static PTQ (Day 1 PM)
- [ ] `quant/quantizers.py` + `tests/test_quantizers.py` passing
- [ ] `quant/qlinear.py` (static + dynamic modes)
- [ ] `quant/calibrate.py` (num_buckets = 1)
- [ ] `eval/metrics.py` (degenerate, PSNR, LPIPS)
- [ ] `scripts/02_static_ptq.py` → W8A8 end-to-end, **error-vs-timestep plot**

### Milestone C: time-aware (Day 2)
- [ ] `timestep.py::uniform_buckets`, then `clustered_buckets`
- [ ] `scripts/03_component_sweep.py`: component × timestep sensitivity
- [ ] `scripts/04_timestep_buckets.py`: **all-timestep vs early/mid/late vs K buckets**

### Milestone D: contribution (after)
- [ ] `quant/policy.py::sensitivity_to_bits` → `scripts/05_mixed_precision.py`
- [ ] `quant/mx.py` → `scripts/06_mx_formats.py`
- [ ] `scripts/07_eval_fid.py` on the survivors; repeat key results on DiT-XL/2-512

---

## 4. Experiment matrix

| Method | W | A | Purpose | Script |
|---|---|---|---|---|
| FP16 | 16 | 16 | Reference | 00 |
| Weight-only PTQ | 8 | 16 | Sanity (should be ~lossless) | 02 |
| Static PTQ (per-tensor) | 8 | 8 | Baseline | 02 |
| Static PTQ (per-channel W) | 8 | 8 | Better baseline | 02 |
| Time-bucket PTQ (2/3/4/8, uniform & clustered) | 8 | 8 (+ 6) | Tests temporal adaptation | 04 |
| Per-timestep PTQ | 8 | 8 | Static-calibration upper bound | 04 |
| Dynamic PTQ | 8 | 8 | Accuracy upper bound | 02/04 |
| Time-aware mixed precision | 8/6/4 | 8 | First contribution candidate | 05 |
| MXFP8 / NVFP4 (sim) | MX | MX | Secondary axis | 06 |

Metrics: **LPIPS / PSNR vs FP16** (same seeds) for every ablation. **FID / sFID / IS** only for final rows.

---

## 5. Setup & running

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
pip install -e .
export HF_TOKEN=...            # avoids HF rate limits (the notebooks warned about this)

pytest -q                      # quant primitives first
python scripts/00_sanity_fp16.py --out_dir results/00_sanity
python scripts/01_collect_stats.py --out_dir results/01_stats
# ... and so on in order
```

Long runs on a remote VM: use `tmux` or `nohup python ... > log.txt 2>&1 &`, so a dropped SSH doesn't kill a FID job.

---

## 6. Rules for myself

1. **Fix scheduler + steps + CFG once** (in `Config`) and never compare across different settings.
2. **Calibration seed ≠ evaluation seed.**
3. **Fake quant for quality, real kernels for speed.** Don't mix the claims.
4. **Every script saves `config.json` and a CSV** next to its figures. Plots are regenerated from CSVs.
5. **Run `is_degenerate` on every output.** NaN images silently ruined the old MSE experiment.
6. **Use relative error** (MSE / var), not raw MSE, when comparing layers.
7. Check the stat-vs-t plot *early*. If ranges barely move with t, rethink the story before building more.

---

## 7. Hardware notes (A3 = H100, Hopper)

- 80 GB HBM: two copies of DiT-XL/2 (~1.35 GB each in fp16) are trivial; batch generation of 128+ images is fine.
- Native **FP8 (E4M3/E5M2)** tensor cores: real FP8 speed benchmarks are possible here (e.g. torchao / Transformer Engine).
- **No native MXFP8 / NVFP4**: those are Blackwell features. On H100 they are **simulation only** (quality, not speed).
- Speed numbers from fake quant are meaningless. Report latency only for real-kernel runs.

---

## 8. Reading list (re-read the key sections only)

- **PTQ4DiT**: salient channels + temporal variability; Channel-wise Salience Balancing, Spearman-ρ-guided calibration.
- **Q-DiT**: automatic granularity allocation + sample-wise dynamic activation quantization.
- **TQ-DiT**: time-grouped / timestep-aware quantization for DiT.
- **TaQ-DiT**: time-aware quantization; joint reconstruction, handling of post-GELU / shift-sensitive activations.
- **OCP Microscaling (MX) spec v1.0**, **NVIDIA NVFP4** docs: for `mx.py`.

For each paper, note in `docs/problem_statement.md` the bits, sampler, steps, CFG and FID protocol, so my numbers are comparable.

---

## 9. Candidate paper directions (pick one)

1. **Timestep-clustered mixed precision**: more bits where (layer, timestep-group) sensitivity is high.
2. **Calibration-efficient time-aware PTQ**: few clusters, or predict scales from the timestep embedding.
3. **Temporal sensitivity metric**: cheaply predicts which layers need precision across denoising; validate on 256 + 512.
4. **Time-aware microscaling**: does MXFP8/NVFP4 block scaling absorb temporal outliers?
