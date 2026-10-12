# Resources

Suggested layout (PDFs and cloned repos are gitignored; this README is tracked):

```
resources/
├── 1_foundations/       diffusion + DiT + quantization basics
├── 2_diffusion_quant/   UNet-era diffusion PTQ (where time-aware ideas started)
├── 3_dit_quant/         the DiT papers you compete with
├── 4_formats/           MX / NVFP4
├── repos/               official code (READ ONLY, for checking details, not copying)
└── data/                FID reference batches (only needed in Phase 4)
```

Priority: ⭐ = must read (Phase 1) · ○ = read when you reach that phase · · = optional

---

## 1_foundations
| | Resource | Link | Why |
|---|---|---|---|
| ⭐ | Lilian Weng, *What are Diffusion Models?* (blog, save as PDF) | https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ | x_t, ε-prediction, DDPM/DDIM, CFG in one place |
| ⭐ | **DiT**: Scalable Diffusion Models with Transformers | https://arxiv.org/abs/2212.09748 | §3: adaLN-Zero, patchify |
| ⭐ | Nagel et al., *A White Paper on Neural Network Quantization* | https://arxiv.org/abs/2106.08295 | §2–3: the math behind `quantizers.py` |
| ○ | DDPM (Ho et al.) | https://arxiv.org/abs/2006.11239 | reference if the blog isn't enough |
| ○ | DDIM (Song et al.) | https://arxiv.org/abs/2010.02502 | which sampler / why timesteps get skipped |
| ○ | Classifier-Free Guidance | https://arxiv.org/abs/2207.12598 | why the batch doubles |
| · | Gholami et al., *A Survey of Quantization Methods* | https://arxiv.org/abs/2103.13630 | alternative to the white paper |
| · | AdaRound | https://arxiv.org/abs/2004.10568 | rounding optimization, cited by everyone |
| · | BRECQ | https://arxiv.org/abs/2102.05426 | block reconstruction, the basis of Q-Diffusion |

## 2_diffusion_quant (UNet era: where time-aware PTQ started)
| | Resource | Link | Why |
|---|---|---|---|
| ⭐ | **Q-Diffusion** | https://arxiv.org/abs/2302.04304 | time-step-aware calibration data |
| ⭐ | **TDQ**: Temporal Dynamic Quantization (NeurIPS'23) | https://arxiv.org/abs/2306.02316 | ⚠️ **MLP predicts quant params from the timestep.** Very close to candidate direction #2. Must read |
| ○ | PTQ4DM | https://arxiv.org/abs/2211.15736 | first diffusion PTQ |
| ○ | TFMQ-DM (CVPR'24) | https://arxiv.org/abs/2311.16503 | temporal-feature-aware reconstruction + finite-set calibration |
| · | SmoothQuant | https://arxiv.org/abs/2211.10438 | channel-outlier migration (PTQ4DiT's CSB is related) |

## 3_dit_quant (direct competitors)
| | Resource | Link | Code |
|---|---|---|---|
| ⭐ | **PTQ4DiT** (NeurIPS'24) | https://arxiv.org/abs/2405.16005 | https://github.com/adreamwu/PTQ4DiT |
| ⭐ | **Q-DiT** (CVPR'25) | https://arxiv.org/abs/2406.17343 | https://github.com/Juanerx/Q-DiT |
| ⭐ | **TaQ-DiT** (Nanjing; TCSVT) | https://arxiv.org/abs/2411.14172 | https://github.com/6xy-liu/Taq-DiT |
| ⭐ | **TQ-DiT** (KAIST) ⚠️ different paper from TaQ-DiT | https://arxiv.org/abs/2502.04056 | — |
| ○ | ViDiT-Q (ICLR'25): image + video DiT, mixed precision | https://arxiv.org/abs/2406.02540 | https://github.com/A-suozhang/ViDiT-Q |

⚠️ **TQ-DiT already does time-grouping quantization (TGQ) on DiT.** Your bucketing baseline is *their* method, so your contribution must go beyond "use timestep groups".
⚠️ **Q-DiT's numbers changed between arXiv v1 and v2** (W8A8 → W6A8 headline). Cite the version you compare against.

## 4_formats (Phase 3+)
| | Resource | Link |
|---|---|---|
| ○ | Microscaling Data Formats for Deep Learning | https://arxiv.org/abs/2310.10537 |
| ○ | OCP Microscaling Formats (MX) v1.0 spec | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf |
| ○ | NVIDIA blog, *Introducing NVFP4…* (save as PDF) | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ |

## repos/ (clone, don't copy from)
```bash
cd resources/repos
git clone https://github.com/adreamwu/PTQ4DiT
git clone https://github.com/Juanerx/Q-DiT
git clone https://github.com/6xy-liu/Taq-DiT
git clone https://github.com/openai/guided-diffusion   # evaluations/ = the FID evaluator the papers use
git clone https://github.com/huggingface/diffusers     # read models/transformers/dit_transformer_2d.py, attention.py, normalization.py, pipelines/dit/
```
Use them to answer *"what exactly did they do?"* (calibration size, steps, which layers stay FP). Write your own code.

## data/ (Phase 4 only, several GB)
- ImageNet 256 reference batch for FID: `VIRTUAL_imagenet256_labeled.npz`. Get the link from `guided-diffusion/evaluations/README.md`.
- 512 equivalent for DiT-XL/2-512.
- **Model weights:** nothing to download manually. `diffusers` caches `facebook/DiT-XL-2-256` / `-512` in `~/.cache/huggingface` on first load (set `HF_TOKEN`).
