"""
Milestone 1: FP16 DiT-XL/2-256 sampling works locally.

Ports: dit-pretrained.ipynb, cell 1.

TODO:
  - parse args (--model_id, --steps, --out_dir) -> load_config(...)
  - seed, load_pipeline, generate 4 classes, save grid + individual PNGs
  - print latency + peak memory
  - also do fp16 vs bf16: are the images (near-)identical? (LPIPS between them = your noise floor)
  - run twice with same seed: bit-identical? If not, note it (nondeterministic kernels).

Done when: results/00_sanity/ has a grid that looks like a golden retriever, tabby cat, sports car, pizza.
"""

if __name__ == "__main__":
    raise NotImplementedError
