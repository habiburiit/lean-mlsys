# LEAN: Loss-Engineered Modulation of the Weight-Sensitivity Distribution

Anonymous code release for the MLSys submission. Everything needed to reproduce the
paper's experiments is here: the LEAN training objective for vision models and
language models, the sensitivity measurement, the attacks (progressive bit search,
T-BFA, AttentionBreaker, SilentStriker), the selective code and the real BCH codec,
the tiered low-bit storage format, the published baselines, the hardware loader
(RTL, FPGA and ASIC flows) and the Timeloop/Accelergy energy model.

## Layout

```
code/      all Python and shell scripts, flat (run everything from inside code/)
hw/        the tiered-format weight loader: RTL, testbenches, DC synthesis, Vivado FPGA and SoC flows
energy/    Timeloop + Accelergy architecture, constraints and workload files
SUPPLEMENTARY.pdf   reproduction manual (this README in long form, with the paper's table map)
```

Scripts are named by what they produce. `lean_*.py` are the LEAN method and its
measurements, `run_*.sh` are the exact command sequences used for the paper, each one
writing a log that can be followed with `tail -f`.

## Environment

```
conda create -n lean python=3.10
conda activate lean
pip install -r requirements.txt
```

CUDA GPUs are assumed (48 GB for the 7B/8B language models in bfloat16; 24 GB is
enough for everything else). Vision data: CIFAR-10/100 are downloaded by torchvision;
ImageNet experiments use the 50k validation set split 40k/10k as described in the
paper, expected at `./imagenet-val`. Language models are loaded from the Hugging Face
hub by name.

## Core files

| file | what it is |
|---|---|
| `code/lean_sensreg.py` | the LEAN objective for CNNs (`train`), the sensitivity measurement (`hist`), post-training quantization (`ptq`), real sign-bit flips (`flip`), the lambda sweep (`knob`) and the figures |
| `code/lean_sensreg_llm.py` | the same objective for the linear layers of language models (WikiText-2 fine-tune) |
| `code/lean_llm_train_fd3.py`, `lean_llm_train_big.py` | the finite-difference form of the tail term for 1.5B and larger models |
| `code/lean_pcstep.py`, `lean_train_pcstep_cnn.py` | the tail term evaluated at the per-channel INT8 step |
| `code/pbs_four.py`, `lean_pbs_codec*.py` | progressive bit search with no code, random code, selective code, the real BCH codec and the key-aware attacker |
| `code/lean_tbfa*.py` | targeted bit-flip attack on held-out data |
| `code/attnbreaker_core.py`, `lean_llm_attnbreaker_w.py`, `lean_llm_showcase*.py` | AttentionBreaker (GenBFA) on INT8 language models with a weight-level code |
| `code/lean_llm_silentstriker_w.py` | SilentStriker on the same models |
| `code/lean_llm_harness.py`, `lean_llm_tasks.py`, `lean_llm_ppl_owt.py` | six-task evaluation, MMLU and OpenWebText perplexity |
| `code/lean_train_qat_group5.py`, `lean_train_qat_mp9.py`, `lean_eval_mp.py` | the sensitivity-ordered 8/6/4/3-bit format (per-block and per-weight tiers) |
| `code/lean_nvfp4.py`, `lean_train_qat_nvfp4*.py` | the NVFP4 mode |
| `code/lean_train_hessian.py`, `lean_train_pc_wcat.py`, `lean_train_far.py`, `lean_train_lsq.py` | Hessian-aware training, Piece-wise Clustering, WCAT, Forget-and-Rewire, LSQ and KURE under the paper's protocol |
| `code/lean_train_inset*.py` | LEAN switched on in the last epochs of training |
| `code/lean_archs.py` | MobileNet-V2, EfficientNet-B0, ViT-B/16 and Swin-T |
| `code/lean_overhead.py` | training-time and memory overhead |
| `hw/` | see `hw/README.md` |

## Paper table and figure map

| paper | scripts |
|---|---|
| Table 1, Figure 3, Appendix B | `run_sensreg_all_models.sh`, `run_all_sensreg.sh`, `lean_sensreg.py hist/knob/figure` |
| Table 2 (ablation, real flips) | `run_lean_ablation.sh`, `run_lean_ablation2.sh`, `lean_truedamage.py`, `lean_train_gradnorm.py` |
| Appendix D (per-channel ablation) | `run_c4_cifar_pc.sh`, `run_c4_imnet_train.sh`, `run_c4_imnet_pbs.sh`, `run_c4_imnet_pcstep.sh` |
| Table 4 (published defenses) | `run_lean_e3_pbsfour.sh`, `pbs_four.py`, `plot_compare_hessian.py` |
| Table 5 (WCAT, six runs) | `run_lean_e3_seeds.sh` |
| Table 6, Appendix E (T-BFA) | `run_lean_tbfa*.sh`, `run_fill_part2_tbfa.sh` |
| Table 7, Appendix F (parity, codec, code-aware) | `run_lean_codec.sh`, `run_fill_part1_codec.sh`, `run_c5_codeaware.sh`, `lean_ecc_four.py` |
| Table 8, Appendix G (storage, low-bit, NVFP4) | `run_lean_group5.sh`, `run_lean_mp9.sh`, `run_lean_nvfp4*.sh`, `run_trunc_sweep.sh` |
| Table 9, Appendix L (hardware) | `hw/` |
| Table 10, Appendix J (language models) | `run_sensreg_llm.sh`, `run_lean_llm_full*.sh`, `run_lean_attnbreaker_w*.sh`, `run_lean_silentstriker*.sh`, `run_lean_qwen7b_*.sh`, `run_qwen7b_*.sh`, `run_llama8b_*.sh`, `run_lean_owt.sh` |
| Appendix C (LEAN inside training) | `run_lean_inset*.sh` |
| Appendix H (other architectures) | `run_lean_archs*.sh` |
| Appendix I (energy) | `energy/run_all.sh`, `lean_energy.py` |
| Appendix K (further variants) | `run_lean_e6_logit.sh`, `run_lean_e7_band.sh`, `run_lean_e8_flipaware.sh`, `run_lean_archs_dw*.sh`, `run_lean_archs_clamp_all.sh` |

## Quick start: one model end to end

```
cd code
# 1. LEAN fine-tune from a clean checkpoint, ResNet-18 CIFAR-10, lambda = 40
python lean_sensreg.py train --dataset cifar10 --arch resnet18 --init artifacts/models/cifar10/resnet18/model_float32.pth \
       --reg both --lam 40 --epochs 10 --out artifacts/models/cifar10/resnet18/model_float32_sensreg_lam40.pth
# 2. sensitivity distribution of the LEAN model against the baseline's HIGH/MID thresholds
python lean_sensreg.py hist --dataset cifar10 --arch resnet18 --ckpt artifacts/models/cifar10/resnet18/model_float32.pth \
       --ckpt2 artifacts/models/cifar10/resnet18/model_float32_sensreg_lam40.pth --out sensreg_results/cifar10_resnet18/hist_lam40
# 3. progressive bit search at 100 to 500 flips with the selective 1% code
python pbs_four.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.01 --flips 100,200,300,500 --label lam40
```

`run_sensreg_all_models.sh` does steps 1 and 2 for all nine model/dataset pairs and the
lambda sweep; `run_lean_e3_pbsfour.sh` does step 3 for the baseline, the published
defenses and LEAN. Clean checkpoints are trained with `gen_models.py` (CIFAR) or taken
from torchvision (ImageNet) and stored under `artifacts/models/<dataset>/<arch>/`.

Every `run_*.sh` has a header comment stating what it reproduces, its arguments
(usually the GPU index) and where it writes its log.

## Hardware

`hw/README.md` describes the loader. In short: `python lean_pack.py` packs a
checkpoint into the tiered format, `lean_hw_agent.py` runs the VCS simulation and
Design Compiler synthesis, `fpga/run_fpga.sh` the Vivado implementation and
`fpga/run_xsim.sh` the post-route simulation, `fpga/run_cycles.sh` the cycle-accurate
throughput and latency measurement, `run_power_saif.sh` the simulation-driven power,
and `fpga_soc/run_soc.sh` the Zynq block design.

## License

MIT. Anonymous authors.
