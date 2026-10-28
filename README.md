# KD & Robustness to Sub-sampling — Medical Point Clouds

Point clouds applied to **medical imaging** with a focus on **Knowledge Distillation (KD)**.
Assigned subject: **Robustness to sub-sampling (classification)**.

> **TL;DR.** On NoduleMNIST3D point clouds, a distilled PointNet student is much more robust
> to sub-sampling than a student trained from scratch (+0.03 → +0.08 AUC, 5/5 seeds) — **but
> exactly as robust as a student trained with random point dropping and no teacher at all**.
> With the proper control, **H1 is refuted**: across logit KD, feature KD, a dense teacher and
> a robust teacher, the teacher's soft targets add nothing measurable. The gain is the
> augmentation. The reason: neither teacher knows more than the augmented student.

## Research question & hypothesis

**RQ.** Does knowledge distillation from a teacher trained on *dense* point clouds make a
lightweight student **more robust to point sub-sampling at inference** for lung-nodule
classification?

**H1.** A *distilled* student degrades more gracefully as test points are removed than the
same student trained without the teacher.

## Data — why NoduleMNIST3D

**NoduleMNIST3D** (MedMNIST v2): 28³ CT nodule volumes, binary (benign/malignant),
1158/165/310 train/val/test. Chosen because it is **medical & public**, **lightweight**
(laptop-GPU budget), **naturally convertible to point clouds** (intensity-weighted voxel
sampling → `(x,y,z,intensity)` points), and **ideal for sub-sampling** experiments (build a
dense 1024-point cloud, then drop points at test). Class imbalance ~21–25 % positive → we
report **ROC-AUC** and **balanced accuracy** (raw accuracy is misleading) and use
inverse-frequency class weights.

## Architectures — why PointNet

Vanilla **PointNet** (shared MLP + global max-pool, no T-Net) for both teacher and student:
canonical, permutation-invariant, and its single global max-pool makes robustness to
sub-sampling easy to reason about; light enough to train dozens of variants on a laptop GPU.
**Teacher**: emb 1024 (0.81M params). **Student**: emb 256 (54k params, 15× smaller).

## Protocol

1. **Clouds**: a dense 1024-point cloud per nodule (intensity-weighted voxel sampling, coords
   centred + scaled to the unit ball) — `prepare_data.py`.
2. **Robustness curve**: evaluate a trained model while keeping a random subset of
   N ∈ {1024, 512, 256, 128, 64, 32, 16} test points; each point of the curve is averaged over
   5 random sub-samples.
3. **Statistics**: every configuration is trained on **5 seeds**; we report mean ± std, a 95 %
   bootstrap CI over seeds, and **paired** comparisons (same seed ⇒ same teacher, same data
   order) with a t-test and the number of seeds won.
4. **Hyper-parameters** (T, α) chosen on the **validation** set only (sweep T ∈ {2,4,8},
   α ∈ {0.3,0.5,0.7}, all within 0.83–0.85 val AUC → insensitive). Retained: T = 2, α = 0.7.
5. **Model selection**: for every student variant, best epoch = mean val AUC at
   N ∈ {1024, 128, 32} (same rule for all, so variants differ only by their training signal).

### Variants (`src/kd.py`, one training loop)

| name | teacher | student input | loss |
|---|---|---|---|
| `scratch` | – | dense | CE |
| `aug` | – | random sub-sample, N ~ log-U[16, 1024] per batch | CE — **control for KD-robust** |
| `kd` | dense teacher, frozen | dense | α·CE + (1−α)·T²·KL (Hinton) |
| `kd_robust` | dense teacher sees the **dense** cloud | random sub-sample | α·CE + (1−α)·T²·KL |
| `kd_robust_feat` | idem | random sub-sample | + ‖P(g_s) − g_t‖² on L2-normalised global features |
| `kd_robust_rt` | **robust** teacher (trained as `aug`) | random sub-sample | α·CE + (1−α)·T²·KL |

## Results (test ROC-AUC, mean ± std over 5 seeds)

| N | teacher | teacher_aug | scratch | aug | kd | kd_robust | kd_robust_feat | kd_robust_rt |
|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1024 | 0.765±.008 | 0.783±.018 | 0.763±.021 | 0.791±.020 | 0.779±.028 | 0.793±.020 | **0.796±.012** | 0.794±.015 |
| 512 | 0.762±.013 | 0.788±.019 | 0.758±.020 | 0.796±.017 | 0.779±.020 | 0.795±.021 | **0.798±.012** | 0.794±.016 |
| 256 | 0.748±.019 | 0.793±.023 | 0.746±.018 | **0.798±.015** | 0.765±.023 | 0.795±.019 | 0.797±.012 | 0.793±.019 |
| 128 | 0.720±.029 | 0.774±.023 | 0.714±.015 | **0.779±.011** | 0.739±.023 | 0.778±.008 | **0.779±.009** | 0.775±.017 |
| 64 | 0.668±.034 | 0.728±.016 | 0.660±.020 | 0.743±.015 | 0.673±.026 | 0.740±.013 | **0.744±.008** | 0.739±.013 |
| 32 | 0.671±.030 | 0.709±.011 | 0.654±.019 | 0.726±.017 | 0.666±.031 | 0.726±.023 | 0.722±.014 | **0.731±.014** |
| 16 | 0.605±.030 | 0.646±.016 | 0.585±.020 | 0.648±.023 | 0.582±.030 | 0.643±.020 | 0.648±.015 | **0.652±.018** |

Figures: `robustness_curves_seeds.png` (baselines), `robustness_curves_kd.png` (all KD
variants, AUC + balanced acc.), `robustness_curves_robust_teacher.png` (robust teacher).

### Paired AUC differences (mean over seeds, range across N)

| comparison | Δ AUC | significance |
|---|---|---|
| kd_robust − scratch | **+0.030 → +0.080** | p ≤ 0.01 at every N, 5/5 seeds |
| aug − scratch | **+0.028 → +0.082** | p ≤ 0.02 for N ≤ 512, 5/5 seeds |
| **kd_robust − aug** | **−0.004 → +0.002** | p ≥ 0.69 at every N |
| **kd_robust_rt − aug** | **−0.005 → +0.005** | p ≥ 0.45 at every N |
| kd_robust_rt (α = 0.3) − aug | −0.022 → +0.004 | worse at 16 pts (p = 0.02) |
| kd − scratch | −0.003 → +0.025 | n.s. (p ≥ 0.08) |
| kd_robust_feat − kd_robust | ≈ 0 | n.s. |
| teacher_aug − student aug | −0.017 → −0.002 | teacher never better |

## Critical analysis

1. **H1 is refuted.** Read alone, *kd_robust vs scratch* looks like a clear KD success. The
   `aug` control (same sub-sampled inputs, no teacher) reaches the same curve: the robustness
   comes entirely from **training on sparse clouds**, not from the teacher's soft targets.
   Without this control we would have reported a false positive.
2. **Why KD cannot help here: the teacher is never better than the student it teaches.**
   - The dense teacher is no more robust than the dense student (16 pts: 0.605 vs 0.585) and
     far below `aug` → nothing to transfer.
   - Made robust (`teacher_aug`), the 0.8M teacher is still slightly **below** the 54k `aug`
     student (0/5 seeds won at 64 pts): with 1158 training nodules, extra capacity overfits.
   - Leaning harder on it (α = 0.3) makes the student *worse*. KD transfers what the teacher
     knows beyond the student; here that set is empty.
3. **Single seeds were misleading.** The first-half single-seed table had the student beating
   the teacher at full density (0.810 vs 0.791); over 5 seeds they are equal (0.763 vs 0.765)
   and the student is ~3× less stable. Seed-level variance (std ≈ 0.02) is of the same order
   as many reported "effects".
4. **Sub-sampling regularises.** `aug` also improves *dense* AUC (+0.028): random point
   dropping acts like dropout on the point set.
5. **The decision threshold depends on density.** AUC (ranking) survives sub-sampling, but
   balanced accuracy collapses to 0.5 at ≤ 64 pts for dense-trained models (all-benign
   predictions: the max-pooled feature shrinks with N, shifting the logits). Sub-sampling-
   trained models have the mirror problem (bal. acc. ≈ 0.52 at 1024 pts, ≈ 0.66 at 64 pts),
   and KD from soft targets worsens it (kd_robust_rt ≈ 0.50 at 1024). A deployable model needs
   a **density-aware threshold** (calibrate on val per N, or feed N as an input).

### Limitations

- **Unit-ball normalisation erases nodule size**, and per-volume intensity scaling erases
  absolute density — both strong malignancy cues. This likely caps AUC near 0.8 for every model.
- Small dataset (310 test nodules, ~64 malignant) → wide CIs; 5 seeds give paired tests limited
  power (a ±0.01 KD effect would be undetectable).
- One architecture family (PointNet) for teacher and student; sub-sampling is random
  (uniform) only — no structured/region dropout.
- The (T, α) sweep used one seed and the dense teacher; the robust-teacher run reuses it.

### Perspectives

- **A teacher that genuinely knows more**: a 3D CNN on the full voxel grid distilled into the
  point-cloud student (cross-modal KD), or a teacher with size/HU information the student lacks.
- Keep absolute scale (normalise by a fixed constant, or add the radius as a feature).
- Density-aware threshold calibration; structured sub-sampling (local holes, slab removal).

## Reproduce

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt ipykernel medmnist
source .venv/bin/activate
python prepare_data.py          # rebuilds data/processed_clouds.npz (bit-identical, seed 0)
python run_half.py              # first half, single seed (original baseline)
python run_seeds.py             # teacher + scratch on 5 seeds          (~6 min, GTX 1650)
python run_kd.py                # (T, α) sweep + 5 student variants × 5 seeds (~15 min)
python run_robust_teacher.py    # robust teacher + distilled students   (~12 min)
```

Each script writes a `results_*.json` (mean, std, CI, per-seed values, paired tests) and a
`robustness_curves_*.png`. Per-seed checkpoints `models/*_s{seed}.pt` are not versioned
(regenerated by the scripts).

**Files.** `src/data.py` (volumes → clouds + sub-sampling), `src/models.py` (scalable
PointNet), `src/engine.py` (train/eval, robustness curves, bootstrap CI), `src/kd.py` (all
KD variants + augmentation control), `run_*.py` (experiments),
`notebooks/01_pointclouds_robustness.ipynb` (narrated first half + second-half results).
