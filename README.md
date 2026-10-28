# KD & Robustness to Sub-sampling — Medical Point Clouds

Point clouds applied to **medical imaging** with a focus on **Knowledge Distillation (KD)**.
Assigned subject: **Robustness to sub-sampling (classification)**.

> **Status: first half of the project.** This delivers the research framing, the data
> pipeline (volumes → point clouds), the teacher + from-scratch student **baselines**, and the
> **robustness-to-sub-sampling protocol + curves**. The **second half** (the KD itself + the
> comparative analysis) is specified at the end.

## Research question & hypothesis

**RQ.** Does knowledge distillation from a teacher trained on *dense* point clouds make a
lightweight student **more robust to point sub-sampling at inference** for lung-nodule
classification?

**H1.** A *distilled* student degrades more gracefully as test points are removed than the
same student trained from scratch (cross-entropy only).

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
sub-sampling easy to reason about; light enough to train teacher + student + KD variants on a
laptop GPU. **Teacher**: emb 1024 (~0.8M params). **Student**: emb 256 (~54k params) — the
model we want to make robust. (DGCNN / Point Transformer are possible heavier extensions.)

## Protocol

1. Build a **dense 1024-point** cloud per nodule (intensity-weighted voxel sampling, coords
   centred + scaled to the unit ball).
2. Train on dense clouds (light jitter augmentation).
3. **Robustness curve**: evaluate the *same* model while keeping a random subset of
   N ∈ {1024, 512, 256, 128, 64, 32, 16} points.

<p align="center">
  <img src="presentation/assets/pointcloud_subsampling.png" alt="Same nodule point cloud sub-sampled from 1024 down to 16 points" width="100%">
</p>
<p align="center"><em>Same nodule, sub-sampled from dense (1024 pts) to very sparse (16 pts). This is exactly the degradation the robustness curve below measures.</em></p>

## Results (baselines, test ROC-AUC)

| # points | Teacher (0.8M) | Student-scratch (54k) |
|---:|:---:|:---:|
| 1024 | 0.791 | **0.810** |
| 512 | 0.776 | 0.785 |
| 256 | 0.773 | 0.779 |
| 128 | 0.726 | 0.731 |
| 64 | **0.708** | 0.677 |
| 32 | **0.667** | 0.616 |
| 16 | 0.572 | 0.568 |

<p align="center">
  <img src="presentation/assets/robustness_curves.png" alt="ROC-AUC vs number of points at inference, teacher vs student-scratch" width="600">
</p>

## Critical analysis

- **Both models degrade** under sub-sampling (AUC 0.80 → 0.57): the phenomenon is real and
  measurable — the playground for KD.
- **Honest surprise**: the 54k student **matches/beats** the 0.8M teacher at full density.
  NoduleMNIST3D is small and hard (AUC caps ~0.8 even for 3D CNNs); the teacher's extra
  capacity doesn't help and may overfit.
- **But** the teacher degrades **more gracefully at heavy sub-sampling** (64 pts: 0.708 vs
  0.677; 32 pts: 0.667 vs 0.616). This re-frames H1 honestly: KD shouldn't aim to beat the
  teacher's accuracy (the student already does) but to **inherit its low-density robustness**.

## Second half (to do) — the distillation

1. **Logit KD (Hinton)**: `α·CE + (1-α)·KL(softmax(z_s/T) ‖ softmax(z_t/T))`, teacher frozen.
2. **Robust KD (the original idea)**: teacher sees the **dense** cloud, student sees a
   **sub-sampled** cloud and must match the teacher's dense output → learns sparsity robustness.
3. **Feature KD**: align the global feature `g` (already returned by `forward`).
4. **Compare** `student_KD` vs `student_scratch` vs `teacher` robustness curves; test H1;
   conclude honestly (including if H1 is not confirmed).

## Reproduce

```bash
# environment: reuses the GPU torch venv (torch 2.12+cu130). Non-torch deps install offline
# from the uv cache; torch itself can't be fetched offline, so the existing GPU build is reused.
python data/... (already downloaded nodulemnist3d.npz from MedMNIST v2 / Zenodo)
python <venv>/.. run_half.py            # script version: trains both + saves curves/plot/json
# or run notebooks/01_pointclouds_robustness.ipynb  (kernel: "Python (KD PointCloud)")
```

Artifacts: `models/teacher.pt`, `models/student_scratch.pt`, `results.json`,
`robustness_curves.png`, `data/processed_clouds.npz`.

**Files.** `src/data.py` (volumes→clouds + sub-sampling), `src/models.py` (scalable PointNet),
`src/engine.py` (train/eval + robustness curve), `run_half.py` (orchestration),
`notebooks/01_pointclouds_robustness.ipynb` (narrated, reproducible).
