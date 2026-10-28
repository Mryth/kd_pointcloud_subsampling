# What we've done so far — explained simply

A plain-language walkthrough of the project: the idea, every step, why we made each choice,
and what the results actually mean. (The [`README.md`](README.md) is the terse reference; this
file is the "understand it properly" companion.)

---

## 1. The assignment, in plain words

The course asked for an **original research question** about **point clouds applied to medical
imaging**, with a focus on **Knowledge Distillation (KD)**. Rules: the data must be **medical**;
we're free to pick the dataset and the neural-network architectures (but must *justify* them);
and we must deliver **reproducible code + numbers + an honest critical analysis** (even if our
hypothesis turns out wrong).

Our assigned sub-topic: **"Robustness to sub-sampling (classification)."**

Three terms to unpack before anything else:

- **Point cloud** = a shape represented as a *set of 3D points* `(x, y, z)` instead of a grid
  of pixels/voxels. Think of a scatter of dots floating in space that outline an object.
- **Sub-sampling** = keeping only *some* of those points (e.g. 64 instead of 1024). Real
  sensors give fewer points when scans are faster/cheaper/lower-resolution, so a model that
  still works with few points is valuable.
- **Knowledge Distillation (KD)** = train a small "student" network to imitate a bigger, better
  "teacher" network, so the small one punches above its weight.

**So the big question of the field here is:** *can distillation make a small point-cloud
classifier that stays accurate even when it only sees a few points?*

---

## 2. The research question we chose

> **Does distilling a teacher (trained on dense point clouds) into a lightweight student make
> that student more robust to sub-sampling at test time — for lung-nodule classification?**

**Hypothesis H1:** a *distilled* student degrades more gracefully as we remove test points
than the *same* student trained the ordinary way (plain cross-entropy).

Why it's a good question: it's **specific** (robustness, not just accuracy), **falsifiable**
(we can plot a curve and check), and **practically motivated** (sparse medical scans are real).

> ✅ **The project is complete.** Sections 4–9 cover the first half (data, models, baselines);
> sections 10–14 cover the second half (statistics over 5 seeds, the distillation itself, and
> the final verdict). **Spoiler: H1 is refuted** — and section 12 explains why that is a good,
> defensible result.

---

## 3. The whole pipeline in one picture

```
CT volume (28×28×28 voxels)
        │   intensity-weighted voxel sampling
        ▼
Dense point cloud  (1024 points, each = x,y,z,intensity)
        │   train PointNet on dense clouds
        ▼
Teacher (0.8M params)  &  Student-from-scratch (54k params)
        │   at TEST time: keep only N random points  (N = 1024…16)
        ▼
Robustness curve: accuracy (AUC) vs number of points
        │   second half: repeat on 5 seeds + train distilled students
        ▼
Compare: KD students  vs  scratch  vs  "aug" control (same sparse inputs, no teacher)
```

Everything after the point-cloud step is "just" normal deep-learning on a table of numbers —
the clever part is turning a medical image into points and then measuring what happens as we
starve the model of points.

---

## 4. The data — NoduleMNIST3D

**What it is:** a public medical dataset (part of **MedMNIST v2**). Each sample is a tiny
**28×28×28 CT scan** of a **lung nodule**, labelled **benign (0)** or **malignant (1)**.
1158 train / 165 val / 310 test volumes.

**Why we chose it:**
- **Medical & public** — satisfies the imposed condition, open license.
- **Small** — fits a laptop GPU, so we can afford to train a teacher *and* a student *and* KD
  variants.
- **Naturally turns into point clouds** — a CT volume is a cube of brightness values; the
  bright voxels trace the nodule's structure, which we can sample as points.
- **Perfect for sub-sampling** — we build one dense cloud per nodule and then drop points.

**One catch — class imbalance:** only ~21–25 % of nodules are malignant. So plain **accuracy
is misleading** (a lazy model that always says "benign" already scores ~75 %). We therefore
report **ROC-AUC** (ranking quality, threshold-free) and **balanced accuracy**, and we weight
the training loss by the inverse class frequencies so the rare malignant class isn't ignored.

---

## 5. From CT volumes to point clouds (the key step)

A model can't eat a 3D image directly as a point cloud — we have to *make* the points. Our
recipe (`src/data.py`):

1. **Normalise** each volume's intensities to `[0, 1]`.
2. **Sample voxels weighted by intensity** (`weight ∝ intensity^1.5`): brighter voxels — the
   actual nodule tissue — are more likely to become points, so the cloud concentrates on the
   structure rather than empty air.
3. Take **1024 points**; each point carries **4 numbers**: its position `(x, y, z)` **plus its
   intensity** (a 4th feature that helps distinguish nodule tissue).
4. **Normalise the coordinates** — centre them and scale to the unit ball — so every nodule
   lives in the same coordinate range regardless of size/position.

The picture `presentation/assets/pointcloud_subsampling.png` shows one nodule rendered this
way at 1024 → 256 → 64 → 16 points: you can literally watch the shape dissolve as points are
removed. That dissolving is exactly what "robustness to sub-sampling" is about.

---

## 6. The models — PointNet (teacher & student)

**What PointNet is (simply):** a neural net that eats an *unordered set* of points. It pushes
every point through the *same* small MLP, then **collapses all points into one vector with a
`max` over the set** ("global max-pooling"), and classifies from that vector. The `max` is the
magic: it's **permutation-invariant** (the order of points doesn't matter) and it means the
prediction depends on the *strongest* response across points — which makes it easy to reason
about what happens when points disappear.

**Why PointNet** (and not DGCNN / Point Transformer): it's the canonical, simplest point-cloud
classifier; its single global max-pool makes the sub-sampling analysis clean; and it's light
enough to train many variants on a laptop GPU. (The heavier architectures are a possible
extension.)

Same architecture, two sizes:
- **Teacher** — wide PointNet, embedding 1024, **~0.8 M parameters**. Trained on dense clouds.
  This is the "knowledgeable" model KD will later distill *from*.
- **Student** — narrow PointNet, embedding 256, **~54 k parameters** (~15× smaller). This is
  the model we ultimately want to be small *and* robust.

---

## 7. The robustness protocol (how we actually measure it)

This is the heart of the project:

1. Build each nodule's **dense cloud (1024 points)** once.
2. **Train** on those dense clouds (with a tiny bit of random "jitter" for regularisation).
3. **At test time**, keep a **random subset of N points** of the *same* cloud, for
   `N ∈ {1024, 512, 256, 128, 64, 32, 16}`.
4. Measure **ROC-AUC at each N** → the **robustness curve**.

Dropping points from a fixed dense cloud (rather than re-scanning) is the standard, controlled
way to probe sub-sampling robustness: the information content shrinks, everything else is held
equal.

---

## 8. The results (real numbers, on the test set)

ROC-AUC as we remove points:

| # points | Teacher (0.8M) | Student-scratch (54k) |
|---:|:---:|:---:|
| 1024 | 0.791 | **0.810** |
| 512 | 0.776 | 0.785 |
| 256 | 0.773 | 0.779 |
| 128 | 0.726 | 0.731 |
| 64 | **0.708** | 0.677 |
| 32 | **0.667** | 0.616 |
| 16 | 0.572 | 0.568 |

(Plot: `presentation/assets/robustness_curves.png`.)

> ⚠️ These are **single-seed** numbers. Section 10 shows that some of them do not survive
> re-running with other seeds — read section 9 as "what we believed after the first half".

---

## 9. What the results mean (critical analysis)

**① The phenomenon is real.** Both models fall from AUC ≈ 0.80 (1024 points) to ≈ 0.57 (16
points). Sub-sampling genuinely hurts — so there *is* something for KD to improve. Good.

**② An honest surprise: the tiny student matches the big teacher** at full density (0.810 vs
0.791). Why? NoduleMNIST3D is **small and hard** (even dedicated 3D CNNs cap around AUC ~0.87),
so the teacher's extra capacity doesn't buy accuracy — it likely just **overfits** 1158
examples. This is the kind of result the assignment explicitly wants us to report honestly.

**③ But the teacher is more robust where it counts.** At heavy sub-sampling the teacher holds
up better (64 pts: **0.708 vs 0.677**; 32 pts: **0.667 vs 0.616**). So the teacher *does* have
something worth transferring — **not accuracy, but low-density robustness.**

**→ This reframes our hypothesis honestly.** KD here shouldn't aim to beat the teacher on
accuracy (the student already ties it); its job is to **transfer the teacher's graceful
degradation under sparsity** to the student. That's a sharper, more defensible version of H1.

**On metrics:** we read **AUC and balanced-accuracy**, not raw accuracy — the ~1:4 imbalance
makes accuracy look good for the wrong reasons (e.g. the teacher's 0.571 accuracy at 1024 pts
sits next to a healthy 0.791 AUC; the AUC is the trustworthy number).

---

## 10. Second half, step 1 — are the first-half numbers real? (5 seeds)

A single training run depends on luck: the random initial weights, the order of the batches,
which points get dropped. With only 310 test nodules (~64 malignant), that luck alone moves the
AUC by ~0.02 — the same size as the effects we were discussing. So before any KD we re-trained
teacher and student on **5 seeds** (`run_seeds.py`) and averaged each point of the curve over
5 random sub-samples. We report **mean ± std** and a **95 % confidence interval**.

| # points | Teacher | Student-scratch |
|---:|:---:|:---:|
| 1024 | 0.765 ± 0.008 | 0.770 ± 0.028 |
| 128 | 0.720 ± 0.029 | 0.710 ± 0.020 |
| 32 | 0.671 ± 0.030 | 0.640 ± 0.019 |
| 16 | 0.605 ± 0.030 | 0.573 ± 0.021 |

What changed vs. section 9:
- The "student beats teacher" surprise **disappears**: they are equal at full density. The
  student is just ~3× more unstable from seed to seed.
- The teacher's robustness advantage is **small** and only clear at ≤ 32 points.

Lesson: *one seed is an anecdote, not a result.*

---

## 11. Second half, step 2 — the distillation variants

All variants live in one training loop (`src/kd.py`). In plain words:

| variant | what the student learns from |
|---|---|
| **scratch** | the labels only, on dense clouds |
| **aug** (the *control*) | the labels only, but on **randomly thinned** clouds (16–1024 points each batch) — **no teacher** |
| **kd** (Hinton) | labels + the teacher's "soft" probabilities, on dense clouds |
| **kd_robust** (our idea) | labels + the teacher's soft probabilities — the teacher sees the **dense** cloud, the student a **thinned** one |
| **kd_robust_feat** | same + copy the teacher's internal global vector `g` |
| **kd_robust_rt** | same as kd_robust, but from a **robust teacher** (itself trained on thinned clouds) |

**Why the `aug` control is essential.** `kd_robust` changes *two* things at once compared to
scratch: the student sees thinned clouds (an augmentation) **and** it gets the teacher's
targets. If kd_robust wins, which of the two did it? `aug` has the first ingredient without the
second. So: **kd_robust − aug = the real contribution of the teacher.**

**Soft probabilities, temperature T, weight α.** The teacher doesn't just say "malignant"; it
says "72 % malignant". Dividing the logits by a temperature `T` softens these probabilities so
the student sees more of the teacher's nuance; `α` balances "match the labels" vs "match the
teacher". We picked `T = 2, α = 0.7` on the **validation** set (never the test set); all 9
settings tried were within 0.83–0.85 val AUC, so the choice barely matters.

**Fair comparison.** Every variant is trained on the same 5 seeds (seed *s* distils from the
teacher of seed *s*) and picks its best epoch with the same rule. Comparisons are **paired**:
for each seed we compute "variant A − variant B", then test whether that difference is zero.

---

## 12. The results of the second half — and the verdict

Test ROC-AUC, mean over 5 seeds:

| # points | teacher | scratch | **aug** | kd | **kd_robust** | kd_robust_rt |
|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1024 | 0.765 | 0.763 | 0.791 | 0.779 | 0.793 | 0.794 |
| 128 | 0.720 | 0.714 | 0.779 | 0.739 | 0.778 | 0.775 |
| 64 | 0.668 | 0.660 | 0.743 | 0.673 | 0.740 | 0.739 |
| 16 | 0.605 | 0.585 | 0.648 | 0.582 | 0.643 | 0.652 |

(Plots: `robustness_curves_kd.png`, `robustness_curves_robust_teacher.png`.)
(Scratch here is 0.763, not 0.770 as in section 10: it was re-trained with the same
epoch-selection rule as the KD variants, so that every column is comparable.)

**① KD robust looks like a big win…** vs scratch: **+0.03 to +0.08 AUC**, on 5 seeds out of 5,
p ≤ 0.01. If we had stopped here, we would have announced "KD works".

**② …but the control does exactly the same.** `kd_robust − aug` ≈ **0** at every density
(at most ±0.004, p ≥ 0.69). The teacher adds **nothing measurable**. The robustness comes
entirely from **training on thinned clouds**. Feature KD and Hinton KD don't help either.

**③ Even a robust teacher doesn't help.** Natural objection: "the dense teacher had no
robustness to give". So we made the teacher robust. Result: the 0.8M robust teacher is still
slightly **worse** than the 54k `aug` student (it wins 0 of 5 seeds at 64 points). Distilling
from it gives ≈ 0 gain; trusting it more (α = 0.3) makes the student *worse*.

**→ Verdict: H1 is refuted.** And we can say *why*: knowledge distillation can only transfer
what the teacher knows *beyond* the student. With 1158 training nodules, the big teacher just
overfits — it never knows more than the small student trained the right way.

**Two side findings worth remembering:**
- **Thinning regularises.** `aug` is better even at full density (+0.03): randomly dropping
  points acts like dropout.
- **The decision threshold depends on the density.** The AUC (how well the model *ranks*
  patients) survives thinning, but at ≤ 64 points the dense-trained models predict "benign" for
  everyone (balanced accuracy 0.5) — fewer points ⇒ smaller max-pooled values ⇒ shifted logits.
  Models trained on thinned clouds have the opposite problem at 1024 points. A real system
  would need a threshold calibrated per density.

---

## 13. Limits and what we would do next

- **We erased nodule size.** Scaling every cloud to the unit ball removes the nodule's size —
  one of the strongest malignancy cues for radiologists. Likely why every model caps near
  AUC 0.8. Fix: normalise with a fixed constant, or add the radius as a feature.
- **Small test set.** 310 nodules and 5 seeds: a real but tiny KD effect (±0.01) would be
  invisible.
- **A teacher that truly knows more.** The promising version of this project: a 3D CNN on the
  full voxel grid (more information than any point cloud) distilled into the point-cloud
  student — *cross-modal* KD.
- **Structured thinning** (holes, missing slices) instead of only random point removal.

---

## 14. Done

- ✅ Data pipeline, now fully reproducible (`prepare_data.py` rebuilds the clouds bit-for-bit).
- ✅ Teacher / student baselines, single seed (first half) and 5 seeds (second half).
- ✅ Six training variants incl. the `aug` control and a robust teacher.
- ✅ Statistics: mean ± std, bootstrap CIs, paired tests.
- ✅ Honest conclusion: H1 refuted, with the reason and the limits.

---

## 15. Where everything lives

```
kd_pointcloud_subsampling/
├── data/
│   ├── nodulemnist3d.npz          # raw dataset (MedMNIST v2)
│   └── processed_clouds.npz       # precomputed 1024-pt clouds (train/val/test)
├── src/
│   ├── data.py      # volumes → point clouds + sub-sampling Dataset
│   ├── models.py    # scalable PointNet (teacher / student)
│   ├── engine.py    # train / evaluate / robustness curves / bootstrap CI
│   └── kd.py        # all KD variants + the aug control
├── prepare_data.py                # rebuilds processed_clouds.npz
├── run_half.py                    # first half: one seed, teacher + student
├── run_seeds.py                   # baselines on 5 seeds
├── run_kd.py                      # (T, α) sweep + 5 student variants × 5 seeds
├── run_robust_teacher.py          # robust teacher + distilled students
├── notebooks/01_pointclouds_robustness.ipynb   # narrated, reproducible (FR)
├── models/teacher.pt, student_scratch.pt  (+ per-seed *_s{0..4}.pt, not versioned)
├── results*.json, robustness_curves*.png
├── presentation/                  # deck + assets + PROMPT.md (see that folder)
├── README.md                      # terse reference
└── explain.md                     # this file
```

**To reproduce the numbers:** create the venv (`uv venv .venv && uv pip install --python
.venv/bin/python -r requirements.txt`), then run `run_seeds.py` → `run_kd.py` →
`run_robust_teacher.py` (~35 min in total on a GTX 1650). The dataset is already downloaded;
nothing else needs the network.

*(Written in English to match the README; happy to give a French version for the report.)*
