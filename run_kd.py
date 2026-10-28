"""Second half: knowledge distillation vs baselines, robustness to sub-sampling.

1. Small (T, alpha) sweep for kd_robust on the VALIDATION set (seed 0) -> pick hyper-params.
2. Train every student variant on S seeds; seed s distils from models/teacher_s{s}.pt
   (produced by run_seeds.py). All variants use the same selection rule (mean val AUC
   over VAL_NS), so they differ only by the training signal.
3. Robustness curves (mean ± std, 95% bootstrap CI) + paired comparisons per seed.

    python run_kd.py                    # needs models/teacher_s*.pt (python run_seeds.py)
"""
import sys, json, time, argparse, itertools; sys.path.insert(0, "src")
import numpy as np, torch
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data import NoduleCloudDataset
from models import teacher as make_teacher, student as make_student
from kd import train_kd, val_score
from engine import robustness_curve_multi, bootstrap_ci, DEVICE

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--no-sweep", action="store_true", help="skip the sweep, use T=4, alpha=0.5")
args = ap.parse_args()

NS = [1024, 512, 256, 128, 64, 32, 16]
VAL_NS = (1024, 128, 32)
print("device:", DEVICE)
d = np.load("data/processed_clouds.npz")
yb = d["train_labels"]; cw = [len(yb) / (2 * np.bincount(yb)[c]) for c in [0, 1]]


def ds(split, seed, train=False):
    return NoduleCloudDataset(d[f"{split}_clouds"], d[f"{split}_labels"], train=train, seed=seed)


def load_teacher(seed):
    t = make_teacher(); t.load_state_dict(torch.load(f"models/teacher_s{seed}.pt", map_location="cpu"))
    return t


def fit(mode, seed, **kw):
    torch.manual_seed(seed); np.random.seed(seed)
    s = make_student()
    teacher = load_teacher(seed) if mode.startswith("kd") else None
    return train_kd(s, ds("train", seed, True), ds("val", seed), teacher=teacher, mode=mode,
                    epochs=args.epochs, class_weight=cw, val_ns=VAL_NS, seed=seed, log=False, **kw)


# 1. hyper-parameter sweep on validation (never touches the test set)
T, ALPHA = 4.0, 0.5
sweep = []
if not args.no_sweep:
    print("\n=== (T, alpha) sweep for kd_robust on val, seed 0 ===")
    for T_, a_ in itertools.product([2.0, 4.0, 8.0], [0.3, 0.5, 0.7]):
        m = fit("kd_robust", 0, T=T_, alpha=a_)
        v = val_score(m, ds("val", 0), VAL_NS)
        sweep.append({"T": T_, "alpha": a_, "val_auc": v}); print(f"  T={T_:<4} alpha={a_:<4} val auc {v:.3f}")
    best = max(sweep, key=lambda r: r["val_auc"]); T, ALPHA = best["T"], best["alpha"]
print(f"using T={T}, alpha={ALPHA}")

# 2. all variants x seeds
VARIANTS = {
    "student_scratch":   dict(mode="scratch"),
    "student_aug":       dict(mode="aug"),
    "student_kd":        dict(mode="kd", T=T, alpha=ALPHA),
    "student_kd_robust": dict(mode="kd_robust", T=T, alpha=ALPHA),
    "student_kd_robust_feat": dict(mode="kd_robust", T=T, alpha=ALPHA, feat_w=1.0),
}
runs = {name: [] for name in ["teacher", *VARIANTS]}
for seed in range(args.seeds):
    runs["teacher"].append(robustness_curve_multi(load_teacher(seed).to(DEVICE), ds("test", seed), NS, args.draws))
    for name, kw in VARIANTS.items():
        t = time.time(); m = fit(seed=seed, **kw)
        torch.save(m.state_dict(), f"models/{name}_s{seed}.pt")
        runs[name].append(robustness_curve_multi(m, ds("test", seed), NS, args.draws))
        print(f"seed {seed} {name:24s} ({time.time()-t:4.0f}s) auc: "
              + " ".join(f"{r['n_points']}:{r['auc']:.3f}" for r in runs[name][-1]))

# 3. aggregate
summary = {}
for name, curves in runs.items():
    summary[name] = []
    for i, n in enumerate(NS):
        row = {"n_points": n}
        for key in ["auc", "balacc", "f1"]:
            v = [c[i][key] for c in curves]; mean, lo, hi = bootstrap_ci(v)
            row[key] = {"mean": mean, "std": float(np.std(v, ddof=1)), "ci95": [lo, hi], "per_seed": v}
        summary[name].append(row)

# paired comparisons (same seed = same teacher, same data order)
PAIRS = [("student_kd_robust", "student_aug"), ("student_kd_robust", "student_scratch"),
         ("student_kd", "student_scratch"), ("student_kd_robust_feat", "student_kd_robust"),
         ("student_aug", "student_scratch")]
paired = {}
for a, b in PAIRS:
    paired[f"{a} - {b}"] = []
    for i, n in enumerate(NS):
        diff = np.array(summary[a][i]["auc"]["per_seed"]) - np.array(summary[b][i]["auc"]["per_seed"])
        mean, lo, hi = bootstrap_ci(diff)
        p = float(stats.ttest_1samp(diff, 0).pvalue)
        paired[f"{a} - {b}"].append({"n_points": n, "mean_diff": mean, "ci95": [lo, hi], "p_ttest": p,
                                     "wins": int((diff > 0).sum())})

json.dump({"config": {**vars(args), "T": T, "alpha": ALPHA, "val_ns": VAL_NS}, "sweep": sweep,
           "results": summary, "paired_auc": paired}, open("results_kd.json", "w"), indent=2)

for key in ["auc", "balacc"]:
    print(f"\n{key} (mean ± std over {args.seeds} seeds)")
    print(f"{'N':>5} " + "".join(f"{n[8:] if n.startswith('student_') else n:>16s}" for n in runs))
    for i, n in enumerate(NS):
        print(f"{n:>5} " + "".join(f"{summary[m][i][key]['mean']:>9.3f}±{summary[m][i][key]['std']:.3f}" for m in runs))
print("\npaired AUC differences: mean [95% CI]  p(t-test)  wins/seeds")
for k, rows in paired.items():
    print(" ", k); print("    " + "  ".join(f"{r['n_points']}:{r['mean_diff']:+.3f}[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p={r['p_ttest']:.2f} {r['wins']}/{args.seeds}" for r in rows))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, key, lab in [(axes[0], "auc", "test ROC-AUC"), (axes[1], "balacc", "test balanced accuracy")]:
    for name in runs:
        mu = np.array([r[key]["mean"] for r in summary[name]]); sd = np.array([r[key]["std"] for r in summary[name]])
        ls = "--" if name == "teacher" else "-"
        ax.plot(NS, mu, ls, marker="o", label=name); ax.fill_between(NS, mu - sd, mu + sd, alpha=.12)
    ax.set_xscale("log", base=2); ax.invert_xaxis(); ax.grid(alpha=.3)
    ax.set_xlabel("# points at inference"); ax.set_ylabel(lab)
axes[0].legend(fontsize=8)
fig.suptitle(f"KD vs baselines — robustness to sub-sampling (mean ± std, {args.seeds} seeds, T={T}, α={ALPHA})")
fig.tight_layout(); fig.savefig("robustness_curves_kd.png", dpi=120)
print("\nsaved results_kd.json + robustness_curves_kd.png")
