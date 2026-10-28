"""Follow-up to run_kd.py: distil from a ROBUST teacher.

run_kd.py showed kd_robust == aug: a teacher trained on dense clouds has no low-density
robustness to transfer. Here the teacher itself is trained with random sub-sampling
(mode "aug"), then students are distilled from it with kd_robust. The decisive comparison
is still against student_aug (same inputs, no teacher), loaded from results_kd.json.

    python run_robust_teacher.py        # needs results_kd.json (python run_kd.py)
"""
import sys, json, time, argparse; sys.path.insert(0, "src")
import numpy as np, torch
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data import NoduleCloudDataset
from models import teacher as make_teacher, student as make_student
from kd import train_kd
from engine import robustness_curve_multi, bootstrap_ci, DEVICE

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--draws", type=int, default=5)
args = ap.parse_args()

prev = json.load(open("results_kd.json"))
T, ALPHA, VAL_NS = prev["config"]["T"], prev["config"]["alpha"], tuple(prev["config"]["val_ns"])
NS = [r["n_points"] for r in prev["results"]["teacher"]]
print(f"device: {DEVICE} | T={T} alpha={ALPHA}")
d = np.load("data/processed_clouds.npz")
yb = d["train_labels"]; cw = [len(yb) / (2 * np.bincount(yb)[c]) for c in [0, 1]]


def ds(split, seed, train=False):
    return NoduleCloudDataset(d[f"{split}_clouds"], d[f"{split}_labels"], train=train, seed=seed)


def fit(model, seed, **kw):
    return train_kd(model, ds("train", seed, True), ds("val", seed), epochs=args.epochs,
                    class_weight=cw, val_ns=VAL_NS, seed=seed, log=False, **kw)


# alpha = weight of CE; the lower alpha variant leans harder on the (now robust) teacher
VARIANTS = {"student_kd_robust_rt": ALPHA, "student_kd_robust_rt_a0.3": 0.3}
runs = {"teacher_aug": [], **{k: [] for k in VARIANTS}}
for seed in range(args.seeds):
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    teacher = fit(make_teacher(), seed, mode="aug")
    torch.save(teacher.state_dict(), f"models/teacher_aug_s{seed}.pt")
    runs["teacher_aug"].append(robustness_curve_multi(teacher, ds("test", seed), NS, args.draws))
    print(f"seed {seed} {'teacher_aug':26s} ({time.time()-t0:4.0f}s) auc: "
          + " ".join(f"{r['n_points']}:{r['auc']:.3f}" for r in runs["teacher_aug"][-1]))
    for name, a in VARIANTS.items():
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        m = fit(make_student(), seed, teacher=teacher, mode="kd_robust", T=T, alpha=a)
        torch.save(m.state_dict(), f"models/{name}_s{seed}.pt")
        runs[name].append(robustness_curve_multi(m, ds("test", seed), NS, args.draws))
        print(f"seed {seed} {name:26s} ({time.time()-t0:4.0f}s) auc: "
              + " ".join(f"{r['n_points']}:{r['auc']:.3f}" for r in runs[name][-1]))

summary = {}
for name, curves in runs.items():
    summary[name] = []
    for i, n in enumerate(NS):
        row = {"n_points": n}
        for key in ["auc", "balacc", "f1"]:
            v = [c[i][key] for c in curves]; mean, lo, hi = bootstrap_ci(v)
            row[key] = {"mean": mean, "std": float(np.std(v, ddof=1)), "ci95": [lo, hi], "per_seed": v}
        summary[name].append(row)
for k in ["teacher", "student_scratch", "student_aug", "student_kd_robust"]:
    summary[k] = prev["results"][k]

PAIRS = [("student_kd_robust_rt", "student_aug"), ("student_kd_robust_rt_a0.3", "student_aug"),
         ("student_kd_robust_rt", "student_kd_robust"), ("teacher_aug", "student_aug")]
paired = {}
for a, b in PAIRS:
    paired[f"{a} - {b}"] = []
    for i, n in enumerate(NS):
        diff = (np.array(summary[a][i]["auc"]["per_seed"][:args.seeds])
                - np.array(summary[b][i]["auc"]["per_seed"][:args.seeds]))
        mean, lo, hi = bootstrap_ci(diff)
        paired[f"{a} - {b}"].append({"n_points": n, "mean_diff": mean, "ci95": [lo, hi],
                                     "p_ttest": float(stats.ttest_1samp(diff, 0).pvalue),
                                     "wins": int((diff > 0).sum())})
json.dump({"config": {**vars(args), "T": T, "alpha": ALPHA, "val_ns": VAL_NS},
           "results": summary, "paired_auc": paired}, open("results_robust_teacher.json", "w"), indent=2)

SHOW = ["teacher", "teacher_aug", "student_scratch", "student_aug", "student_kd_robust", *VARIANTS]
for key in ["auc", "balacc"]:
    print(f"\n{key} (mean ± std over {args.seeds} seeds)")
    print(f"{'N':>5} " + "".join(f"{n.replace('student_', ''):>17s}" for n in SHOW))
    for i, n in enumerate(NS):
        print(f"{n:>5} " + "".join(f"{summary[m][i][key]['mean']:>10.3f}±{summary[m][i][key]['std']:.3f}" for m in SHOW))
print("\npaired AUC differences: mean [95% CI]  p(t-test)  wins/seeds")
for k, rows in paired.items():
    print(" ", k); print("    " + "  ".join(f"{r['n_points']}:{r['mean_diff']:+.3f}[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p={r['p_ttest']:.2f} {r['wins']}/{args.seeds}" for r in rows))

fig, ax = plt.subplots(figsize=(8, 5))
for name in SHOW:
    mu = np.array([r["auc"]["mean"] for r in summary[name]]); sd = np.array([r["auc"]["std"] for r in summary[name]])
    ax.plot(NS, mu, "--" if name.startswith("teacher") else "-", marker="o", label=name)
    ax.fill_between(NS, mu - sd, mu + sd, alpha=.1)
ax.set_xscale("log", base=2); ax.invert_xaxis(); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_xlabel("# points at inference"); ax.set_ylabel("test ROC-AUC")
ax.set_title(f"Distilling from a robust (sub-sampling-trained) teacher — {args.seeds} seeds")
fig.tight_layout(); fig.savefig("robustness_curves_robust_teacher.png", dpi=120)
print("\nsaved results_robust_teacher.json + robustness_curves_robust_teacher.png")
