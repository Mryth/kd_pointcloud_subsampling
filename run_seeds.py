"""Baselines over several seeds: teacher + student_scratch, robustness curves with mean ± std
and a 95% bootstrap CI over seeds. Single-seed differences of ~0.03 AUC on a 310-sample
test set are not interpretable; this script is the statistical baseline for the KD study.

    python run_seeds.py                 # 5 seeds, 40 epochs
    python run_seeds.py --seeds 3 --epochs 20
"""
import sys, json, time, argparse; sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data import NoduleCloudDataset, N_FULL
from models import teacher as make_teacher, student as make_student, n_params
from engine import train_model, robustness_curve_multi, bootstrap_ci, DEVICE

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--draws", type=int, default=5, help="random sub-samples averaged per N at test")
args = ap.parse_args()

NS = [1024, 512, 256, 128, 64, 32, 16]
MODELS = {"teacher": make_teacher, "student_scratch": make_student}
print("device:", DEVICE)
d = np.load("data/processed_clouds.npz")
yb = d["train_labels"]; cw = [len(yb) / (2 * np.bincount(yb)[c]) for c in [0, 1]]

runs = {name: [] for name in MODELS}          # runs[name][seed] = curve
for seed in range(args.seeds):
    for name, make in MODELS.items():
        torch.manual_seed(seed); np.random.seed(seed)
        ds = lambda s, train=False: NoduleCloudDataset(d[f"{s}_clouds"], d[f"{s}_labels"],
                                                       train=train, seed=seed)
        t = time.time(); m = make()
        train_model(m, ds("train", True), ds("val"), epochs=args.epochs, class_weight=cw, log=False)
        curve = robustness_curve_multi(m, ds("test"), NS, n_draws=args.draws)
        runs[name].append(curve)
        torch.save(m.state_dict(), f"models/{name}_s{seed}.pt")
        print(f"seed {seed} {name:16s} ({time.time()-t:5.0f}s) auc: "
              + " ".join(f"{r['n_points']}:{r['auc']:.3f}" for r in curve))

# aggregate: per model, per N, per metric -> mean, std, 95% bootstrap CI over seeds
summary = {}
for name, curves in runs.items():
    summary[name] = {"params": n_params(MODELS[name]()), "curve": []}
    for i, n in enumerate(NS):
        row = {"n_points": n}
        for key in ["auc", "balacc", "acc", "f1"]:
            v = [c[i][key] for c in curves]
            mean, lo, hi = bootstrap_ci(v)
            row[key] = {"mean": mean, "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        "ci95": [lo, hi], "per_seed": v}
        summary[name]["curve"].append(row)
json.dump({"config": vars(args), "results": summary}, open("results_seeds.json", "w"), indent=2)

# table
print(f"\n{'N':>5} | " + " | ".join(f"{n:^30s}" for n in MODELS))
for i, n in enumerate(NS):
    cells = []
    for name in MODELS:
        a = summary[name]["curve"][i]["auc"]
        cells.append(f"{a['mean']:.3f}±{a['std']:.3f} [{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}]")
    print(f"{n:>5} | " + " | ".join(f"{c:^30s}" for c in cells))

# plot AUC and balanced accuracy, mean ± std band
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, key, lab in [(axes[0], "auc", "test ROC-AUC"), (axes[1], "balacc", "test balanced accuracy")]:
    for name in MODELS:
        c = summary[name]["curve"]
        mu = np.array([r[key]["mean"] for r in c]); sd = np.array([r[key]["std"] for r in c])
        ax.plot(NS, mu, marker="o", label=name); ax.fill_between(NS, mu - sd, mu + sd, alpha=.2)
    ax.set_xscale("log", base=2); ax.invert_xaxis(); ax.grid(alpha=.3)
    ax.set_xlabel("# points at inference"); ax.set_ylabel(lab)
axes[0].legend(); fig.suptitle(f"Robustness to sub-sampling — mean ± std over {args.seeds} seeds")
fig.tight_layout(); fig.savefig("robustness_curves_seeds.png", dpi=120)
print("\nsaved results_seeds.json + robustness_curves_seeds.png")
