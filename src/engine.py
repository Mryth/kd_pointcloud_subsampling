"""Train / evaluate loops + the robustness-to-subsampling curve."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(ds, batch=32, shuffle=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, drop_last=False)


def train_model(model, train_ds, val_ds, epochs=30, lr=1e-3, class_weight=None, log=True):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)
    tl = make_loader(train_ds, shuffle=True)
    best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(x)
            loss = crit(logits, y)
            loss.backward(); opt.step()
        sched.step()
        m = evaluate(model, val_ds)
        if m["auc"] > best_auc:
            best_auc = m["auc"]; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if log and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:2d} | val acc {m['acc']:.3f} auc {m['auc']:.3f} balacc {m['balacc']:.3f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, ds, batch=64):
    model.eval().to(DEVICE)
    ys, ps, probs = [], [], []
    for x, y in make_loader(ds, batch=batch):
        logits, _ = model(x.to(DEVICE))
        p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        ps.append(logits.argmax(1).cpu().numpy()); probs.append(p); ys.append(y.numpy())
    y = np.concatenate(ys); pred = np.concatenate(ps); prob = np.concatenate(probs)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    return {"acc": accuracy_score(y, pred), "auc": auc,
            "balacc": balanced_accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}


def robustness_curve(model, test_ds, n_points_list):
    """Evaluate the SAME model while sub-sampling the test clouds to fewer points."""
    rows = []
    for n in n_points_list:
        test_ds.set_n_points(n)
        test_ds.rng = np.random.default_rng(0)      # fixed subsample for comparability
        m = evaluate(model, test_ds)
        rows.append({"n_points": n, **m})
    return rows


def robustness_curve_multi(model, test_ds, n_points_list, n_draws=5):
    """Like `robustness_curve`, but averages each metric over `n_draws` random sub-samples
    (removes the luck of one particular subset of points)."""
    rows = []
    for n in n_points_list:
        test_ds.set_n_points(n)
        ms = []
        for k in range(n_draws):
            test_ds.rng = np.random.default_rng(k)
            ms.append(evaluate(model, test_ds))
        rows.append({"n_points": n, **{key: float(np.mean([m[key] for m in ms])) for key in ms[0]}})
    return rows


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Mean and percentile-bootstrap (1-alpha) CI of the mean of `values` (e.g. one AUC per seed)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
