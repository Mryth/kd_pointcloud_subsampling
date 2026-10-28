"""Knowledge distillation for robustness to sub-sampling.

One training loop covers every variant of the study:

  mode           teacher  student input        loss
  -------------  -------  -------------------  -----------------------------------------
  scratch        -        dense                CE                     (same selection rule as the others)
  aug            -        random sub-sample    CE                     (augmentation-only baseline)
  kd             frozen   dense                a*CE + (1-a)*T^2*KL    (Hinton logit KD)
  kd_robust      frozen   random sub-sample    a*CE + (1-a)*T^2*KL    teacher always sees DENSE cloud
  + feat_w > 0   frozen   (as above)           + feat_w * ||P(g_s) - g_t||^2 on L2-normalised features

`aug` is the control for `kd_robust`: both see the same sub-sampled inputs, so any gap
between them is attributable to the teacher's soft targets, not to the augmentation.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import DEVICE, make_loader, evaluate


def subsample_batch(x, n_min=16, n_max=None, gen=None):
    """Keep a random subset of points, with n ~ log-uniform in [n_min, n_max] for the whole
    batch (one n per batch so the tensor stays rectangular; different points per sample)."""
    B, N, _ = x.shape
    n_max = n_max or N
    u = torch.rand(1, generator=gen).item()
    n = int(round(math.exp(math.log(n_min) + u * (math.log(n_max) - math.log(n_min)))))
    idx = torch.rand(B, N, generator=gen).argsort(1)[:, :n].to(x.device)
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[2]))


def kd_loss(z_s, z_t, T):
    """Hinton KD term: T^2 * KL(softmax(z_t/T) || softmax(z_s/T))."""
    return F.kl_div(F.log_softmax(z_s / T, 1), F.softmax(z_t / T, 1), reduction="batchmean") * T * T


def val_score(model, val_ds, val_ns):
    """Mean val AUC over the densities in `val_ns` (model selection criterion)."""
    aucs = []
    for n in val_ns:
        val_ds.set_n_points(n); val_ds.rng = np.random.default_rng(0)
        aucs.append(evaluate(model, val_ds)["auc"])
    val_ds.set_n_points(max(val_ns))
    return float(np.mean(aucs))


def train_kd(student, train_ds, val_ds, teacher=None, mode="kd_robust", epochs=40, lr=1e-3,
             alpha=0.5, T=4.0, feat_w=0.0, n_min=16, class_weight=None, val_ns=(1024,),
             seed=0, log=True):
    assert mode in ("scratch", "aug", "kd", "kd_robust")
    use_teacher = mode in ("kd", "kd_robust")
    assert not use_teacher or teacher is not None, f"mode {mode} needs a teacher"
    subsample = mode in ("aug", "kd_robust")

    student.to(DEVICE)
    proj = None
    params = list(student.parameters())
    if use_teacher:
        teacher.to(DEVICE).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if feat_w > 0:                       # student emb -> teacher emb
            proj = nn.Linear(student.emb, teacher.emb).to(DEVICE)
            params += list(proj.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=DEVICE)
    gen = torch.Generator().manual_seed(seed)
    tl = make_loader(train_ds, shuffle=True)
    best, best_state = -1, None
    for ep in range(epochs):
        student.train()
        if proj is not None:
            proj.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            xs = subsample_batch(x, n_min=n_min, gen=gen) if subsample else x
            z_s, g_s = student(xs)
            ce = F.cross_entropy(z_s, y, weight=cw)
            if use_teacher:
                with torch.no_grad():
                    z_t, g_t = teacher(x)    # teacher always sees the dense cloud
                loss = alpha * ce + (1 - alpha) * kd_loss(z_s, z_t, T)
                if proj is not None:
                    loss = loss + feat_w * F.mse_loss(F.normalize(proj(g_s), dim=1),
                                                      F.normalize(g_t, dim=1), reduction="sum") / x.shape[0]
            else:
                loss = ce
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        s = val_score(student, val_ds, val_ns)
        if s > best:
            best = s; best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        if log and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:2d} | loss {loss.item():.3f} | val auc{list(val_ns)} {s:.3f}")
    if best_state:
        student.load_state_dict(best_state)
    return student
