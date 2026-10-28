"""Build data/processed_clouds.npz (dense 1024-point clouds) from data/nodulemnist3d.npz.

    python prepare_data.py            # downloads nodulemnist3d.npz via medmnist if missing
"""
import sys, os; sys.path.insert(0, "src")
import numpy as np
from data import build_clouds, N_FULL

RAW, OUT = "data/nodulemnist3d.npz", "data/processed_clouds.npz"

if not os.path.exists(RAW):
    from medmnist import NoduleMNIST3D
    NoduleMNIST3D(split="train", download=True, root="data", size=28)

out = {}
for split in ["train", "val", "test"]:
    clouds, labels = build_clouds(RAW, split, n_full=N_FULL, seed=0)
    out[f"{split}_clouds"], out[f"{split}_labels"] = clouds, labels
    print(f"{split}: clouds {clouds.shape}, pos frac {labels.mean():.2f}")
np.savez_compressed(OUT, **out)
print("saved", OUT)
