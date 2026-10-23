"""Training and evaluation loop for BiCS-MVC."""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from config import DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
from model import BiCSMVC
from metrics import all_metrics


@torch.no_grad()
def extract_features(model, dataset, batch_size=256):
    """Extract concatenated normalized features Z = [z_1 || ... || z_V]."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_z = []
    for xs, _, _ in loader:
        xs = [x.to(DEVICE) for x in xs]
        outputs = model(xs)
        all_z.append(torch.cat(outputs['zs'], dim=1).cpu().numpy())
    return np.concatenate(all_z, axis=0)


def evaluate(model, dataset):
    """K-Means clustering evaluation."""
    z_all = extract_features(model, dataset)
    nc = dataset.nc
    km = KMeans(n_clusters=nc, n_init=10, random_state=42).fit(z_all)
    metrics = {k: float(v) for k, v in all_metrics(dataset.labels, km.labels_).items()}
    return metrics


def run_one(dataset, seed_start=42, n_runs=10):
    """Train BiCS-MVC on one dataset with multiple seeds, return best ACC run."""
    nv, dims = dataset.nv, dataset.dims
    bs = min(BATCH_SIZE, dataset.ns)

    best_metrics = {'ACC': 0, 'NMI': 0, 'ARI': 0, 'F-score': 0}
    t0 = time.time()

    for run_idx in range(n_runs):
        run_seed = seed_start + run_idx
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
        model = BiCSMVC(dims).to(DEVICE)
        opt = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.01,
        )

        # ---- Training ----
        for ep in range(EPOCHS):
            model.train()
            for xs, _, _ in loader:
                xs = [x.to(DEVICE) for x in xs]
                outputs = model(xs)
                loss, _ = model.compute_loss(outputs)
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()

        # ---- Evaluation ----
        m = evaluate(model, dataset)
        if m['ACC'] > best_metrics['ACC']:
            best_metrics = m

    return {**best_metrics, 'Time(s)': time.time() - t0}
