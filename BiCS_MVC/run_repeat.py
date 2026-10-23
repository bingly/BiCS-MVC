#!/usr/bin/env python
"""Run BiCS-MVC with repeated K-Means evaluations (different random seeds).

This script runs the model training once, then evaluates with K-Means
using 10 different random seeds to get mean ± std statistics.

Usage:
    python run_repeat.py                # run on all datasets
    python run_repeat.py 3Sources       # run on a specific dataset
"""
import sys
import os
os.environ['OMP_NUM_THREADS']="2"
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from config import DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, METHOD, TARGETS, RESULTS_DIR
from dataset import MultiViewDataset
from model import BiCSMVC
from metrics import all_metrics


def extract_features(model, dataset, batch_size=256):
    """Extract concatenated normalized features Z = [z_1 || ... || z_V]."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_z = []
    with torch.no_grad():
        for xs, _, _ in loader:
            xs = [x.to(DEVICE) for x in xs]
            outputs = model(xs)
            all_z.append(torch.cat(outputs['zs'], dim=1).cpu().numpy())
    return np.concatenate(all_z, axis=0)


def evaluate_with_seed(z_all, labels, nc, seed):
    """K-Means clustering with specific random seed."""
    km = KMeans(n_clusters=nc, n_init=10, random_state=seed).fit(z_all)
    metrics = {k: float(v) for k, v in all_metrics(labels, km.labels_).items()}
    return metrics


def run_repeat(dataset, n_kmeans=10, seed_start=0):
    """Train BiCS-MVC once, then evaluate with K-Means using different seeds.

    Args:
        dataset: MultiViewDataset object
        n_kmeans: Number of K-Means runs with different seeds
        seed_start: Starting random seed for K-Means

    Returns:
        dict with individual results and mean ± std
    """
    nv, dims = dataset.nv, dataset.dims
    bs = min(BATCH_SIZE, dataset.ns)

    # Use a fixed seed for training
    train_seed = 42
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    print(f"  Training with seed={train_seed} ...")
    t0 = time.time()

    # ---- Training ----
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    model = BiCSMVC(dims).to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.01,
    )

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

    # Extract features
    z_all = extract_features(model, dataset)
    nc = dataset.nc
    labels = dataset.labels
    train_time = time.time() - t0

    print(f"  Running K-Means with {n_kmeans} different seeds ...")

    # ---- K-Means with different seeds ----
    results = []
    for i in range(n_kmeans):
        km_seed = seed_start + i
        m = evaluate_with_seed(z_all, labels, nc, km_seed)
        m['seed'] = km_seed
        results.append(m)

    # Calculate mean and std
    metrics_names = ['ACC', 'NMI', 'ARI', 'F-score']
    mean_std = {}
    for metric in metrics_names:
        values = [r[metric] for r in results]
        mean_std[f'{metric}_mean'] = np.mean(values)
        mean_std[f'{metric}_std'] = np.std(values)

    return {
        'individual': results,
        'mean_std': mean_std,
        'train_time': train_time,
    }


def save_results(all_results, method, result_dir):
    """Save results to JSON and TXT files."""
    os.makedirs(result_dir, exist_ok=True)

    # Save JSON
    json_file = os.path.join(result_dir, f'{method}_repeat_results.json')
    with open(json_file, 'a+') as f:
        json.dump(all_results, f, indent=2)

    # Save TXT with mean ± std
    txt_file = os.path.join(result_dir, f'{method}_repeat_results.txt')

    with open(txt_file, 'a+', encoding='utf-8') as f:
        # Header
        f.write(f"{'Dataset':<12} {'Method':<12} {'ACC_mean':>10} {'ACC_std':>10} "
                f"{'NMI_mean':>10} {'NMI_std':>10} {'ARI_mean':>10} {'ARI_std':>10} "
                f"{'F-score_mean':>10} {'F-score_std':>10} {'Time':>10}\n")
        f.write("-" * 130 + "\n")

        for r in all_results:
            ds = r['dataset']
            ms = r['mean_std']
            f.write(
                f"{ds:<12} {method:<12} "
                f"{ms['ACC_mean']*100:>9.2f}±{ms['ACC_std']*100:>6.2f} "
                f"{ms['NMI_mean']*100:>9.2f}±{ms['NMI_std']*100:>6.2f} "
                f"{ms['ARI_mean']*100:>9.2f}±{ms['ARI_std']*100:>6.2f} "
                f"{ms['F-score_mean']*100:>9.2f}±{ms['F-score_std']*100:>6.2f} "
                f"{r['time']:>9.2f}\n"
            )

    # Also save detailed results with each seed
    detail_file = os.path.join(result_dir, f'{method}_repeat_details.txt')
    with open(detail_file, 'a+', encoding='utf-8') as f:
        for r in all_results:
            ds = r['dataset']
            f.write(f"\n{'='*60}\nDataset: {ds}\n{'='*60}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Training time: {r['time']:.2f}s\n\n")

            # Mean ± std summary
            ms = r['mean_std']
            f.write("Mean ± Std:\n")
            f.write(f"  ACC:     {ms['ACC_mean']*100:.2f} ± {ms['ACC_std']*100:.2f}\n")
            f.write(f"  NMI:     {ms['NMI_mean']*100:.2f} ± {ms['NMI_std']*100:.2f}\n")
            f.write(f"  ARI:     {ms['ARI_mean']*100:.2f} ± {ms['ARI_std']*100:.2f}\n")
            f.write(f"  F-score: {ms['F-score_mean']*100:.2f} ± {ms['F-score_std']*100:.2f}\n\n")

            # Individual results
            f.write("Individual results (each K-Means seed):\n")
            f.write(f"{'Seed':>6} {'ACC':>8} {'NMI':>8} {'ARI':>8} {'F-score':>8}\n")
            f.write("-" * 44 + "\n")
            for ind in r['individual']:
                f.write(
                    f"{ind['seed']:>6} "
                    f"{ind['ACC']*100:>7.2f} "
                    f"{ind['NMI']*100:>7.2f} "
                    f"{ind['ARI']*100:>7.2f} "
                    f"{ind['F-score']*100:>7.2f}\n"
                )


def main():
    if len(sys.argv) > 1:
        targets = [dn for dn in sys.argv[1:] if dn in TARGETS]
        if not targets:
            print(f'Unknown dataset(s): {sys.argv[1:]}. Available: {TARGETS}')
            return
    else:
        targets = TARGETS

    print(f"Running {METHOD} with repeated K-Means evaluation")
    print(f"Datasets: {targets}")
    print(f"=" * 60)

    all_results = []

    for dn in targets:
        print(f"\n===== {METHOD} on {dn} =====")
        try:
            ds = MultiViewDataset(dn)
            print(f"  Views:{ds.nv}  Samples:{ds.ns}  Classes:{ds.nc}  Dims:{ds.dims}")

            result = run_repeat(ds, n_kmeans=10, seed_start=0)
            t_total = result['train_time']

            print(f"  ACC:     {result['mean_std']['ACC_mean']*100:.2f} ± {result['mean_std']['ACC_std']*100:.2f}")
            print(f"  NMI:     {result['mean_std']['NMI_mean']*100:.2f} ± {result['mean_std']['NMI_std']*100:.2f}")
            print(f"  ARI:     {result['mean_std']['ARI_mean']*100:.2f} ± {result['mean_std']['ARI_std']*100:.2f}")
            print(f"  F-score: {result['mean_std']['F-score_mean']*100:.2f} ± {result['mean_std']['F-score_std']*100:.2f}")
            print(f"  Time:    {t_total:.2f}s")

            all_results.append({
                'dataset': dn,
                'mean_std': result['mean_std'],
                'individual': result['individual'],
                'time': t_total,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    save_results(all_results, METHOD, RESULTS_DIR)
    print(f"\n{'='*60}")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
