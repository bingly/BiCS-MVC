"""Clustering evaluation metrics."""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true, y_pred):
    """ACC with Hungarian optimal label matching."""
    y_true = np.array(y_true).flatten().astype(np.int64)
    y_pred = np.array(y_pred).flatten().astype(np.int64)
    nc = max(y_true.max(), y_pred.max()) + 1
    D = confusion_matrix(y_pred, y_true)[:nc, :nc]
    ri, ci = linear_sum_assignment(-D)
    mapped = np.zeros_like(y_pred)
    for o, n in zip(ri, ci):
        mapped[y_pred == o] = n
    return (mapped == y_true).mean()


def pairwise_fscore(y_true, y_pred):
    """F-score computed from pairwise precision/recall."""
    yt = np.array(y_true).flatten().astype(np.int64)
    yp = np.array(y_pred).flatten().astype(np.int64)
    n = len(yt)
    ts = yt[:, None] == yt[None, :]
    ps = yp[:, None] == yp[None, :]
    m = np.triu(np.ones((n, n), dtype=bool), k=1)
    TP = np.sum(ts[m] & ps[m])
    FP = np.sum(~ts[m] & ps[m])
    FN = np.sum(ts[m] & ~ps[m])
    P = TP / (TP + FP) if TP + FP > 0 else 0.0
    R = TP / (TP + FN) if TP + FN > 0 else 0.0
    return 2 * P * R / (P + R) if P + R > 0 else 0.0


def all_metrics(y_true, y_pred):
    """Return ACC, NMI, ARI, F-score as a dict."""
    return {
        'ACC': clustering_accuracy(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
        'ARI': adjusted_rand_score(y_true, y_pred),
        'F-score': pairwise_fscore(y_true, y_pred),
    }
