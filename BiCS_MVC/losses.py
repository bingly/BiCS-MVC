"""Loss functions for BiCS-MVC."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalContrastiveLoss(nn.Module):
    """Bidirectional contrastive loss across all view pairs.

    L_bi-con = 1/|P| * sum( 1/2 * (L_{v->u} + L_{u->v}) )

    For each unordered view pair, computes symmetric InfoNCE:
    loss = (cross_entropy(S, I) + cross_entropy(S^T, I)) / 2
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_list):
        if len(features_list) < 2:
            return torch.tensor(0.0, device=features_list[0].device)

        loss = 0.0
        valid_pairs = 0
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                f1 = features_list[i]
                f2 = features_list[j]
                sim = torch.matmul(f1, f2.T) / max(self.temperature, 0.1)
                lbl = torch.arange(f1.shape[0], device=f1.device)
                loss += (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.T, lbl)) / 2
                valid_pairs += 1

        return loss / max(valid_pairs, 1)


class SemanticConsistencyLoss(nn.Module):
    """Semantic consistency regularization across all view pairs.

    L_sem = 1/|P| * sum( mean((1 - cos_sim(z_i, z_j))^2) )

    Penalizes angular deviation between same-instance cross-view features.
    The quadratic form tolerates small view-specific variation while
    strongly suppressing large semantic drift.
    """

    def __init__(self):
        super().__init__()

    def forward(self, zs_list):
        if len(zs_list) < 2:
            return torch.tensor(0.0, device=zs_list[0].device)

        loss = 0.0
        valid_pairs = 0
        for i in range(len(zs_list)):
            for j in range(i + 1, len(zs_list)):
                zi = F.normalize(zs_list[i], dim=1)
                zj = F.normalize(zs_list[j], dim=1)
                cos_sim = torch.sum(zi * zj, dim=1)
                loss += torch.mean((1 - cos_sim) ** 2)
                valid_pairs += 1

        return loss / max(valid_pairs, 1)
