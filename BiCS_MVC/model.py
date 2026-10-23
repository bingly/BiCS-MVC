"""BiCS-MVC model."""

import torch.nn as nn
import torch.nn.functional as F
from losses import BidirectionalContrastiveLoss, SemanticConsistencyLoss


class BiCSMVC(nn.Module):
    """Bidirectional Contrastive Learning with Semantic Consistency for MVC.

    Architecture:
      encoder:  Linear(d_v, 512) -> BatchNorm -> ReLU
      projector: Linear(512, 128) -> BatchNorm -> ReLU -> L2-normalize

    Loss:
      L_total = 1.0 * L_bi-con + 0.3 * L_sem
    """

    def __init__(self, view_dims, feature_dim=512, high_dim=128, temperature=0.5):
        super().__init__()
        self.num_views = len(view_dims)

        # View-specific encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(),
            )
            for d in view_dims
        ])

        # View-specific projectors
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, high_dim),
                nn.BatchNorm1d(high_dim),
                nn.ReLU(),
            )
            for _ in range(self.num_views)
        ])

        # Losses
        self.contrastive_loss = BidirectionalContrastiveLoss(temperature=temperature)
        self.semantic_loss = SemanticConsistencyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xs):
        """Encode all views and return L2-normalized embeddings."""
        zs = []
        for v in range(self.num_views):
            h = self.encoders[v](xs[v])
            z = F.normalize(self.projectors[v](h), dim=1)
            zs.append(z)
        return {'zs': zs}

    def compute_loss(self, outputs):
        """Compute total loss: L_total = 1.0 * L_bi-con + 0.3 * L_sem."""
        zs = outputs['zs']
        loss_con = self.contrastive_loss(zs)
        loss_sem = self.semantic_loss(zs)
        total = 1.0 * loss_con + 0.3 * loss_sem
        return total, {'contrastive': loss_con, 'semantic': loss_sem}
