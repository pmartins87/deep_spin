from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AdvantageNet(nn.Module):
    """
    Entrada: obs (B, 292)
    Saída: advantage (B, 7)

    Importante:
    - aplicamos máscara para ações ilegais -> colocamos -inf no logit/adv na hora de fazer regret-matching
    - para loss, treinamos só nas ações legais (para não “ensinar” coisa em ação impossível)
    """
    def __init__(self, obs_dim: int | None = None, num_actions: int = 7, hidden: list[int] | None = None):
        super().__init__()
        if obs_dim is None:
            # Fallback so code still runs if the caller forgets to pass obs_dim.
            # Prefer passing explicit obs_dim from the trainer.
            obs_dim = int(os.environ.get("SPIN_OBS_DIM", "292"))
        if hidden is None:
            hidden = [1024, 1024, 512, 512]
        self.mlp = MLP(obs_dim, hidden, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)

    @staticmethod
    def masked_mse(pred: torch.Tensor, target: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        # legal: (B,7) em {0,1}
        diff2 = (pred - target) ** 2
        diff2 = diff2 * legal
        denom = legal.sum().clamp_min(1.0)
        return diff2.sum() / denom


class PolicyNet(nn.Module):
    """
    Entrada: obs (B,obs_dim)
    Saída: logits (B,7) -> policy via softmax mascarado
    """
    def __init__(self, obs_dim: int | None = None, num_actions: int = 7, hidden: list[int] | None = None):
        super().__init__()
        if obs_dim is None:
            obs_dim = int(os.environ.get("SPIN_OBS_DIM", "292"))
        if hidden is None:
            hidden = [1024, 512, 512]
        self.mlp = MLP(obs_dim, hidden, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)

    @staticmethod
    def masked_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        # coloca -inf onde ilegal
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(legal > 0.0, logits, torch.full_like(logits, neg_inf))
        return F.softmax(masked, dim=-1)

    @staticmethod
    def masked_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        # target_probs já deve ser distribuição (B,7) com zeros em ilegais
        probs = PolicyNet.masked_softmax(logits, legal)
        # evita log(0)
        probs = probs.clamp_min(1e-12)
        loss = -(target_probs * probs.log())
        denom = legal.sum(dim=-1).clamp_min(1.0)
        return (loss.sum(dim=-1) / denom).mean()
