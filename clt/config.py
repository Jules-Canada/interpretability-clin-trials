"""
clt/config.py — CLTConfig dataclass

All CLT hyperparameters live here. No magic numbers elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CLTConfig:
    n_layers: int          # Number of layers in the underlying model (L)
    d_model: int           # Residual stream dimension
    d_mlp: int             # MLP output dimension (often 4 * d_model)
    n_features: int        # CLT features per layer
    jumprelu_threshold: float = 0.03    # Initial JumpReLU threshold θ (learned)
    jumprelu_bandwidth: float = 0.1    # STE bandwidth; must be ~same order as pre-activation scale
    sparsity_coeff: float = 2e-4       # λ: weight on sparsity loss
    sparsity_c: float = 1.0            # c: hyperparameter in sparsity penalty
