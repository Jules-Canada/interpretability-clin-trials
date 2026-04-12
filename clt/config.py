"""
clt/config.py — CLTConfig and TrainConfig dataclasses

All CLT hyperparameters live here. No magic numbers elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CLTConfig:
    n_layers: int          # Number of layers in the underlying model (L)
    d_model: int           # Residual stream dimension
    d_mlp: int             # MLP output dimension (often 4 * d_model)
    n_features: int        # CLT features per layer
    jumprelu_threshold: float = 0.03    # Initial JumpReLU threshold θ (learned)
    jumprelu_bandwidth: float = 1.0    # STE bandwidth; paper uses 1.0 (§ Building an Interpretable Replacement Model)
    sparsity_coeff: float = 1e-2       # λ: weight on sparsity loss (2e-4 too weak — see training diagnostics)
    sparsity_c: float = 1.0            # c: hyperparameter in sparsity penalty
    normalize_activations: bool = True  # Divide residual/MLP by per-layer RMS before CLT (paper §3.2)


@dataclass
class AttributionConfig:
    target_position: int = -1        # Sequence position to trace (default: last token)
    min_activation: float = 1e-4     # Minimum feature activation to include as a node
    top_k_nodes: int = 20            # K for node pruning (keep top-K by indirect influence)
    top_k_edges: int = 50            # K for edge pruning
    max_path_length: int = 3         # ℓ in B_ℓ = Σ A^i — limits indirect influence depth


@dataclass
class TrainConfig:
    n_steps: int = 50_000          # Total number of gradient steps
    lr: float = 2e-4               # Adam learning rate
    batch_size: int = 512          # Token positions sampled per step
    log_every: int = 50            # Log metrics every N steps
    save_every: int = 5_000        # Save checkpoint every N steps
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "ignis-clt"
    wandb_group: str = ""          # Group runs by model size, e.g. "pythia-410m"
    use_wandb: bool = True         # Set False to disable W&B (dev runs)
