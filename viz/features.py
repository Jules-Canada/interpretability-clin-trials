"""
viz/features.py — Feature activation visualizations

All functions return a matplotlib Figure so callers can save or display.
Designed for both developer diagnostics and non-technical readouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from torch import Tensor


def _load_feature_labels(labels_path: str | Path) -> dict[str, str]:
    """Load feature index → plain-English label from prompts/feature_labels.jsonl."""
    labels: dict[str, str] = {}
    p = Path(labels_path)
    if not p.exists():
        return labels
    with p.open() as f:
        for line in f:
            obj = json.loads(line)
            labels[str(obj["feature_index"])] = obj["label"]
    return labels


def plot_top_features(
    feature_acts: list[Tensor],
    layer: int,
    position: int,
    topk: int = 15,
    token_strings: Optional[list[str]] = None,
    feature_labels: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of the top-k active features at a given layer and token position.

    Args:
        feature_acts:   list of length L, each (batch, seq, n_features) — output of CLT.encode()
        layer:          which CLT layer to inspect
        position:       token position index
        topk:           number of features to show
        token_strings:  list of token strings for the prompt, used in the subtitle
        feature_labels: dict mapping str(feature_index) → plain-English label
        title:          optional override for the figure title

    Returns a Figure ready for plt.show() or fig.savefig().
    """
    # (n_features,) — activations at the requested layer and position
    acts: Tensor = feature_acts[layer][0, position].detach().cpu()
    values, indices = acts.topk(topk)

    feature_labels = feature_labels or {}
    labels = []
    for idx in indices.tolist():
        label = feature_labels.get(str(idx), f"Feature {idx}")
        labels.append(label)

    fig, ax = plt.subplots(figsize=(9, max(4, topk * 0.4)))
    bars = ax.barh(range(topk), values.tolist(), color="#4C72B0")
    ax.set_yticks(range(topk))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Activation strength")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    tok_label = f'"{token_strings[position]}"' if token_strings else f"position {position}"
    fig.suptitle(
        title or f"Top {topk} active features — Layer {layer}, token {tok_label}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_activation_heatmap(
    feature_acts: list[Tensor],
    token_strings: list[str],
    layer: int,
    topk_features: int = 30,
    feature_labels: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of token positions × top features at a given layer.

    Rows = tokens, columns = the top-k features by max activation across the sequence.
    Good for showing a non-technical audience *where* in the prompt each concept fires.

    Args:
        feature_acts:   list of length L, each (1, seq, n_features)
        token_strings:  list of decoded token strings, length seq
        layer:          which CLT layer to visualize
        topk_features:  how many features to show as columns
        feature_labels: dict mapping str(feature_index) → plain-English label
        title:          optional figure title override
    """
    # (seq, n_features)
    acts: Tensor = feature_acts[layer][0].detach().cpu()

    # Select top-k features by max activation across all positions
    max_per_feature, _ = acts.max(dim=0)
    _, top_feat_indices = max_per_feature.topk(topk_features)
    top_feat_indices = top_feat_indices.sort().values

    # (seq, topk_features)
    plot_data = acts[:, top_feat_indices]

    feature_labels = feature_labels or {}
    col_labels = [
        feature_labels.get(str(idx.item()), f"F{idx.item()}")
        for idx in top_feat_indices
    ]

    fig, ax = plt.subplots(figsize=(max(8, topk_features * 0.45), max(4, len(token_strings) * 0.35)))
    im = ax.imshow(plot_data.numpy(), aspect="auto", cmap="Blues")

    ax.set_yticks(range(len(token_strings)))
    ax.set_yticklabels(token_strings, fontsize=9)
    ax.set_xticks(range(topk_features))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Features (by max activation)")
    ax.set_ylabel("Token")

    fig.colorbar(im, ax=ax, label="Activation")
    fig.suptitle(
        title or f"Feature activations — Layer {layer}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_decoder_norms(
    clt: "CrossLayerTranscoder",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of W_dec[l' → l] Frobenius norms for every (source, target) layer pair.

    Confirms the cross-layer wiring is set up correctly: the lower triangle
    (l_target < l_source) should be empty; the diagonal and upper triangle should
    have non-zero values initialized from the model.

    Args:
        clt:   a CrossLayerTranscoder instance (CPU or any device)
        title: optional figure title override
    """
    from clt.model import CrossLayerTranscoder  # local import to avoid circular dep

    L = clt.cfg.n_layers
    # (L, L) grid — entry [l_source, l_target] is the Frobenius norm, or NaN if undefined
    norm_grid = [[float("nan")] * L for _ in range(L)]

    for l_source in range(L):
        for offset, decoder in enumerate(clt.decoders[l_source]):
            l_target = l_source + offset
            # Frobenius norm of (n_features, d_mlp) weight matrix
            norm_grid[l_source][l_target] = decoder.weight.detach().cpu().float().norm().item()

    import numpy as np
    data = np.array(norm_grid, dtype=float)

    fig, ax = plt.subplots(figsize=(max(4, L + 1), max(4, L + 1)))
    # Mask NaN cells (undefined l_target < l_source pairs)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="Blues", aspect="auto")

    ax.set_xticks(range(L))
    ax.set_xticklabels([f"L{l}" for l in range(L)])
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"L{l}" for l in range(L)])
    ax.set_xlabel("Target layer (l)")
    ax.set_ylabel("Source layer (l′)")

    # Annotate each cell with its value
    for l_src in range(L):
        for l_tgt in range(L):
            val = data[l_src, l_tgt]
            if not np.isnan(val):
                ax.text(l_tgt, l_src, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if val > data[~np.isnan(data)].max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, label="Frobenius norm")
    fig.suptitle(
        title or "Decoder weight norms ‖W_dec[l′→l]‖  (cross-layer structure)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_reconstruction_per_layer(
    mlp_reconstructions: "list[torch.Tensor]",
    mlp_targets: "list[torch.Tensor]",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of per-layer reconstruction MSE between CLT output and MLP targets.

    Args:
        mlp_reconstructions: list of length L, each (batch, seq, d_mlp) — from CLT.decode()
        mlp_targets:         list of length L, each (batch, seq, d_mlp) — ground-truth MLP outputs
        title:               optional figure title override
    """
    mse_per_layer = []
    for y_hat, y in zip(mlp_reconstructions, mlp_targets):
        mse = ((y_hat.detach().cpu() - y.detach().cpu()) ** 2).mean().item()
        mse_per_layer.append(mse)

    L = len(mse_per_layer)
    fig, ax = plt.subplots(figsize=(max(5, L * 1.2), 4))
    bars = ax.bar([f"Layer {l}" for l in range(L)], mse_per_layer, color="#4C72B0")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylabel("MSE")
    ax.set_ylim(0, max(mse_per_layer) * 1.25)
    fig.suptitle(
        title or "Per-layer reconstruction MSE (untrained CLT)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_training_dynamics(
    steps: list[int],
    loss_history: dict[str, list[float]],
    l0_history: dict[int, list[float]],
    title: str = "Toy CLT — training dynamics (50 steps)",
) -> plt.Figure:
    """
    Two-panel figure: loss components (top) and L0 sparsity per layer (bottom).

    Args:
        steps:         list of step indices
        loss_history:  dict with keys 'total', 'reconstruction', 'sparsity' → list of floats
        l0_history:    dict mapping layer index → list of L0 values (same length as steps)
        title:         figure suptitle
    """
    fig, (ax_loss, ax_l0) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # — Loss panel —
    ax_loss.plot(steps, loss_history["reconstruction"], label="Reconstruction", color="#4C72B0")
    ax_loss.plot(steps, loss_history["sparsity"],       label="Sparsity",       color="#C44E52", linestyle="--")
    ax_loss.plot(steps, loss_history["total"],          label="Total",          color="#333333", linewidth=1.5)
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_loss.set_title("Loss components", fontsize=10)

    # — L0 panel —
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for layer, values in sorted(l0_history.items()):
        ax_l0.plot(steps, values, label=f"Layer {layer}",
                   color=colors[layer % len(colors)])
    ax_l0.set_ylabel("Avg active features (L0)")
    ax_l0.set_xlabel("Step")
    ax_l0.legend(loc="upper right", fontsize=9)
    ax_l0.set_title("L0 sparsity per layer", fontsize=10)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_run_comparison(
    steps: list[int],
    run1_losses: dict[str, list[float]],
    run1_l0: dict[int, list[float]],
    run2_losses: dict[str, list[float]],
    run2_l0: dict[int, list[float]],
    run1_label: str = "λ = 2e-4 (original default)",
    run2_label: str = "λ = 1e-2 (new default)",
    n_features: int = 512,
    suptitle: str = "Training run comparison",
) -> plt.Figure:
    """
    2×2 grid: loss curves and L0 sparsity for two training runs side-by-side.

    Args:
        steps:       list of step indices (same for both runs)
        run1_losses: dict with keys 'total', 'reconstruction', 'sparsity'
        run1_l0:     dict mapping layer index → list of L0 values
        run2_losses: same for run 2
        run2_l0:     same for run 2
        run1_label:  title for left column
        run2_label:  title for right column
        n_features:  used to draw the L0 saturation line
        suptitle:    figure suptitle
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for col, (losses, l0, label) in enumerate([
        (run1_losses, run1_l0, run1_label),
        (run2_losses, run2_l0, run2_label),
    ]):
        ax = axes[0, col]
        ax.plot(steps, losses["reconstruction"], label="Reconstruction", color="#4C72B0")
        ax.plot(steps, losses["sparsity"],       label="Sparsity",       color="#C44E52", linestyle="--")
        ax.plot(steps, losses["total"],          label="Total",          color="#333333", linewidth=1.5)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        for i, (layer, values) in enumerate(sorted(l0.items())):
            ax.plot(steps, values, label=f"Layer {layer}", color=colors[i % len(colors)])
        ax.axhline(n_features, color="red", linestyle=":", linewidth=0.8,
                   label=f"Max ({n_features} features)")
        ax.set_ylabel("Avg active features (L0)")
        ax.set_xlabel("Step")
        ax.set_ylim(0, n_features * 1.1)
        ax.legend(fontsize=8)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_l0_over_training(
    steps: list[int],
    l0_by_layer: dict[int, list[float]],
    title: str = "L0 sparsity per layer over training",
) -> plt.Figure:
    """
    Line plot of average L0 (active features per token) per CLT layer over training steps.

    Args:
        steps:         list of step numbers
        l0_by_layer:   dict mapping layer index → list of L0 values (same length as steps)
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for layer, values in sorted(l0_by_layer.items()):
        ax.plot(steps, values, label=f"Layer {layer}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Avg active features per token (L0)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
