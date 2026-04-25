"""
clt/model.py — Cross-Layer Transcoder (CLT)

Architecture follows:
  "Circuit Tracing: Revealing Computational Graphs in Language Models"
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html
  § Building an Interpretable Replacement Model
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from clt.config import CLTConfig


# ---------------------------------------------------------------------------
# JumpReLU
# ---------------------------------------------------------------------------

class JumpReLU(nn.Module):
    """
    JumpReLU(x) = x * (x > θ)   where θ is a learned per-feature threshold.

    Forward pass: standard thresholded activation (non-differentiable at θ).
    Gradients through θ use a sigmoid straight-through estimator:
      - Forward:  hard gate  gate_hard = (x > θ).float()
      - Backward: sigmoid gate  gate_soft = sigmoid((x - θ) / bandwidth)
    This approximates d/dθ [x * H(x - θ)] ≈ -x * δ(x - θ) via a smooth kernel.

    Shape: (*, n_features) → (*, n_features)
    """

    def __init__(self, n_features: int, init_threshold: float = 0.03, bandwidth: float = 0.1):
        super().__init__()
        # log-parameterize threshold to keep it positive
        self.log_threshold = nn.Parameter(
            torch.full((n_features,), math.log(init_threshold))
        )
        self.bandwidth = bandwidth

    @property
    def threshold(self) -> Tensor:
        return self.log_threshold.exp()

    def forward(
        self,
        x: Float[Tensor, "... n_features"],
    ) -> Float[Tensor, "... n_features"]:
        theta = self.threshold
        # Hard gate — used in the forward pass
        gate_hard = (x > theta).to(x.dtype)
        # Soft gate — used only for its gradient w.r.t. theta
        # gate_soft.grad flows to theta; its forward value is discarded
        gate_soft = torch.sigmoid((x - theta) / self.bandwidth)
        # STE trick: value = gate_hard, gradient = gate_soft gradient
        gate = gate_hard.detach() + gate_soft - gate_soft.detach()
        return x * gate


# ---------------------------------------------------------------------------
# Cross-Layer Transcoder
# ---------------------------------------------------------------------------

class CrossLayerTranscoder(nn.Module):
    """
    A CLT with L layers of features. Each feature at layer l':
      - reads from the residual stream x_{l'} via encoder W_enc[l']
      - contributes to MLP output reconstruction y_hat_l for all l >= l'
        via decoder W_dec[l' -> l]

    Reconstruction at layer l:
      y_hat_l = sum_{l'=0}^{l} W_dec[l'][l] @ a_{l'}

    where a_{l'} = JumpReLU(W_enc[l'] @ x_{l'})

    Training loss:
      L = sum_l ||y_hat_l - y_l||^2   (MSE, reconstruction)
        + λ * sum_l sum_i (a_{l,i} / (|a_{l,i}| + c))  (sparsity)
    """

    @staticmethod
    def _init_decoder(in_features: int, out_features: int, n_layers: int, d_model: int) -> nn.Linear:
        """
        Linear layer initialized to U(-bound, bound) where
          bound = 1 / sqrt(n_layers * d_model)
        following the paper (§ Building an Interpretable Replacement Model).
        """
        bound = 1.0 / math.sqrt(n_layers * d_model)
        layer = nn.Linear(in_features, out_features, bias=False)
        nn.init.uniform_(layer.weight, -bound, bound)
        return layer

    def __init__(self, cfg: CLTConfig):
        super().__init__()
        self.cfg = cfg
        L = cfg.n_layers
        F = cfg.n_features

        # Encoders: one per layer
        # W_enc[l]: (d_model,) → (n_features,)
        # Initialized to U(-1/sqrt(n_features), 1/sqrt(n_features)) per paper.
        enc_bound = 1.0 / math.sqrt(F)
        self.encoders = nn.ModuleList([
            self._make_encoder(cfg.d_model, F, enc_bound)
            for _ in range(L)
        ])

        # JumpReLU activations: one per layer (separate learned thresholds)
        self.jump_relus = nn.ModuleList([
            JumpReLU(F, init_threshold=cfg.jumprelu_threshold, bandwidth=cfg.jumprelu_bandwidth)
            for _ in range(L)
        ])

        # Decoders: W_dec[l'][l] maps features at l' to MLP output at l
        # Only defined for l >= l', so we use a nested ModuleList
        # self.decoders[l_source][l_target - l_source]: Linear(n_features -> d_mlp)
        # Initialized to U(-1/sqrt(n_layers*d_model), 1/sqrt(n_layers*d_model)) per paper.
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                self._init_decoder(F, cfg.d_mlp, L, cfg.d_model)
                for _ in range(L - l_source)   # targets: l_source, l_source+1, ..., L-1
            ])
            for l_source in range(L)
        ])

    @staticmethod
    def _make_encoder(d_model: int, n_features: int, bound: float) -> nn.Linear:
        """Linear encoder with bias, initialized to U(-bound, bound)."""
        layer = nn.Linear(d_model, n_features, bias=True)
        nn.init.uniform_(layer.weight, -bound, bound)
        nn.init.zeros_(layer.bias)
        return layer

    def encode(
        self,
        resid_streams: list[Float[Tensor, "batch seq d_model"]],
    ) -> list[Float[Tensor, "batch seq n_features"]]:
        """
        Encode residual streams at each layer into sparse feature activations.

        Args:
            resid_streams: list of length L, resid_streams[l] is x_l

        Returns:
            feature_acts: list of length L, feature_acts[l] is a_l
        """
        feature_acts = []
        for l, (encoder, jump_relu) in enumerate(zip(self.encoders, self.jump_relus)):
            # (batch, seq, d_model) → (batch, seq, n_features)
            pre_act = encoder(resid_streams[l])
            a_l = jump_relu(pre_act)
            feature_acts.append(a_l)
        return feature_acts

    def decode(
        self,
        feature_acts: list[Float[Tensor, "batch seq n_features"]],
    ) -> list[Float[Tensor, "batch seq d_mlp"]]:
        """
        Reconstruct MLP outputs at each layer from feature activations.

        Args:
            feature_acts: list of length L, output of encode()

        Returns:
            mlp_reconstructions: list of length L, y_hat_l for each layer
        """
        L = self.cfg.n_layers
        ref = feature_acts[0]
        # Initialize reconstructions to zero: (batch, seq, d_mlp)
        reconstructions = [
            torch.zeros(*ref.shape[:-1], self.cfg.d_mlp, device=ref.device, dtype=ref.dtype)
            for _ in range(L)
        ]

        for l_source in range(L):
            a = feature_acts[l_source]  # (batch, seq, n_features)
            for offset, decoder in enumerate(self.decoders[l_source]):
                l_target = l_source + offset
                # (batch, seq, n_features) → (batch, seq, d_mlp)
                reconstructions[l_target] = reconstructions[l_target] + decoder(a)

        return reconstructions

    def forward(
        self,
        resid_streams: list[Float[Tensor, "batch seq d_model"]],
    ) -> tuple[
        list[Float[Tensor, "batch seq n_features"]],
        list[Float[Tensor, "batch seq d_mlp"]],
    ]:
        """
        Full forward pass: encode then decode.

        Returns:
            feature_acts: sparse activations per layer
            mlp_reconstructions: reconstructed MLP outputs per layer
        """
        feature_acts = self.encode(resid_streams)
        mlp_reconstructions = self.decode(feature_acts)
        return feature_acts, mlp_reconstructions

    # -----------------------------------------------------------------------
    # Loss functions
    # -----------------------------------------------------------------------

    def reconstruction_loss(
        self,
        mlp_reconstructions: list[Float[Tensor, "batch seq d_mlp"]],
        mlp_targets: list[Float[Tensor, "batch seq d_mlp"]],
    ) -> Float[Tensor, ""]:
        """
        MSE reconstruction loss summed across layers.

        L_MSE = sum_l ||y_hat_l - y_l||^2
        """
        loss = torch.tensor(0.0, device=mlp_reconstructions[0].device)
        for y_hat, y in zip(mlp_reconstructions, mlp_targets):
            loss = loss + ((y_hat - y) ** 2).mean()
        return loss

    def sparsity_loss(
        self,
        feature_acts: list[Float[Tensor, "batch seq n_features"]],
    ) -> Float[Tensor, ""]:
        """
        Sparsity penalty following the paper exactly (§ Building an Interpretable
        Replacement Model):

            L_sparsity = λ Σ_ℓ Σ_i tanh(c · ‖W_dec_i^ℓ‖ · a_i^ℓ)

        where ‖W_dec_i^ℓ‖ is the L2 norm of the concatenation of all decoder
        vectors for feature i at source layer ℓ (one vector per target layer).

        Weighting by decoder norm means features with larger downstream influence
        are penalised more heavily, not just features that happen to be active.
        With zero-init decoders the norm starts at zero so there is no sparsity
        pressure at step 0; it grows naturally as decoders are learned.
        """
        cfg = self.cfg
        loss = torch.tensor(0.0, device=feature_acts[0].device)

        for l_source in range(cfg.n_layers):
            # Concatenate decoder weight matrices for all target layers into one
            # tall matrix: (n_targets * d_mlp, n_features).
            # Column i of this matrix is the concatenated decoder vector for
            # feature i at source layer l_source.
            # (n_targets * d_mlp, n_features)
            all_dec_weights = torch.cat(
                [dec.weight for dec in self.decoders[l_source]], dim=0
            )
            # (n_features,) — L2 norm of each feature's concatenated decoder
            dec_norm = all_dec_weights.norm(dim=0)

            # a: (batch, seq, n_features) — broadcasts with (n_features,)
            a = feature_acts[l_source]
            scaled = cfg.sparsity_c * dec_norm * a
            # Sum over features (Σ_i), mean over batch/seq (minibatch normalization)
            # This keeps the sparsity coefficient scale-invariant w.r.t. batch size,
            # consistent with reconstruction loss which also uses .mean() over batch/seq.
            loss = loss + torch.tanh(scaled).sum(dim=-1).mean()

        return cfg.sparsity_coeff * loss

    def loss(
        self,
        resid_streams: list[Float[Tensor, "batch seq d_model"]],
        mlp_targets: list[Float[Tensor, "batch seq d_mlp"]],
    ) -> dict[str, Float[Tensor, ""]]:
        """
        Compute total loss and return a dict of components for logging.

        Usage:
            losses = clt.loss(resid_streams, mlp_targets)
            losses['total'].backward()

        Returns dict with keys: 'total', 'reconstruction', 'sparsity'
        """
        feature_acts, mlp_reconstructions = self.forward(resid_streams)
        rec_loss = self.reconstruction_loss(mlp_reconstructions, mlp_targets)
        spar_loss = self.sparsity_loss(feature_acts)
        return {
            "total": rec_loss + spar_loss,
            "reconstruction": rec_loss,
            "sparsity": spar_loss,
        }

    # -----------------------------------------------------------------------
    # Inspection helpers
    # -----------------------------------------------------------------------

    def active_features(
        self,
        feature_acts: list[Float[Tensor, "batch seq n_features"]],
        threshold: float = 1e-6,
    ) -> list[dict]:
        """
        Return a list of dicts describing non-zero features, for inspection.

        Each dict: {'layer': l, 'batch': b, 'pos': p, 'feature': f, 'activation': v}
        Useful for exploring which features fire on a given prompt.
        """
        results = []
        for l, a in enumerate(feature_acts):
            nonzero = (a.abs() > threshold).nonzero(as_tuple=False)
            for idx in nonzero:
                b, p, f = idx.tolist()
                results.append({
                    "layer": l,
                    "batch": b,
                    "pos": p,
                    "feature": f,
                    "activation": a[b, p, f].item(),
                })
        return results

    def l0_per_layer(
        self,
        feature_acts: list[Float[Tensor, "batch seq n_features"]],
    ) -> list[float]:
        """Average L0 sparsity (number of active features) per layer."""
        return [
            (a.abs() > 1e-6).float().sum(dim=-1).mean().item()
            for a in feature_acts
        ]


# ---------------------------------------------------------------------------
# Quick smoke test (run with: python -m clt.model)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = CLTConfig(
        n_layers=2,
        d_model=64,
        d_mlp=256,
        n_features=128,
    )
    clt = CrossLayerTranscoder(cfg)

    batch, seq = 2, 16
    resid_streams = [torch.randn(batch, seq, cfg.d_model) for _ in range(cfg.n_layers)]
    mlp_targets   = [torch.randn(batch, seq, cfg.d_mlp)   for _ in range(cfg.n_layers)]

    losses = clt.loss(resid_streams, mlp_targets)
    print("Losses:", {k: f"{v.item():.4f}" for k, v in losses.items()})

    feature_acts, _ = clt(resid_streams)
    print("L0 per layer:", clt.l0_per_layer(feature_acts))
    print("Smoke test passed ✓")
