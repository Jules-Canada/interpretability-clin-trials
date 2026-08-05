"""scripts/_tokenizer_resolve.py — single source of truth for picking the
tokenizer that decodes HDF5 token ids back to context strings.

Kept dependency-free (h5py + stdlib only) so the lightweight recovery tool
``fix_feature_activations_tokenizer.py`` can import it without pulling in
torch / transformer_lens / clt the way ``find_top_activations.py`` does.

Background: a tokenizer that does not match the model that produced an HDF5
decodes its token ids into silent garbage (no error). This bit the Phase 4
MedGemma run when find_top_activations.py fell back to a Pythia default.
extract_activations.py now stamps ``attrs["model_name"]`` into the HDF5 so
the tokenizer provably follows the data.
"""

from __future__ import annotations

import sys

import h5py


def resolve_model_name(h5_path: str, cli_model_name: str | None) -> str:
    """Decide which tokenizer to decode context strings with.

    The HDF5 is self-describing: extract_activations.py stamps
    ``attrs["model_name"]``. That attr is the source of truth. Rules:

      - attr present, no CLI flag        -> use attr (normal path)
      - attr present, CLI flag agrees    -> use attr
      - attr present, CLI flag CONFLICTS -> hard error (the wrong-value
                                            case — exactly what we want
                                            to catch loudly)
      - attr absent (legacy HDF5), flag  -> use flag
      - attr absent, no flag             -> hard error: legacy file,
                                            caller must say which model
    """
    with h5py.File(h5_path, "r") as h5:
        attr_model = h5.attrs.get("model_name")
    if attr_model is not None:
        attr_model = str(attr_model)

    if attr_model is not None:
        if cli_model_name is not None and cli_model_name != attr_model:
            sys.exit(
                f"ERROR: --model_name '{cli_model_name}' conflicts with the "
                f"model recorded in {h5_path} ('{attr_model}'). The HDF5 was "
                f"produced by '{attr_model}'; decoding its token ids with a "
                f"different tokenizer yields silent garbage. Drop --model_name "
                f"(the recorded value is used automatically) or pass the "
                f"matching one."
            )
        return attr_model

    if cli_model_name is None:
        sys.exit(
            f"ERROR: {h5_path} predates the self-describing 'model_name' attr "
            f"and no --model_name was given. Pass --model_name explicitly and "
            f"make sure it matches the model that produced this HDF5 — a "
            f"mismatched tokenizer decodes silently into garbage."
        )
    return cli_model_name
