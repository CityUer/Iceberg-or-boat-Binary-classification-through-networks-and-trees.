"""Candidate registry for the 2026-09-17 round.

Each entry changes exactly ONE thing relative to its declared baseline.
`baseline` names the config it must be compared against with a paired OOF test.
"""

from __future__ import annotations

from typing import Any

# The historical best single model, reproduced under the frozen protocol.
BASE_KWARGS: dict[str, Any] = dict(
    angle_mode="film",
    activation="mish",
    width=32,
    depth=3,
    dropout=0.15,
    head_dropout=0.4,
)


def _cnn(**overrides: Any) -> dict[str, Any]:
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    return kwargs


CONFIGS: dict[str, dict[str, Any]] = {
    # ---------------------------------------------------------------- A0 baseline
    "a0_base": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(),
        baseline=None,
        hypothesis="Reproduce E18c_mish_film (Mish+FiLM) as the frozen-protocol reference.",
        change="none (baseline)",
    ),
    # ------------------------------------------- A0 angle-mode re-test (plan 1.4)
    "a0_noangle": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(angle_mode="none"),
        baseline="a0_base",
        hypothesis="How much does the angle actually contribute under the frozen protocol?",
        change="angle_mode: film -> none",
    ),
    "a0_scalar": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(angle_mode="scalar"),
        baseline="a0_base",
        hypothesis=(
            "Re-test the report's claim that a constant angle channel is mathematically "
            "equivalent to using no angle. Phase 0 showed the channel holds 879 distinct "
            "values, so the claim is false as stated."
        ),
        change="angle_mode: film -> scalar",
    ),
    "a0_channel": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(angle_mode="channel"),
        baseline="a0_base",
        hypothesis="Spatial angle channels (constant + distance + proximity) vs FiLM.",
        change="angle_mode: film -> channel",
    ),
    # ------------------------------------------------------------ A1 third channel
    "a1_diff3ch": dict(
        model_name="cnn",
        channel_mode="hh_hv_diff",
        model_kwargs=_cnn(),
        baseline="a0_base",
        hypothesis=(
            "HH-HV is the polarimetric difference and gives the model a useful inductive "
            "bias for free; no new information, only a better parameterisation."
        ),
        change="input channels: (HH,HV) -> (HH,HV,HH-HV)",
    ),
    # ------------------------------------------------- A2 pretrained vs scratch CNN
    "a2_resnet18_scratch": dict(
        model_name="resnet18",
        channel_mode="hh_hv_diff",
        model_kwargs=dict(angle_mode="mlp", pretrained=False, input_size=112,
                          head_dropout=0.4, angle_embed=32),
        baseline="a1_diff3ch",
        hypothesis="ResNet-18 architecture alone (random init) vs the compact CNN.",
        change="architecture: compact CNN -> ResNet-18, random init",
    ),
    "a2_resnet18_pretrained": dict(
        model_name="resnet18",
        channel_mode="hh_hv_diff",
        model_kwargs=dict(angle_mode="mlp", pretrained=True, input_size=112,
                          head_dropout=0.4, angle_embed=32),
        baseline="a2_resnet18_scratch",
        hypothesis=(
            "Does ImageNet pretraining transfer to SAR? Only meaningful against the "
            "identically-architected random-init run."
        ),
        change="weights: random init -> ImageNet pretrained",
    ),
    # ------------------------------------- B members: historical ensemble components
    # These exist to give work line B (ensembling) an honest OOF basis. The old
    # ensemble experiments had no OOF, so their weight choices could not be
    # validated. Registering these two reproduces the two non-Mish families that
    # appeared in the old champion core.
    "b_member_gelu_channel": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(activation="gelu", angle_mode="channel"),
        baseline="a0_base",
        hypothesis=(
            "Reproduce E11c_gelu_ch as an OOF-scored ensemble member, so ensemble "
            "weights can be fitted on out-of-fold predictions instead of guessed."
        ),
        change="activation mish->gelu, angle_mode film->channel",
    ),
    "b_member_gelu_film": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(activation="gelu", angle_mode="film"),
        baseline="a0_base",
        hypothesis=(
            "Reproduce E11f_gelu_film as an OOF-scored ensemble member."
        ),
        change="activation mish->gelu (angle_mode stays film)",
    ),
    "b_member_silu_film": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(activation="silu", angle_mode="film"),
        baseline="a0_base",
        hypothesis=(
            "Reproduce E05g_noaug_silu (SiLU+FiLM) as an OOF-scored ensemble member."
        ),
        change="activation mish->silu (angle_mode stays film)",
    ),
    # ------------------------------------------- A1 secondary: physically motivated
    "a1_bgsub": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(),
        baseline="a0_base",
        hypothesis=(
            "Subtracting a 3x3 box-blurred background suppresses the smooth ocean "
            "return and boosts the strong target scatterer (high-pass). Physical "
            "motivation: target-background contrast is what separates ship/iceberg."
        ),
        change="add background subtraction (high-pass) before standardisation",
        preprocess=dict(subtract_background=True),
    ),
    "a1_log": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(),
        baseline="a0_base",
        hypothesis=(
            "Backscatter is already in dB but retains a bright point-scatterer tail; "
            "log1p compression tames it."
        ),
        change="add log1p compression before standardisation",
        preprocess=dict(image_log=True),
    ),
    # ------------------------------------------------ A1 inference-only: 8-view TTA
    "a1_tta": dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(),
        baseline="a0_base",
        hypothesis=(
            "Target orientation in SAR is arbitrary, so the decision should be roughly "
            "invariant under the dihedral group. TTA is inference-only and cannot "
            "affect training, so it is a clean variance-reduction lever."
        ),
        change="OOF inference: single view -> 8-view rotation/flip TTA",
        tta=True,
    ),
}


def get(name: str) -> dict[str, Any]:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; known: {sorted(CONFIGS)}")
    return CONFIGS[name]


# ---------------------------------------------------------------------------
# Epoch-budget ablation.
#
# The 15-epoch budget is the ONE protocol constant every candidate shares, and it
# was chosen from a noisy 8-run pilot (mean validation loss minimised at epoch 13,
# but epochs 12-17 formed a broad plateau). Two things are worth testing:
#   1. is 15 actually near the optimum, or was 13 just noise?
#   2. the full-data model sees 1604 samples while the fold model sees 1283
#      (+25%), so it may need a LONGER budget than the one tuned on folds.
#
# Registered as its own family so the comparison stays one-change-at-a-time.
# ---------------------------------------------------------------------------
for _e in (10, 12, 18, 20, 25):
    CONFIGS[f"a0_ep{_e}"] = dict(
        model_name="cnn",
        channel_mode="hh_hv",
        model_kwargs=_cnn(),
        baseline="a0_base",
        epochs=_e,
        hypothesis=(f"Epoch budget {_e} vs the frozen 15 "
                    f"(pilot optimum was 13 on a noisy 8-run curve)."),
        change=f"training budget: 15 -> {_e} epochs",
    )

