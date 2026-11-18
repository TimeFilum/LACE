"""Check Moirai-1's frozen mean-plus-last-context representation.

Run from the repository root with src on the Python import path:
    python tests/integration/test_moirai1_mean_last_context.py"""

import math
import torch

from lace.backbone import FrozenMoiraiBackbone


def main():
    # One week of five-minute history and a one-day horizon.
    context_len = 2016
    pred_len = 288
    batch_size = 2

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    backbone = FrozenMoiraiBackbone(
        context_len=context_len,
        pred_len=pred_len,
        model_name="Salesforce/moirai-1.0-R-small",
    ).to(device)

    # ------------------------------------------------------------
    # 1. Static architecture checks
    # ------------------------------------------------------------
    print("encoder_dim:", backbone.encoder_dim)
    print("LACE latent_dim:", backbone.latent_dim)
    print("candidate patches:", backbone.patch_sizes)

    assert backbone.encoder_dim == 384, (
        f"Expected Moirai-1 small encoder width 384, "
        f"got {backbone.encoder_dim}"
    )

    assert backbone.latent_dim == 768, (
        f"Expected [mean;last] width 768, got {backbone.latent_dim}"
    )

    assert all(
        not p.requires_grad
        for p in backbone.model.parameters()
    ), "Frozen Moirai unexpectedly has trainable parameters."

    # ------------------------------------------------------------
    # 2. Explicitly inspect token semantics BEFORE using forward().
    #    This proves raw latent[:, -1] corresponds to prediction,
    #    not context, for every candidate patch.
    # ------------------------------------------------------------
    x = torch.randn(
        batch_size,
        context_len,
        1,
        device=device,
    )

    (
        past_target,
        past_observed,
        past_is_pad,
    ) = backbone._build_auto_inputs(x)

    context_target = past_target[
        ..., -context_len:, :
    ]
    context_observed = past_observed[
        ..., -context_len:, :
    ]
    context_is_pad = past_is_pad[
        ..., -context_len:
    ]

    print("\n=== TOKEN MASK CHECK ===")

    for patch_size in backbone.patch_sizes:
        (
            _,
            _,
            _,
            _,
            _,
            prediction_mask,
        ) = backbone.model._convert(
            patch_size=patch_size,
            past_target=context_target,
            past_observed_target=context_observed,
            past_is_pad=context_is_pad,
            future_target=None,
            future_observed_target=None,
            future_is_pad=None,
            feat_dynamic_real=None,
            observed_feat_dynamic_real=None,
            past_feat_dynamic_real=None,
            past_observed_feat_dynamic_real=None,
        )

        context_mask = ~prediction_mask
        context_counts = context_mask.sum(dim=1)
        prediction_counts = prediction_mask.sum(dim=1)

        expected_context_tokens = math.ceil(
            context_len / patch_size
        )
        expected_prediction_tokens = math.ceil(
            pred_len / patch_size
        )

        print(
            f"patch={patch_size:>3d} | "
            f"total={prediction_mask.shape[1]:>4d} | "
            f"context={context_counts.tolist()} | "
            f"prediction={prediction_counts.tolist()} | "
            f"raw_last_is_prediction="
            f"{prediction_mask[:, -1].tolist()}"
        )

        assert torch.all(
            context_counts == expected_context_tokens
        ), (
            f"Context token count mismatch for patch={patch_size}: "
            f"got {context_counts.tolist()}, "
            f"expected {expected_context_tokens}"
        )

        assert torch.all(
            prediction_counts == expected_prediction_tokens
        ), (
            f"Prediction token count mismatch for patch={patch_size}: "
            f"got {prediction_counts.tolist()}, "
            f"expected {expected_prediction_tokens}"
        )

        assert torch.all(
            prediction_mask[:, -1]
        ), (
            f"Raw final token was not a prediction token for "
            f"patch={patch_size}."
        )

        # For this univariate/no-feature wrapper, context tokens precede
        # prediction tokens. Therefore the actual last context index is
        # expected_context_tokens - 1.
        token_idx = torch.arange(
            prediction_mask.shape[1],
            device=device,
        ).unsqueeze(0).expand(
            batch_size,
            -1,
        )

        last_context_idx = (
            token_idx
            .masked_fill(~context_mask, -1)
            .max(dim=1)
            .values
        )

        assert torch.all(
            last_context_idx == expected_context_tokens - 1
        ), (
            f"Unexpected last context index for patch={patch_size}: "
            f"{last_context_idx.tolist()}"
        )

    # ------------------------------------------------------------
    # 3. Full auto-patch + representation test
    # ------------------------------------------------------------
    print("\n=== FULL FORWARD CHECK ===")

    backbone.eval()

    with torch.no_grad():
        z = backbone(x)

    print("z shape:", tuple(z.shape))
    print(
        "selected patches:",
        backbone.last_selected_patch_sizes.tolist(),
    )
    print(
        "selected context counts:",
        backbone.last_selected_context_token_counts.tolist(),
    )
    print(
        "selected last-context indices:",
        backbone.last_selected_context_token_indices.tolist(),
    )

    assert z.shape == (
        batch_size,
        2 * backbone.encoder_dim,
    )

    assert torch.isfinite(z).all(), (
        "Representation contains NaN/Inf."
    )

    assert all(
        int(p) in backbone.patch_sizes
        for p in backbone.last_selected_patch_sizes.tolist()
    )

    # In this univariate setup:
    # last_context_idx == number_of_context_tokens - 1.
    assert torch.equal(
        backbone.last_selected_context_token_indices.cpu(),
        (
            backbone.last_selected_context_token_counts - 1
        ).cpu(),
    )

    # Split the representation to make sure both halves exist.
    z_mean = z[:, : backbone.encoder_dim]
    z_last = z[:, backbone.encoder_dim :]

    print("mean half:", tuple(z_mean.shape))
    print("last half:", tuple(z_last.shape))

    assert z_mean.shape == (
        batch_size,
        backbone.encoder_dim,
    )
    assert z_last.shape == (
        batch_size,
        backbone.encoder_dim,
    )

    # ------------------------------------------------------------
    # 4. Frozen-mode protection
    # ------------------------------------------------------------
    backbone.train()

    assert backbone.model.training is False, (
        "MoiraiForecast left eval mode after parent train()."
    )

    assert all(
        not p.requires_grad
        for p in backbone.model.parameters()
    )

    print("\nPASS: mean+last-context Moirai-1 backbone is consistent.")


if __name__ == "__main__":
    main()
