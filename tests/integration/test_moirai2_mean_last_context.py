"""Check Moirai-2's frozen mean-plus-last-context representation.

Run from the repository root with src on the Python import path:
    python tests/integration/test_moirai2_mean_last_context.py"""

import torch

from lace.backbone_moirai2 import FrozenMoirai2Backbone


def main():
    # Road-62 / K-traffic both use:
    #   finest resolution = 5m
    #   one week context  = 2016
    #   one day horizon   = 288
    context_len = 2016
    pred_len = 288
    batch_size = 2

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    backbone = FrozenMoirai2Backbone(
        context_len=context_len,
        pred_len=pred_len,
        model_name="Salesforce/moirai-2.0-R-small",
    ).to(device)

    # ------------------------------------------------------------
    # 1. Static architecture checks
    # ------------------------------------------------------------
    print("encoder_dim:", backbone.encoder_dim)
    print("LACE latent_dim:", backbone.latent_dim)
    print("patch_size:", backbone.patch_size)

    assert backbone.encoder_dim == 384, (
        f"Expected Moirai-2 small encoder width 384, "
        f"got {backbone.encoder_dim}"
    )

    assert backbone.latent_dim == 768, (
        f"Expected [mean;last] width 768, got {backbone.latent_dim}"
    )

    assert all(
        not p.requires_grad
        for p in backbone.model.parameters()
    ), "Frozen Moirai-2 unexpectedly has trainable parameters."

    # ------------------------------------------------------------
    # 2. Inspect token semantics directly from _convert()
    # ------------------------------------------------------------
    x = torch.randn(
        batch_size,
        context_len,
        1,
        device=device,
    )

    (
        past_target,
        past_observed_target,
        past_is_pad,
    ) = backbone._build_inputs(x)

    (
        _,
        _,
        _,
        _,
        _,
        prediction_mask,
    ) = backbone.model._convert(
        backbone.patch_size,
        past_target,
        past_observed_target,
        past_is_pad,
        feat_dynamic_real=None,
        observed_feat_dynamic_real=None,
        past_feat_dynamic_real=None,
        past_observed_feat_dynamic_real=None,
    )

    context_mask = ~prediction_mask

    context_counts = context_mask.sum(dim=1)
    prediction_counts = prediction_mask.sum(dim=1)

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

    batch_idx = torch.arange(
        batch_size,
        device=device,
    )

    print("\n=== TOKEN MASK CHECK ===")
    print("mask shape:", tuple(prediction_mask.shape))
    print("context counts:", context_counts.tolist())
    print("prediction counts:", prediction_counts.tolist())
    print("last context indices:", last_context_idx.tolist())
    print(
        "raw final token is prediction:",
        prediction_mask[:, -1].tolist(),
    )

    assert torch.all(context_counts > 0)
    assert torch.all(prediction_counts > 0)

    # The selected index itself must be context.
    assert torch.all(
        ~prediction_mask[
            batch_idx,
            last_context_idx,
        ]
    ), "Recovered last-context index is not context."

    # By construction, there must be no later context token.
    for b in range(batch_size):
        idx = int(last_context_idx[b].item())
        assert not context_mask[b, idx + 1 :].any(), (
            f"Sample {b}: found a context token after "
            f"last_context_idx={idx}."
        )

    # For the current Moirai-2 token layout, the raw final token should
    # be part of the forecast/prediction region.
    assert torch.all(prediction_mask[:, -1]), (
        "Raw final Moirai-2 token was not marked as prediction."
    )

    # ------------------------------------------------------------
    # 3. Full frozen forward
    # ------------------------------------------------------------
    print("\n=== FULL FORWARD CHECK ===")

    backbone.eval()

    with torch.no_grad():
        z = backbone(x)

    print("z shape:", tuple(z.shape))
    print(
        "context counts:",
        backbone.last_context_token_count.tolist(),
    )
    print(
        "prediction counts:",
        backbone.last_prediction_token_count.tolist(),
    )
    print(
        "last-context indices:",
        backbone.last_context_token_index.tolist(),
    )
    print(
        "total tokens:",
        backbone.last_total_token_count,
    )

    assert z.shape == (
        batch_size,
        2 * backbone.encoder_dim,
    )

    assert torch.isfinite(z).all(), (
        "Representation contains NaN/Inf."
    )

    # Split the two halves explicitly.
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

    # Diagnostics from forward() must agree with our independent mask check.
    assert torch.equal(
        backbone.last_context_token_count.cpu(),
        context_counts.cpu(),
    )

    assert torch.equal(
        backbone.last_prediction_token_count.cpu(),
        prediction_counts.cpu(),
    )

    assert torch.equal(
        backbone.last_context_token_index.cpu(),
        last_context_idx.cpu(),
    )

    # ------------------------------------------------------------
    # 4. Frozen/eval protection
    # ------------------------------------------------------------
    backbone.train()

    assert backbone.model.training is False
    assert backbone.model.module.training is False

    assert all(
        not p.requires_grad
        for p in backbone.model.parameters()
    )

    print(
        "\nPASS: Moirai-2 mean+last-context backbone is consistent."
    )


if __name__ == "__main__":
    main()
