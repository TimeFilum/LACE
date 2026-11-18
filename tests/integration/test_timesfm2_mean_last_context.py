"""Check TimesFM-2's frozen mean-plus-last-context representation.

Run from the repository root with src on the Python import path:
    python tests/integration/test_timesfm2_mean_last_context.py"""

import math
import torch

from lace.backbone_timesfm2 import FrozenTimesFM2Backbone


def main():
    # Road-62 diagnostic configuration.
    context_len = 2016
    pred_len = 288
    batch_size = 1

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    backbone = FrozenTimesFM2Backbone(
        context_len=context_len,
        pred_len=pred_len,
        model_name="google/timesfm-2.0-500m-pytorch",
    ).to(device)

    print("Device:", device)
    print("encoder_dim:", backbone.encoder_dim)
    print("LACE latent_dim:", backbone.latent_dim)
    print("patch_size:", backbone.patch_size)

    assert backbone.encoder_dim == 1280
    assert backbone.latent_dim == 2560

    expected_tokens = math.ceil(
        context_len / backbone.patch_size
    )

    print("expected context tokens:", expected_tokens)

    assert all(
        not p.requires_grad
        for p in backbone.forecast_model.parameters()
    )

    x = torch.randn(
        batch_size,
        context_len,
        1,
        device=device,
    )

    backbone.eval()

    with torch.no_grad():
        z = backbone(x)

    print("\n=== FULL FORWARD CHECK ===")
    print("z shape:", tuple(z.shape))
    print(
        "actual context tokens:",
        backbone.last_num_context_tokens,
    )
    print(
        "expected context tokens:",
        backbone.last_expected_context_tokens,
    )

    assert backbone.last_num_context_tokens == expected_tokens
    assert (
        backbone.last_expected_context_tokens
        == expected_tokens
    )

    assert z.shape == (
        batch_size,
        2 * backbone.encoder_dim,
    )
    assert torch.isfinite(z).all()

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

    backbone.train()

    assert backbone.forecast_model.training is False
    assert backbone.model.training is False

    assert all(
        not p.requires_grad
        for p in backbone.forecast_model.parameters()
    )

    print(
        "\nPASS: TimesFM-2 mean+last-context representation is consistent."
    )


if __name__ == "__main__":
    main()
