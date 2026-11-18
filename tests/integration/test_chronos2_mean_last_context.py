"""Check Chronos-2's frozen mean-plus-last-context representation.

Run from the repository root with src on the Python import path:
    python tests/integration/test_chronos2_mean_last_context.py"""

import torch

from lace.backbone_chronos2 import FrozenChronos2Backbone


def main():
    # Road-62 diagnostic configuration.
    context_len = 2016
    pred_len = 288
    batch_size = 2

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    backbone = FrozenChronos2Backbone(
        context_len=context_len,
        pred_len=pred_len,
        model_name="amazon/chronos-2",
    ).to(device)

    print("Device:", device)
    print("encoder_dim:", backbone.encoder_dim)
    print("LACE latent_dim:", backbone.latent_dim)
    print("patch_size:", backbone.patch_size)

    assert backbone.encoder_dim == 768
    assert backbone.latent_dim == 1536

    assert all(
        not p.requires_grad
        for p in backbone.model.parameters()
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
        "num context patch slots:",
        backbone.last_num_context_tokens,
    )
    print(
        "valid context counts:",
        backbone.last_context_token_count.tolist(),
    )
    print(
        "last valid context indices:",
        backbone.last_context_token_index.tolist(),
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

    # Last valid index must be inside the context-patch region.
    assert torch.all(
        backbone.last_context_token_index
        < backbone.last_num_context_tokens
    )

    backbone.train()

    assert backbone.model.training is False
    assert all(
        not p.requires_grad
        for p in backbone.model.parameters()
    )

    print(
        "\nPASS: Chronos-2 mean+last-context representation is consistent."
    )


if __name__ == "__main__":
    main()
