import math

import torch
from torch import nn

from transformers import TimesFmModelForPrediction


class FrozenTimesFM2Backbone(nn.Module):
    """
    Frozen TimesFM 2.0 backbone for LACE.

    Input
    -----
    x_enc : [B, context_len, 1]

    Output
    ------
    z0 : [B, latent_dim]

    Uses the pretrained TimesFM context representation only.
    Returns [context-patch mean ; last context patch].
    No future generation is performed.
    """

    def __init__(
        self,
        context_len: int,
        pred_len: int,
        model_name: str = "google/timesfm-2.0-500m-pytorch",
    ):
        super().__init__()

        self.context_len = int(context_len)
        self.pred_len = int(pred_len)

        # ----------------------------------------------------------
        # 1. Load pretrained TimesFM2
        # ----------------------------------------------------------
        forecast_model = TimesFmModelForPrediction.from_pretrained(
            model_name,
            attn_implementation="sdpa",
        )

        forecast_model.eval()

        for param in forecast_model.parameters():
            param.requires_grad = False

        self.forecast_model = forecast_model

        # Native TimesFM hidden width.
        self.encoder_dim = int(
            forecast_model.config.hidden_size
        )

        # LACE representation:
        #   [context-patch mean ; last context patch]
        # TimesFM-2 500M: 1280 + 1280 = 2560.
        self.latent_dim = 2 * self.encoder_dim

        self.patch_size = int(
            forecast_model.config.patch_length
        )

        # Diagnostics for the most recent forward pass.
        self.last_num_context_tokens = None
        self.last_expected_context_tokens = None

        # TimesFmModelForPrediction contains the bare TimesFmModel.
        if not hasattr(forecast_model, "decoder"):
            raise RuntimeError(
                "TimesFmModelForPrediction does not expose `.decoder`."
            )

        # Bare TimesFM transformer/backbone.
        # Calling this directly avoids recursive future generation.
        self.model = forecast_model.decoder

    def train(self, mode: bool = True):
        """
        Keep TimesFM frozen/eval even when parent LACE enters train mode.
        """
        super().train(mode)
        self.forecast_model.eval()
        self.model.eval()
        return self

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Return [mean(context patches) ; last context patch].

        This wrapper calls the bare TimesFM decoder on context only.
        We additionally require the returned hidden-state length to equal
        ceil(context_len / patch_size), so H[:, -1, :] is accepted only
        when the sequence is exactly the expected context-patch sequence.
        """

        if x_enc.ndim != 3:
            raise ValueError(
                f"Expected x_enc [B,L,C], got {x_enc.shape}."
            )

        B, L, C = x_enc.shape

        if C != 1:
            raise ValueError(
                f"LACE currently expects univariate input, got C={C}."
            )

        if L != self.context_len:
            raise ValueError(
                f"Expected context length {self.context_len}, got {L}."
            )

        # TimesFM expects [B,L].
        past_values = (
            x_enc[..., 0]
            .to(dtype=torch.float32)
        )

        # No padded observations in our fixed windows.
        past_values_padding = torch.zeros(
            B,
            L,
            dtype=torch.long,
            device=x_enc.device,
        )

        # All current LACE base resolutions are minute-level:
        # 5m / 10m / 15m -> TimesFM frequency category 0.
        freq = torch.zeros(
            B,
            1,
            dtype=torch.long,
            device=x_enc.device,
        )

        with torch.no_grad():
            outputs = self.model(
                past_values=past_values,
                past_values_padding=past_values_padding,
                freq=freq,
                return_dict=True,
            )

            H = outputs.last_hidden_state

        if H.ndim != 3:
            raise RuntimeError(
                f"Expected TimesFM latent [B,T,D], got {H.shape}."
            )

        if H.shape[0] != B:
            raise RuntimeError(
                f"Batch mismatch: input={B}, latent={H.shape[0]}."
            )

        if H.shape[-1] != self.encoder_dim:
            raise RuntimeError(
                f"Expected encoder dim {self.encoder_dim}, "
                f"got {H.shape[-1]}."
            )

        # The bare TimesFM model is called on context only.
        # Verify the token sequence before accepting the raw final
        # hidden state as the last context-patch representation.
        expected_context_tokens = math.ceil(
            self.context_len / self.patch_size
        )

        if H.shape[1] != expected_context_tokens:
            raise RuntimeError(
                "Unexpected TimesFM hidden-token layout: "
                f"got {H.shape[1]} tokens, expected "
                f"{expected_context_tokens}=ceil("
                f"{self.context_len}/{self.patch_size}). "
                "Refusing to use H[:, -1, :] until the "
                "context-token layout is verified."
            )

        z_mean = H.mean(dim=1)
        z_last = H[:, -1, :]

        z0 = torch.cat(
            [
                z_mean,
                z_last,
            ],
            dim=-1,
        )

        self.last_num_context_tokens = int(H.shape[1])
        self.last_expected_context_tokens = int(
            expected_context_tokens
        )

        if z0.shape != (B, self.latent_dim):
            raise RuntimeError(
                f"Unexpected TimesFM mean+last shape {z0.shape}; "
                f"expected ({B}, {self.latent_dim})."
            )

        return z0
