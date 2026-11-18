import torch
from torch import nn

from uni2ts.model.moirai2 import (
    Moirai2Forecast,
    Moirai2Module,
)


class FrozenMoirai2Backbone(nn.Module):
    """
    Frozen Moirai-2.0 backbone for LACE.

    Input
    -----
    x_enc : [B, context_len, 1]

    Output
    ------
    z0 : [B, latent_dim]

    The representation concatenates the CONTEXT-token mean with the actual last CONTEXT token.
    Prediction tokens are never included in the shared LACE state.
    """

    def __init__(
        self,
        context_len: int,
        pred_len: int,
        model_name: str = "Salesforce/moirai-2.0-R-small",
    ):
        super().__init__()

        self.context_len = int(context_len)
        self.pred_len = int(pred_len)

        # ----------------------------------------------------------
        # 1. Load and freeze Moirai2
        # ----------------------------------------------------------
        module = Moirai2Module.from_pretrained(model_name)
        module.eval()

        for param in module.parameters():
            param.requires_grad = False

        # Native Moirai-2 encoder width.
        self.encoder_dim = int(module.d_model)

        # LACE shared representation:
        #   [context mean ; last context token]
        # Moirai-2 small: 384 + 384 = 768.
        self.latent_dim = 2 * self.encoder_dim

        self.patch_size = int(module.patch_size)

        # Official Moirai2 preprocessing/forecast wrapper.
        self.model = Moirai2Forecast(
            module=module,
            prediction_length=self.pred_len,
            context_length=self.context_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        self.model.eval()

        # ----------------------------------------------------------
        # 2. Encoder hook
        # ----------------------------------------------------------
        self._last_latent = None

        self._encoder_hook_handle = (
            module.encoder.register_forward_hook(
                self._capture_encoder_output
            )
        )

        # Diagnostics for the most recent forward pass.
        self.last_context_token_count = None
        self.last_prediction_token_count = None
        self.last_context_token_index = None
        self.last_total_token_count = None

    def _capture_encoder_output(self, module, inputs, output):
        if not torch.is_tensor(output):
            raise RuntimeError(
                "Expected Moirai2 encoder output Tensor, "
                f"got {type(output)}."
            )

        self._last_latent = output.detach()

    def train(self, mode: bool = True):
        """
        LACE may enter train mode, but the TSFM must stay frozen/eval.
        """
        super().train(mode)
        self.model.eval()
        self.model.module.eval()
        return self

    def _build_inputs(self, x_enc):
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

        past_target = x_enc.to(dtype=torch.float32)

        past_observed_target = torch.ones(
            B,
            L,
            C,
            dtype=torch.bool,
            device=x_enc.device,
        )

        past_is_pad = torch.zeros(
            B,
            L,
            dtype=torch.bool,
            device=x_enc.device,
        )

        return (
            past_target,
            past_observed_target,
            past_is_pad,
        )

    def forward(self, x_enc):
        """
        Extract one frozen Moirai-2 representation per input series.

        Shared LACE representation:

            z0 = [ mean(context encoder tokens) ;
                   last context encoder token ]

        ``prediction_mask`` is used explicitly:
            False -> context token
            True  -> prediction token

        Therefore the last-context representation is selected from the
        actual context-token mask rather than assuming ``H[:, -1, :]``.
        """

        (
            past_target,
            past_observed_target,
            past_is_pad,
        ) = self._build_inputs(x_enc)

        batch_size = x_enc.shape[0]

        # ----------------------------------------------------------
        # Moirai-2 creates context + prediction tokens in one convert
        # call, then performs one frozen encoder pass.
        # ----------------------------------------------------------
        with torch.no_grad():

            (
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
            ) = self.model._convert(
                self.patch_size,
                past_target,
                past_observed_target,
                past_is_pad,
                feat_dynamic_real=None,
                observed_feat_dynamic_real=None,
                past_feat_dynamic_real=None,
                past_observed_feat_dynamic_real=None,
            )

            self._last_latent = None

            _ = self.model.module(
                target=target,
                observed_mask=observed_mask,
                sample_id=sample_id,
                time_id=time_id,
                variate_id=variate_id,
                prediction_mask=prediction_mask,
                training_mode=False,
            )

            if self._last_latent is None:
                raise RuntimeError(
                    "Moirai-2 encoder hook did not fire."
                )

            H = self._last_latent

            if H.ndim != 3:
                raise RuntimeError(
                    f"Expected Moirai-2 latent [B,T,D], got {H.shape}."
                )

            if H.shape[0] != batch_size:
                raise RuntimeError(
                    f"Batch mismatch: input={batch_size}, latent={H.shape[0]}."
                )

            if H.shape[-1] != self.encoder_dim:
                raise RuntimeError(
                    f"Expected encoder dim {self.encoder_dim}, "
                    f"got {H.shape[-1]}."
                )

            if prediction_mask.shape != H.shape[:2]:
                raise RuntimeError(
                    "Moirai-2 prediction_mask shape does not match encoder "
                    f"tokens: mask={prediction_mask.shape}, H={H.shape}."
                )

            # ------------------------------------------------------
            # Context-token semantics.
            # ------------------------------------------------------
            context_mask = ~prediction_mask

            context_counts = context_mask.sum(dim=1)
            prediction_counts = prediction_mask.sum(dim=1)

            if torch.any(context_counts <= 0):
                raise RuntimeError(
                    "Moirai-2 produced no context tokens."
                )

            if torch.any(prediction_counts <= 0):
                raise RuntimeError(
                    "Moirai-2 produced no prediction tokens."
                )

            # ------------------------------------------------------
            # A. Context-only mean representation.
            # ------------------------------------------------------
            weights = context_mask.to(H.dtype).unsqueeze(-1)

            z_mean = (
                (H * weights).sum(dim=1)
                /
                weights.sum(dim=1).clamp_min(1.0)
            )

            # ------------------------------------------------------
            # B. Actual last CONTEXT token.
            #
            # Do not use H[:, -1, :] because the final raw encoder
            # position belongs to the prediction region.
            # ------------------------------------------------------
            token_idx = torch.arange(
                H.shape[1],
                device=H.device,
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

            if torch.any(last_context_idx < 0):
                raise RuntimeError(
                    "Could not locate the last Moirai-2 context token."
                )

            batch_idx = torch.arange(
                batch_size,
                device=H.device,
            )

            z_last = H[
                batch_idx,
                last_context_idx,
                :
            ]

            # Guard against accidentally selecting a prediction token.
            if torch.any(
                prediction_mask[
                    batch_idx,
                    last_context_idx,
                ]
            ):
                raise RuntimeError(
                    "Selected Moirai-2 last-context index is marked as "
                    "a prediction token."
                )

            z0 = torch.cat(
                [
                    z_mean,
                    z_last,
                ],
                dim=-1,
            )

            # Diagnostics.
            self.last_context_token_count = context_counts.detach()
            self.last_prediction_token_count = prediction_counts.detach()
            self.last_context_token_index = last_context_idx.detach()
            self.last_total_token_count = int(H.shape[1])

        if z0.shape != (
            batch_size,
            self.latent_dim,
        ):
            raise RuntimeError(
                "Unexpected Moirai-2 mean+last representation: "
                f"{z0.shape}; expected "
                f"({batch_size}, {self.latent_dim})."
            )

        return z0

    def remove_hook(self):
        if self._encoder_hook_handle is not None:
            self._encoder_hook_handle.remove()
            self._encoder_hook_handle = None
