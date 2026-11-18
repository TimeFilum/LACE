"""
Frozen Moirai-1.0 backbone used by LACE.

Responsibility
--------------
Given a univariate history x_enc [B, L, 1], reproduce Moirai-1.0's
``patch_size='auto'`` selection logic and return the corresponding
context mean + last-context encoder representation:

    x_enc
      -> Moirai auto patch selection
      -> H_p for each candidate patch p
      -> recover context tokens using Moirai prediction_mask
      -> context mean + last context token
      -> select the representation for Moirai's chosen patch
      -> z0 [B, 2 * D_lat]

The patch is selected independently for each sample using the same
per-sample validation loss criterion as MoiraiForecast.forward().

This module performs no adaptation, hierarchy operations, forecast-head
computation, representation normalization, or LACE loss calculation.

Moirai is permanently frozen and kept in evaluation mode.
"""

import torch
from torch import nn

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


class FrozenMoiraiBackbone(nn.Module):
    """Frozen Moirai-1.0 wrapper with faithful automatic patch selection."""

    def __init__(
        self,
        context_len: int,
        pred_len: int,
        model_name: str = "Salesforce/moirai-1.0-R-small",
    ):
        super().__init__()

        self.context_len = int(context_len)
        self.pred_len = int(pred_len)

        # ----------------------------------------------------------
        # 1. Load and freeze pretrained Moirai
        # ----------------------------------------------------------
        backbone = MoiraiModule.from_pretrained(model_name)
        backbone.eval()

        for param in backbone.parameters():
            param.requires_grad = False

        self.encoder_dim = int(
            backbone.encoder.layers[0]
            .self_attn.q_proj.out_features
        )

        # LACE representation = [context mean ; last context token].
        # Moirai-1 small: 384 + 384 = 768.
        self.latent_dim = 2 * self.encoder_dim

        self.patch_sizes = tuple(int(p) for p in backbone.patch_sizes)

        # Keep Moirai configured exactly as the baseline experiment:
        # patch_size="auto". We reproduce its auto-selection loop below,
        # but stop at the selected pooled encoder representation instead
        # of sampling a forecast.
        self.model = MoiraiForecast(
            module=backbone,
            prediction_length=self.pred_len,
            context_length=self.context_len,
            patch_size="auto",
            num_samples=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        # ----------------------------------------------------------
        # 2. Capture encoder output
        # ----------------------------------------------------------
        self._last_latent = None
        self._encoder_hook_handle = (
            backbone.encoder.register_forward_hook(
                self._capture_encoder_output
            )
        )

        # Diagnostics only: values for the most recent batch.
        self.last_selected_patch_sizes = None
        self.last_selected_context_token_indices = None
        self.last_selected_context_token_counts = None

    # ==============================================================
    # Encoder hook
    # ==============================================================

    def _capture_encoder_output(self, module, inputs, output):
        """Store the complete Moirai encoder output [B, T_lat, D_lat]."""
        if not torch.is_tensor(output):
            raise RuntimeError(
                "Expected Moirai encoder output to be a Tensor, "
                f"got {type(output)}."
            )

        self._last_latent = output.detach()

    # ==============================================================
    # Frozen-mode protection
    # ==============================================================

    def train(self, mode: bool = True):
        """Keep Moirai in eval mode when the parent LACE model trains."""
        super().train(mode)
        self.model.eval()
        return self

    # ==============================================================
    # Input construction for Moirai auto patching
    # ==============================================================

    def _build_auto_inputs(self, x_enc: torch.Tensor):
        """
        Build the input expected by Moirai when ``patch_size='auto'``.

        Moirai's auto mode expects ``context_length + prediction_length``
        past positions. LACE has only the physical context, so we prepend
        ``pred_len`` padding positions that are both unobserved and marked
        as padding.

        The resulting sequence lets Moirai use the last ``pred_len``
        points of the real context as its pseudo-validation future while
        still using the full real context for the subsequent latent pass.
        """
        if x_enc.ndim != 3:
            raise ValueError(
                f"Expected x_enc [B,L,C], got {x_enc.shape}."
            )

        batch_size, length, channels = x_enc.shape

        if channels != 1:
            raise ValueError(
                f"LACE currently expects univariate input, got C={channels}."
            )

        if length != self.context_len:
            raise ValueError(
                f"Expected context length {self.context_len}, got {length}."
            )

        x = x_enc.to(dtype=torch.float32)

        prefix = torch.zeros(
            batch_size,
            self.pred_len,
            channels,
            dtype=x.dtype,
            device=x.device,
        )
        past_target = torch.cat([prefix, x], dim=1)

        prefix_observed = torch.zeros(
            batch_size,
            self.pred_len,
            channels,
            dtype=torch.bool,
            device=x.device,
        )
        history_observed = torch.ones(
            batch_size,
            self.context_len,
            channels,
            dtype=torch.bool,
            device=x.device,
        )
        past_observed = torch.cat(
            [prefix_observed, history_observed],
            dim=1,
        )

        prefix_is_pad = torch.ones(
            batch_size,
            self.pred_len,
            dtype=torch.bool,
            device=x.device,
        )
        history_is_pad = torch.zeros(
            batch_size,
            self.context_len,
            dtype=torch.bool,
            device=x.device,
        )
        past_is_pad = torch.cat(
            [prefix_is_pad, history_is_pad],
            dim=1,
        )

        expected_length = self.context_len + self.pred_len
        if past_target.shape[1] != expected_length:
            raise RuntimeError(
                "Unexpected Moirai auto input length: "
                f"got {past_target.shape[1]}, expected {expected_length}."
            )

        if self.model.past_length != expected_length:
            raise RuntimeError(
                "Moirai auto past_length does not match the constructed "
                f"input: {self.model.past_length} vs {expected_length}."
            )

        return past_target, past_observed, past_is_pad

    # ==============================================================
    # Automatic patch selection + pooled latent extraction
    # ==============================================================

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Return Moirai's auto-selected shared representation:

            z0 = [ mean(context encoder tokens) ;
                   last context encoder token ]

        IMPORTANT
        ---------
        ``MoiraiForecast._get_distr()`` appends forecast/prediction tokens.
        Therefore ``latent[:, -1, :]`` is NOT the last context token.

        We reconstruct Moirai's ``prediction_mask`` for each candidate patch:
            False -> context token
            True  -> prediction token

        and use that mask both for context-only mean pooling and for selecting
        the actual last context token.

        Returns
        -------
        z0 : Tensor [B, 2 * encoder_dim]
        """
        (
            past_target,
            past_observed,
            past_is_pad,
        ) = self._build_auto_inputs(x_enc)

        batch_size = x_enc.shape[0]

        val_losses = []
        candidate_representations = []
        candidate_last_context_idx = []
        candidate_context_counts = []

        # Physical context passed to the forecast latent extraction.
        context_target = past_target[
            ..., -self.context_len :, :
        ]
        context_observed = past_observed[
            ..., -self.context_len :, :
        ]
        context_is_pad = past_is_pad[
            ..., -self.context_len :
        ]

        # Frozen backbone: no gradient graph is required.
        with torch.no_grad():
            for patch_size in self.patch_sizes:
                # --------------------------------------------------
                # Moirai automatic patch-selection score.
                # --------------------------------------------------
                val_loss = self.model._val_loss(
                    patch_size=patch_size,
                    target=past_target[..., : self.model.past_length, :],
                    observed_target=past_observed[
                        ..., : self.model.past_length, :
                    ],
                    is_pad=past_is_pad[..., : self.model.past_length],
                    feat_dynamic_real=None,
                    observed_feat_dynamic_real=None,
                    past_feat_dynamic_real=None,
                    past_observed_feat_dynamic_real=None,
                )
                val_losses.append(val_loss)

                # --------------------------------------------------
                # B. Run the same forecasting-context pass and capture
                #    the complete encoder sequence.
                # --------------------------------------------------
                self._last_latent = None

                _ = self.model._get_distr(
                    patch_size=patch_size,
                    past_target=context_target,
                    past_observed_target=context_observed,
                    past_is_pad=context_is_pad,
                    feat_dynamic_real=None,
                    observed_feat_dynamic_real=None,
                    past_feat_dynamic_real=None,
                    past_observed_feat_dynamic_real=None,
                )

                if self._last_latent is None:
                    raise RuntimeError(
                        "Moirai encoder hook did not fire for "
                        f"patch_size={patch_size}."
                    )

                latent = self._last_latent

                if latent.ndim != 3:
                    raise RuntimeError(
                        "Expected latent [B,T,D] for "
                        f"patch_size={patch_size}, got {latent.shape}."
                    )

                if latent.shape[0] != batch_size:
                    raise RuntimeError(
                        "Batch-size mismatch between input and latent for "
                        f"patch_size={patch_size}: "
                        f"{batch_size} vs {latent.shape[0]}."
                    )

                if latent.shape[-1] != self.encoder_dim:
                    raise RuntimeError(
                        "Unexpected encoder dimension for "
                        f"patch_size={patch_size}: expected "
                        f"{self.encoder_dim}, got {latent.shape[-1]}."
                    )

                # --------------------------------------------------
                # C. Recover Moirai's token semantics for THIS patch.
                #
                # _get_distr() internally calls the same _convert().
                # We call _convert() here only to recover prediction_mask;
                # this does not perform another encoder/model forward pass.
                # --------------------------------------------------
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    prediction_mask,
                ) = self.model._convert(
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

                if prediction_mask.shape != latent.shape[:2]:
                    raise RuntimeError(
                        "Moirai token-mask shape does not match encoder output "
                        f"for patch_size={patch_size}: "
                        f"mask={prediction_mask.shape}, latent={latent.shape}."
                    )

                context_mask = ~prediction_mask

                context_counts = context_mask.sum(dim=1)

                if torch.any(context_counts <= 0):
                    raise RuntimeError(
                        f"No context token found for patch_size={patch_size}."
                    )

                # With our univariate/no-feature setup, Moirai should append
                # prediction tokens after context tokens. Keep this explicit
                # guard so a future Uni2TS API/layout change fails loudly.
                if torch.any(~prediction_mask[:, -1]):
                    raise RuntimeError(
                        "Expected the raw final Moirai token to be a "
                        "prediction token, but it was marked as context."
                    )

                # --------------------------------------------------
                # D1. Context-only mean.
                # --------------------------------------------------
                weights = context_mask.to(latent.dtype).unsqueeze(-1)

                z_mean = (
                    (latent * weights).sum(dim=1)
                    /
                    weights.sum(dim=1).clamp_min(1.0)
                )

                # --------------------------------------------------
                # D2. Actual LAST CONTEXT token.
                # Do not assume it is latent[:, -1, :].
                # --------------------------------------------------
                token_idx = torch.arange(
                    latent.shape[1],
                    device=latent.device,
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
                        f"Could not locate last context token for "
                        f"patch_size={patch_size}."
                    )

                batch_idx = torch.arange(
                    batch_size,
                    device=latent.device,
                )

                z_last = latent[
                    batch_idx,
                    last_context_idx,
                    :
                ]

                z_patch = torch.cat(
                    [
                        z_mean,
                        z_last,
                    ],
                    dim=-1,
                )

                if z_patch.shape != (
                    batch_size,
                    self.latent_dim,
                ):
                    raise RuntimeError(
                        "Unexpected mean+last representation for "
                        f"patch_size={patch_size}: "
                        f"{z_patch.shape}; expected "
                        f"({batch_size}, {self.latent_dim})."
                    )

                candidate_representations.append(
                    z_patch
                )
                candidate_last_context_idx.append(
                    last_context_idx
                )
                candidate_context_counts.append(
                    context_counts
                )

            # ------------------------------------------------------
            # E. Faithful Moirai per-sample patch selection.
            # ------------------------------------------------------
            val_losses = torch.stack(
                val_losses,
                dim=0,
            )  # [P, B]

            candidate_representations = torch.stack(
                candidate_representations,
                dim=0,
            )  # [P, B, 2D]

            candidate_last_context_idx = torch.stack(
                candidate_last_context_idx,
                dim=0,
            )  # [P, B]

            candidate_context_counts = torch.stack(
                candidate_context_counts,
                dim=0,
            )  # [P, B]

            selected_idx = val_losses.argmin(
                dim=0
            )  # [B]

            batch_idx = torch.arange(
                batch_size,
                device=x_enc.device,
            )

            z0 = candidate_representations[
                selected_idx,
                batch_idx,
            ]  # [B, 2D]

            patch_tensor = torch.tensor(
                self.patch_sizes,
                dtype=torch.long,
                device=x_enc.device,
            )

            self.last_selected_patch_sizes = (
                patch_tensor[selected_idx].detach()
            )

            self.last_selected_context_token_indices = (
                candidate_last_context_idx[
                    selected_idx,
                    batch_idx,
                ].detach()
            )

            self.last_selected_context_token_counts = (
                candidate_context_counts[
                    selected_idx,
                    batch_idx,
                ].detach()
            )

        if z0.shape != (
            batch_size,
            self.latent_dim,
        ):
            raise RuntimeError(
                "Unexpected shared representation shape: "
                f"got {z0.shape}, expected "
                f"({batch_size}, {self.latent_dim})."
            )

        return z0

    # ==============================================================
    # Encoder hook lifecycle
    # ==============================================================

    def remove_hook(self):
        """Remove the encoder hook if this wrapper is discarded."""
        if self._encoder_hook_handle is not None:
            self._encoder_hook_handle.remove()
            self._encoder_hook_handle = None
