import torch
from torch import nn

from chronos.chronos2 import Chronos2Model


class FrozenChronos2Backbone(nn.Module):
    """
    Frozen Chronos-2 backbone for LACE.

    Input
    -----
    x_enc : [B, context_len, 1]

    Output
    ------
    z0 : [B, latent_dim]

    The representation concatenates:
        [mean(valid CONTEXT patch tokens) ; last valid CONTEXT patch token]

    The Chronos-2 [REG] token is retained during encoding but excluded
    from the LACE representation. Future/prediction patches are never introduced.
    """

    def __init__(
        self,
        context_len: int,
        pred_len: int,
        model_name: str = "amazon/chronos-2",
    ):
        super().__init__()

        self.context_len = int(context_len)
        self.pred_len = int(pred_len)

        # ----------------------------------------------------------
        # 1. Load and freeze Chronos-2
        # ----------------------------------------------------------
        model = Chronos2Model.from_pretrained(
            model_name,
        )

        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        # Native Chronos-2 encoder width.
        self.encoder_dim = int(model.model_dim)

        # LACE representation:
        #   [context mean ; last valid context patch]
        # Chronos-2: 768 + 768 = 1536.
        self.latent_dim = 2 * self.encoder_dim

        self.patch_size = int(
            model.chronos_config.input_patch_size
        )

        # Diagnostics for the most recent forward pass.
        self.last_context_token_count = None
        self.last_context_token_index = None
        self.last_num_context_tokens = None

    def train(self, mode: bool = True):
        """
        Parent LACE may enter train mode, but Chronos-2 remains frozen/eval.
        """
        super().train(mode)
        self.model.eval()
        return self

    def forward(
        self,
        x_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract a history-only Chronos-2 representation.
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

        # Chronos-2 expects [B,L].
        context = x_enc[..., 0].to(
            dtype=torch.float32
        )

        # Explicitly mark every observation as valid.
        context_mask = torch.ones(
            B,
            L,
            dtype=torch.float32,
            device=x_enc.device,
        )

        with torch.no_grad():

            # ------------------------------------------------------
            # 2. Official Chronos-2 context preprocessing
            #
            # Performs:
            #   - instance normalization
            #   - arcsinh transform when configured
            #   - patching
            #   - patch mask construction
            #   - temporal-position encoding
            #
            # Context tokens only.
            # ------------------------------------------------------
            (
                patched_context,
                attention_mask,
                _,
            ) = self.model._prepare_patched_context(
                context=context,
                context_mask=context_mask,
            )

            # ------------------------------------------------------
            # 3. Official input patch embedding
            # ------------------------------------------------------
            input_embeds = (
                self.model.input_patch_embedding(
                    patched_context
                )
            )

            num_context_tokens = input_embeds.shape[1]

            # ------------------------------------------------------
            # 4. Preserve Chronos-2's [REG] token during encoding
            # ------------------------------------------------------
            if self.model.chronos_config.use_reg_token:

                reg_input_ids = torch.full(
                    (B, 1),
                    self.model.config.reg_token_id,
                    dtype=torch.long,
                    device=x_enc.device,
                )

                reg_embeds = self.model.shared(
                    reg_input_ids
                )

                input_embeds = torch.cat(
                    [
                        input_embeds,
                        reg_embeds,
                    ],
                    dim=1,
                )

                attention_mask = torch.cat(
                    [
                        attention_mask.to(
                            self.model.dtype
                        ),
                        torch.ones(
                            B,
                            1,
                            dtype=self.model.dtype,
                            device=x_enc.device,
                        ),
                    ],
                    dim=1,
                )

            else:
                attention_mask = attention_mask.to(
                    self.model.dtype
                )

            # ------------------------------------------------------
            # 5. Each univariate series remains independent.
            #
            # Distinct group IDs prevent Chronos-2 group attention
            # from mixing samples within the batch.
            # ------------------------------------------------------
            group_ids = torch.arange(
                B,
                dtype=torch.long,
                device=x_enc.device,
            )

            # ------------------------------------------------------
            # 6. Context-only encoder pass
            #
            # No future patches are supplied.
            # ------------------------------------------------------
            encoder_outputs = self.model.encoder(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                group_ids=group_ids,
                output_attentions=False,
            )

            H = encoder_outputs.last_hidden_state

            if H is None:
                raise RuntimeError(
                    "Chronos-2 encoder returned no hidden state."
                )

            if H.ndim != 3:
                raise RuntimeError(
                    f"Expected latent [B,T,D], got {H.shape}."
                )

            if H.shape[0] != B:
                raise RuntimeError(
                    f"Batch mismatch: input={B}, "
                    f"latent={H.shape[0]}."
                )

            if H.shape[-1] != self.encoder_dim:
                raise RuntimeError(
                    f"Expected encoder dim "
                    f"{self.encoder_dim}, "
                    f"got {H.shape[-1]}."
                )

            # ------------------------------------------------------
            # 7. CONTEXT PATCH TOKENS ONLY.
            #
            # Exclude [REG] completely from the LACE representation.
            # The official Chronos attention mask tells us which
            # context patch positions are valid.
            # ------------------------------------------------------
            H_context = H[:, :num_context_tokens, :]

            valid_context_mask = (
                attention_mask[:, :num_context_tokens] > 0
            )

            context_counts = valid_context_mask.sum(dim=1)

            if torch.any(context_counts <= 0):
                raise RuntimeError(
                    "Chronos-2 produced no valid context patch tokens."
                )

            # A. Mean over VALID context patch tokens.
            weights = (
                valid_context_mask
                .to(H.dtype)
                .unsqueeze(-1)
            )

            z_mean = (
                (H_context * weights).sum(dim=1)
                /
                weights.sum(dim=1).clamp_min(1.0)
            )

            # B. Actual last VALID context patch token.
            token_idx = torch.arange(
                num_context_tokens,
                device=H.device,
            ).unsqueeze(0).expand(
                B,
                -1,
            )

            last_context_idx = (
                token_idx
                .masked_fill(~valid_context_mask, -1)
                .max(dim=1)
                .values
            )

            if torch.any(last_context_idx < 0):
                raise RuntimeError(
                    "Could not locate Chronos-2 last valid context patch."
                )

            batch_idx = torch.arange(
                B,
                device=H.device,
            )

            z_last = H_context[
                batch_idx,
                last_context_idx,
                :
            ]

            z0 = torch.cat(
                [
                    z_mean,
                    z_last,
                ],
                dim=-1,
            )

            self.last_context_token_count = context_counts.detach()
            self.last_context_token_index = last_context_idx.detach()
            self.last_num_context_tokens = int(num_context_tokens)

        if z0.shape != (
            B,
            self.latent_dim,
        ):
            raise RuntimeError(
                f"Unexpected Chronos-2 mean+last shape "
                f"{z0.shape}; expected "
                f"({B}, {self.latent_dim})."
            )

        return z0
