import torch
from torch import nn
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


import torch
from torch import nn

from uni2ts.model.moirai import (
    MoiraiForecast,
    MoiraiModule,
)


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name

        # Desired physical forecasting context
        self.context_len = configs.seq_len

        self.pred_len = configs.pred_len


        print(
            f">>> Moirai 1.0 | "
            f"context={self.context_len} | "
            f"horizon={self.pred_len} <<<"
        )

        backbone = MoiraiModule.from_pretrained(
            "Salesforce/moirai-1.0-R-small"
        )

        self.model = MoiraiForecast(
            module=backbone,
            prediction_length=self.pred_len,
            context_length=self.context_len,
            patch_size="auto",
            num_samples=64,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )


    def forecast(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3:
            raise ValueError(
                f"Expected [B,L,C], got {x_enc.shape}"
            )

        B, L, C = x_enc.shape

        if C != 1:
            raise ValueError(
                f"Expected univariate input, got C={C}"
            )

        if L != self.context_len:
            raise ValueError(
                f"Expected context length {self.context_len}, got {L}"
            )

        x = x_enc.to(torch.float32)

        # --------------------------------------------------
        # Moirai auto-patching requires:
        # context_length + prediction_length input positions.
        #
        # We prepend UNOBSERVED padding, not fake future data.
        # --------------------------------------------------
        H = self.pred_len

        prefix = torch.zeros(
            (B, H, C),
            dtype=x.dtype,
            device=x.device,
        )

        past_target = torch.cat(
            [prefix, x],
            dim=1,
        )

        # Prefix is explicitly unobserved
        prefix_observed = torch.zeros(
            (B, H, C),
            dtype=torch.bool,
            device=x.device,
        )

        real_observed = torch.ones(
            (B, L, C),
            dtype=torch.bool,
            device=x.device,
        )

        past_observed = torch.cat(
            [prefix_observed, real_observed],
            dim=1,
        )

        # Prefix is padding
        prefix_pad = torch.ones(
            (B, H),
            dtype=torch.bool,
            device=x.device,
        )

        real_pad = torch.zeros(
            (B, L),
            dtype=torch.bool,
            device=x.device,
        )

        past_is_pad = torch.cat(
            [prefix_pad, real_pad],
            dim=1,
        )

        # --------------------------------------------------
        # Moirai auto patch selection + forecasting
        # --------------------------------------------------
        samples = self.model(
            past_target=past_target,
            past_observed_target=past_observed,
            past_is_pad=past_is_pad,
        )

        # Handle possible target_dim=1 output
        if samples.ndim == 4 and samples.shape[-1] == 1:
            samples = samples.squeeze(-1)

        if samples.ndim != 3:
            raise RuntimeError(
                f"Unexpected Moirai output shape: {samples.shape}"
            )

        # [B, samples, H] -> [B, H]
        point_forecast = torch.quantile(
            samples,
            q=0.5,
            dim=1,
        )

        return point_forecast.unsqueeze(-1)

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
    ):

        if self.task_name in [
            "zero_shot_forecast",
            "forecast",
        ]:
            return self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
            )

        raise NotImplementedError(
            f"Unsupported task: {self.task_name}"
        )
