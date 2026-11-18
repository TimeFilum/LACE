import torch
from torch import nn
import timesfm


import numpy as np
import torch
from torch import nn
import timesfm


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        print(">>> TimesFM 2.0 init called <<<")

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # -------------------------------------------------
        # TimesFM 2.0 supports context up to 2048
        # -------------------------------------------------
        if self.seq_len > 2048:
            raise ValueError(
                f"TimesFM 2.0 supports context_len <= 2048, "
                f"but seq_len={self.seq_len}"
            )

        # -------------------------------------------------
        # Backend
        # -------------------------------------------------
        use_gpu = (
            getattr(configs, "use_gpu", False)
            and torch.cuda.is_available()
        )

        backend = "gpu" if use_gpu else "cpu"

        print(
            f"TimesFM backend={backend}, "
            f"context={self.seq_len}, "
            f"horizon={self.pred_len}"
        )

        # -------------------------------------------------
        # Official TimesFM 2.0 checkpoint
        # -------------------------------------------------
        repo_id = "google/timesfm-2.0-500m-pytorch"

        hparams = timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=32,

            context_len=2048,
            horizon_len=self.pred_len,

            # Fixed parameters for TimesFM 2.0 500M
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=False,
        )

        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id=repo_id
        )

        self.tfm = timesfm.TimesFm(
            hparams=hparams,
            checkpoint=checkpoint
        )

    # =====================================================
    # FORECAST
    # =====================================================
    def forecast(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None
    ):
        """
        THUML:
            input  -> [B, L, 1]

        TimesFM:
            input  -> list of B arrays, each [L]
            output -> [B, H]

        THUML:
            output -> [B, H, 1]
        """

        if x_enc.ndim != 3:
            raise ValueError(
                f"Expected x_enc [B,L,C], got {x_enc.shape}"
            )

        B, L, C = x_enc.shape

        # Phase-0 experiments are univariate
        if C != 1:
            raise ValueError(
                f"TimesFM Phase-0 wrapper expects C=1, got C={C}"
            )

        if L != self.seq_len:
            raise ValueError(
                f"Expected context length {self.seq_len}, "
                f"got {L}"
            )

        # -------------------------------------------------
        # [B,L,1] -> [B,L]
        # TimesFM API expects CPU numpy arrays
        # -------------------------------------------------
        series = (
            x_enc[:, :, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        inputs = [
            series[b]
            for b in range(B)
        ]

        # TimesFM frequency category:
        # 0 = minute/hour/day
        freq = [0] * B

        # -------------------------------------------------
        # Zero-shot forecast
        # -------------------------------------------------
        point_forecast, _ = self.tfm.forecast(
            inputs=inputs,
            freq=freq,
        )

        point_forecast = np.asarray(
            point_forecast,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Shape validation
        # -------------------------------------------------
        if point_forecast.ndim != 2:
            raise RuntimeError(
                "Unexpected TimesFM output shape: "
                f"{point_forecast.shape}"
            )

        if point_forecast.shape[0] != B:
            raise RuntimeError(
                f"Expected batch size {B}, "
                f"got {point_forecast.shape[0]}"
            )

        if point_forecast.shape[1] < self.pred_len:
            raise RuntimeError(
                f"TimesFM returned only "
                f"{point_forecast.shape[1]} steps, "
                f"expected {self.pred_len}"
            )

        # Defensive slice
        point_forecast = point_forecast[
            :, :self.pred_len
        ]

        # -------------------------------------------------
        # [B,H] -> [B,H,1]
        # -------------------------------------------------
        dec_out = (
            torch.from_numpy(point_forecast)
            .unsqueeze(-1)
            .to(
                device=x_enc.device,
                dtype=x_enc.dtype
            )
        )

        return dec_out

    # =====================================================
    # THUML INTERFACE
    # =====================================================
    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None
    ):
        if self.task_name in [
            "zero_shot_forecast",
            "forecast"
        ]:
            return self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec
            )

        raise NotImplementedError(
            f"TimesFM wrapper does not support "
            f"task '{self.task_name}'"
        )
