"""
Resolution-specific forecast heads for LACE.

Each head implements:

    y_hat_r = W_r z_r + b_r

where:
    z_r     : adapted representation [B, d_adapter]
    W_r     : resolution-specific forecast mapping
    y_hat_r : forecast [B, L_r]

Together with ResolutionAdapter P_r:

    M_r = W_r P_r

which is the complete linear mapping from the shared TSFM
representation to the forecast at resolution r.
"""

import torch
from torch import nn


class ForecastHead(nn.Module):
    """
    Linear resolution-specific forecast head W_r.

    Parameters
    ----------
    adapter_dim : int
        Dimension of the adapted representation.

    output_len : int
        Number of forecast steps at this resolution.
    """

    def __init__(
        self,
        adapter_dim: int,
        output_len: int,
    ):
        super().__init__()

        self.adapter_dim = int(adapter_dim)
        self.output_len = int(output_len)

        if self.adapter_dim <= 0:
            raise ValueError(
                f"adapter_dim must be positive, got {self.adapter_dim}."
            )

        if self.output_len <= 0:
            raise ValueError(
                f"output_len must be positive, got {self.output_len}."
            )

        # ----------------------------------------------------------
        # W_r z_r + b_r
        # ----------------------------------------------------------
        self.proj = nn.Linear(
            self.adapter_dim,
            self.output_len,
            bias=True,
        )

    def forward(self, z_r):
        """
        Parameters
        ----------
        z_r : Tensor [B, adapter_dim]
            Resolution-adapted representation.

        Returns
        -------
        y_hat : Tensor [B, output_len]
            Forecast at this resolution.
        """

        if z_r.ndim != 2:
            raise ValueError(
                f"Expected z_r [B,d_adapter], got {z_r.shape}."
            )

        if z_r.shape[-1] != self.adapter_dim:
            raise ValueError(
                f"Expected adapter dimension {self.adapter_dim}, "
                f"got {z_r.shape[-1]}."
            )

        return self.proj(z_r)

    @property
    def weight(self):
        """
        Return W_r.

        Shape
        -----
        [output_len, adapter_dim]
        """
        return self.proj.weight

    @property
    def bias(self):
        """
        Return b_r.

        Shape
        -----
        [output_len]
        """
        return self.proj.bias
