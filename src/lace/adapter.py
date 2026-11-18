"""
Resolution-specific linear adapters for LACE.

Each adapter implements:

    z_r = P_r z

where:
    z   : shared pooled TSFM representation [B, D]
    P_r : resolution-specific linear projection
    z_r : adapted representation [B, d_adapter]

Adapters are deliberately linear and bias-free so that the complete
resolution mapping

    M_r = W_r P_r

can be constructed explicitly for structural regularization.
"""

import torch
from torch import nn


class ResolutionAdapter(nn.Module):
    """
    Linear resolution-specific adapter P_r.

    Parameters
    ----------
    input_dim : int
        Dimension D of the shared TSFM representation.

    adapter_dim : int
        Dimension d_a of the adapted representation.
    """

    def __init__(
        self,
        input_dim: int,
        adapter_dim: int,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.adapter_dim = int(adapter_dim)

        # P_r
        #
        # No bias:
        # this keeps the composed mapping W_r P_r mathematically clean.
        self.proj = nn.Linear(
            self.input_dim,
            self.adapter_dim,
            bias=False,
        )

    def forward(self, z):
        """
        Parameters
        ----------
        z : Tensor [B, input_dim]

        Returns
        -------
        Tensor [B, adapter_dim]
        """

        if z.ndim != 2:
            raise ValueError(
                f"Expected z [B,D], got {z.shape}."
            )

        if z.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {z.shape[-1]}."
            )

        return self.proj(z)

    @property
    def weight(self):
        """
        Return P_r directly.

        Shape:
            [adapter_dim, input_dim]
        """
        return self.proj.weight
