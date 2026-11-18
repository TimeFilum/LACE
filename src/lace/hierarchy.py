"""
Dynamic temporal hierarchy operators for LACE.

Responsibility
--------------
Define the temporal relationships between forecast resolutions and
apply the corresponding aggregation operator B_{r->s}.

The same hierarchy operator can act on:

    predictions:     B y_r
    mapping matrix:  B M_r
    bias vector:     B b_r

No model parameters are defined here.

Example
-------
For factors [1, 3, 12, 288]:

    5m  -> 15m : q = 3
    15m -> 1h  : q = 4
    1h  -> 1d  : q = 24

where each factor expresses the number of finest-resolution steps
contained in that resolution.
"""

from typing import List, Tuple

import torch


class TemporalHierarchy:
    """
    Dynamic hierarchy defined relative to the finest resolution.

    Parameters
    ----------
    factors : list[int]
        Aggregation factors relative to the finest resolution.

        Example:
            [1, 3, 12, 288]

    pred_len : int
        Forecast horizon at the finest resolution.

    reduction : str
        Aggregation semantics:

            "sum"  -> coarse value is sum of fine values
            "mean" -> coarse value is mean of fine values

        LACE currently uses "sum" for additive traffic/load quantities.
    """

    def __init__(
        self,
        factors: List[int],
        pred_len: int,
        reduction: str = "sum",
    ):
        self.pred_len = int(pred_len)
        self.factors = sorted(int(f) for f in factors)
        self.reduction = reduction

        self._validate()

        # Number of forecast points at every resolution.
        #
        # This is equivalent to the previous:
        #
        #   (pred_len - f) // f + 1
        #
        # for positive integer factors, but is clearer.
        self.target_lengths = [
            self.pred_len // f
            for f in self.factors
        ]

    # ==============================================================
    # Validation
    # ==============================================================

    def _validate(self):
        if self.pred_len <= 0:
            raise ValueError(
                f"pred_len must be positive, got {self.pred_len}."
            )

        if len(self.factors) == 0:
            raise ValueError(
                "Hierarchy must contain at least one factor."
            )

        if any(f <= 0 for f in self.factors):
            raise ValueError(
                f"All factors must be positive, got {self.factors}."
            )

        if len(set(self.factors)) != len(self.factors):
            raise ValueError(
                f"Hierarchy contains duplicate factors: {self.factors}."
            )

        if self.reduction not in {"sum", "mean"}:
            raise ValueError(
                "reduction must be 'sum' or 'mean', "
                f"got {self.reduction!r}."
            )

        # Adjacent scales must form an exact aggregation hierarchy.
        for fine, coarse in zip(
            self.factors[:-1],
            self.factors[1:],
        ):
            if coarse % fine != 0:
                raise ValueError(
                    "Adjacent hierarchy factors must be integer "
                    f"multiples, got {fine} -> {coarse}."
                )

    # ==============================================================
    # Hierarchy metadata
    # ==============================================================

    @property
    def num_scales(self) -> int:
        return len(self.factors)

    @property
    def adjacent_pairs(self) -> List[Tuple[int, int]]:
        """
        Return adjacent hierarchy factors.

        Example
        -------
        [1, 3, 12, 288]
            ->
        [(1, 3), (3, 12), (12, 288)]
        """
        return list(
            zip(
                self.factors[:-1],
                self.factors[1:],
            )
        )

    def factor_ratio(
        self,
        fine_factor: int,
        coarse_factor: int,
    ) -> int:
        """
        Return q = f_s / f_r.
        """

        fine_factor = int(fine_factor)
        coarse_factor = int(coarse_factor)

        if fine_factor not in self.factors:
            raise ValueError(
                f"Unknown fine factor {fine_factor}."
            )

        if coarse_factor not in self.factors:
            raise ValueError(
                f"Unknown coarse factor {coarse_factor}."
            )

        if coarse_factor <= fine_factor:
            raise ValueError(
                "coarse_factor must be larger than fine_factor, "
                f"got {fine_factor} -> {coarse_factor}."
            )

        if coarse_factor % fine_factor != 0:
            raise ValueError(
                f"{coarse_factor} is not an integer multiple "
                f"of {fine_factor}."
            )

        return coarse_factor // fine_factor

    def target_length(self, factor: int) -> int:
        """
        Forecast length for one hierarchy factor.
        """

        factor = int(factor)

        if factor not in self.factors:
            raise ValueError(
                f"Unknown hierarchy factor {factor}."
            )

        return self.pred_len // factor

    # ==============================================================
    # Core aggregation
    # ==============================================================

    def _aggregate_rows(
        self,
        x: torch.Tensor,
        q: int,
    ) -> torch.Tensor:
        """
        Aggregate consecutive rows/elements in groups of q.

        Supported inputs
        ----------------
        [L]
        [L, D]
        [B, L]

        The temporal/output dimension is:
            dim=0 for [L] and [L,D]
            dim=1 for [B,L]

        This private function is used by the public prediction,
        operator, and bias aggregation methods.
        """

        if q <= 0:
            raise ValueError(
                f"Aggregation ratio must be positive, got {q}."
            )

        # ----------------------------------------------------------
        # Bias vector: [L]
        # ----------------------------------------------------------
        if x.ndim == 1:
            L = x.shape[0]

            usable = (L // q) * q
            x = x[:usable]

            x = x.reshape(-1, q)

            if self.reduction == "sum":
                return x.sum(dim=1)

            return x.mean(dim=1)

        # ----------------------------------------------------------
        # Mapping matrix: [L, D]
        # ----------------------------------------------------------
        if x.ndim == 2:
            L, D = x.shape

            usable = (L // q) * q
            x = x[:usable, :]

            x = x.reshape(
                usable // q,
                q,
                D,
            )

            if self.reduction == "sum":
                return x.sum(dim=1)

            return x.mean(dim=1)

        raise ValueError(
            "_aggregate_rows expects [L] or [L,D], "
            f"got {x.shape}."
        )

    # ==============================================================
    # B applied to predictions
    # ==============================================================

    def aggregate_prediction(
        self,
        y_fine: torch.Tensor,
        fine_factor: int,
        coarse_factor: int,
    ) -> torch.Tensor:
        """
        Apply B_{r->s} to predictions.

        Parameters
        ----------
        y_fine : Tensor [B, L_r]

        Returns
        -------
        Tensor [B, L_s]
        """

        if y_fine.ndim != 2:
            raise ValueError(
                "Expected prediction [B,L], "
                f"got {y_fine.shape}."
            )

        q = self.factor_ratio(
            fine_factor,
            coarse_factor,
        )

        B, L = y_fine.shape

        usable = (L // q) * q

        y = y_fine[:, :usable]

        y = y.reshape(
            B,
            usable // q,
            q,
        )

        if self.reduction == "sum":
            return y.sum(dim=-1)

        return y.mean(dim=-1)

    # ==============================================================
    # B applied to M_r = W_r P_r
    # ==============================================================

    def aggregate_mapping(
        self,
        M_fine: torch.Tensor,
        fine_factor: int,
        coarse_factor: int,
    ) -> torch.Tensor:
        """
        Apply B_{r->s} to a resolution mapping matrix.

        Parameters
        ----------
        M_fine : Tensor [L_r, D]

        Returns
        -------
        Tensor [L_s, D]
        """

        if M_fine.ndim != 2:
            raise ValueError(
                "Expected mapping matrix [L,D], "
                f"got {M_fine.shape}."
            )

        q = self.factor_ratio(
            fine_factor,
            coarse_factor,
        )

        return self._aggregate_rows(
            M_fine,
            q,
        )

    # ==============================================================
    # B applied to b_r
    # ==============================================================

    def aggregate_bias(
        self,
        b_fine: torch.Tensor,
        fine_factor: int,
        coarse_factor: int,
    ) -> torch.Tensor:
        """
        Apply B_{r->s} to a forecast-head bias vector.

        Parameters
        ----------
        b_fine : Tensor [L_r]

        Returns
        -------
        Tensor [L_s]
        """

        if b_fine.ndim != 1:
            raise ValueError(
                "Expected bias [L], "
                f"got {b_fine.shape}."
            )

        q = self.factor_ratio(
            fine_factor,
            coarse_factor,
        )

        return self._aggregate_rows(
            b_fine,
            q,
        )
