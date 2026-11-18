"""
THUML registry bridge for LACE + Moirai-1.0.
"""

from lace.lace import LACE
from lace.lace_config import LACEConfig


class Model(LACE):
    def __init__(self, configs):
        cfg = LACEConfig.from_namespace(configs)

        super().__init__(
            context_len=cfg.context_len,
            pred_len=cfg.pred_len,
            factors=list(cfg.factors),
            adapter_dim=cfg.adapter_dim,
            representation_norm=cfg.representation_norm,
            reduction=cfg.reduction,
            backbone_name="Salesforce/moirai-1.0-R-small",
        )
