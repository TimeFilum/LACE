import numpy as np
import torch
from torch import nn
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module


class Model(nn.Module):
    """
    Clean Moirai-2 wrapper for:
    - zero-shot univariate forecasting
    - capturing Q/K projections
    - reconstructing attention maps
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name

        # Load pretrained backbone
        self.backbone = Moirai2Module.from_pretrained(
            "Salesforce/moirai-2.0-R-small"
        )

        # Forecast wrapper (univariate)
        self.model = Moirai2Forecast(
            module=self.backbone,
            prediction_length=configs.pred_len,
            context_length=configs.seq_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to("cuda")


        # # 🔍 Print all modules so we know where attention lives
        # for name, module in self.model.module.named_modules():
        #     print(name, type(module))

        # Storage for Q/K projections
        self.qk_store = {}
        self.last_attn = None

        # Attach hooks
        self._register_qk_hooks()


    # ---------------------------------------------------------
    # Register hooks on all attention layers
    # ---------------------------------------------------------
    def _register_qk_hooks(self):

        def make_hook(name):
            def hook(module, inputs, output):
                x = inputs[0]  # hidden states before projection
                q = module.q_proj(x).detach().cpu()
                k = module.k_proj(x).detach().cpu()
                self.qk_store[name] = (q, k)
                # print("hook input shape:", x.shape)
            return hook

        for name, module in self.model.module.named_modules():
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                module.register_forward_hook(make_hook(name))
                # print(f"Hook attached to: {name}")


    # ---------------------------------------------------------
    # Reconstruct attention from Q and K
    # ---------------------------------------------------------
    def _reconstruct_attention(self, q, k):
        d = q.shape[-1]
        attn = q @ k.transpose(-1, -2) / np.sqrt(d)
        return torch.softmax(attn, dim=-1)


    # ---------------------------------------------------------
    # Zero-shot forecast + attention extraction
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Zero-shot forecast (THUML-compatible signature)
    # ---------------------------------------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # Clear previous Q/K storage
        self.qk_store = {}

        # x_enc: [B, seq_len, 1]
        arr = x_enc.cpu().numpy()

        # Run Moirai2 zero-shot prediction
        pred = self.model.predict(arr)          # numpy output [B, pred_len, 1]

        # pred: [B, 9, pred_len]
        pred = pred[:, 4:5, :]              # [B, 1, pred_len]

        # Move pred_len to axis 1 → [B, pred_len, 1]
        pred = np.transpose(pred, (0, 2, 1))

        pred = torch.tensor(pred).to(x_enc.device)

        # Reconstruct attention maps from stored Q/K
        attn_maps = {}
        for name, (q, k) in self.qk_store.items():
            attn_maps[name] = self._reconstruct_attention(q, k)

        self.last_attn = attn_maps
        return pred, attn_maps

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "zero_shot_forecast":
            pred, attn = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            self.last_attn = attn          # store attention maps internally
            return pred                    # <-- return ONLY prediction

        else:
            pred, attn = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            self.last_attn = attn
            return pred
