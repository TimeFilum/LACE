import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out

        # Use LSTM-specific parameters
        self.hidden_size = configs.lstm_hidden
        self.num_layers = configs.lstm_layers

        # Embedding layer
        self.enc_embedding = DataEmbedding(configs.enc_in, self.hidden_size, configs.embed, configs.freq, configs.dropout)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Output projection
        self.projection = nn.Linear(self.hidden_size, configs.c_out)

        # Classification head
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.cls_head = nn.Linear(self.hidden_size * self.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, seq_len, hidden]
        lstm_out, _ = self.lstm(enc_out)                 # [B, seq_len, hidden]
        output = self.projection(lstm_out)               # [B, seq_len, c_out]
        return output[:, -self.pred_len:, :]             # [B, pred_len, c_out]


    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        lstm_out, _ = self.lstm(enc_out)
        out = self.projection(lstm_out)  # [B, seq_len, c_out]
        return out

    def anomaly_detection(self, x_enc):
        dummy_mark = torch.zeros_like(x_enc)
        enc_out = self.enc_embedding(x_enc, dummy_mark)
        lstm_out, _ = self.lstm(enc_out)
        out = self.projection(lstm_out)
        return out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        lstm_out, _ = self.lstm(enc_out)
        out = self.act(lstm_out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)  # Flatten [B, seq_len * hidden]
        out = self.cls_head(out)             # [B, num_class]
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'hyperparameter_tuning']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
