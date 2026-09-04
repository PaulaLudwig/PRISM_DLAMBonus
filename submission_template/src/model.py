from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):

    def __init__(self, n_dynamic, n_static, n_series, hidden_size, num_layers, embedding_dim, static_dim, dropout):
        """build encoder, decoder, static pathway, and output head."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.series_embedding = nn.Embedding(n_series, embedding_dim)
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static + embedding_dim, static_dim),
            nn.ReLU(),
            nn.Linear(static_dim, static_dim),
        )

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=n_dynamic + 1,  # covariates plus the normalized target
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=n_dynamic + static_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, past_dynamic, past_target, future_dynamic, static, series_index):

        _, state = self.encoder(torch.cat([past_dynamic, past_target], dim=-1))

        static_vector = self.static_encoder(
            torch.cat([static, self.series_embedding(series_index)], dim=-1)
        )
        horizon = future_dynamic.shape[1]
        decoder_input = torch.cat(
            [future_dynamic, static_vector.unsqueeze(1).expand(-1, horizon, -1)], dim=-1
        )

        decoded, _ = self.decoder(decoder_input, state)
        return self.head(decoded).squeeze(-1)


def denormalize(predictions, location, scale):
    return predictions * scale.unsqueeze(1) + location.unsqueeze(1)
