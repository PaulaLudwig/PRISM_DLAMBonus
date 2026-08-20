from __future__ import annotations

import torch
from torch import nn

N_COVARIATES = 18
FIRST_MASKED_COVARIATE = 8


class Chomp1d(nn.Module):

    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    
    def __init__(self, channels, kernel_size, dilation, dropout):

        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.res_scale * self.net(x)


class PatchTCNEncoder(nn.Module):

    def __init__(self, d_model, patch_len, kernel_size = 3, dilations = (1, 2, 4), dropout = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.patch_embed = nn.Linear(patch_len * 2, d_model)
        self.tcn = nn.Sequential(
            *[TemporalBlock(d_model, kernel_size, d, dropout) for d in dilations]
        )

    def forward(self, series):
        n, length, _ = series.shape
        n_patches = length // self.patch_len
        patches = series.reshape(n, n_patches, self.patch_len * 2)
        tokens = self.patch_embed(patches) 
        encoded = self.tcn(tokens.transpose(1, 2)) 
        return encoded.mean(dim=2)


class PRISMForecaster(nn.Module):
   

    def __init__(
        self,
        n_static, n_series, history = 168, horizon = 336, d_model = 128, patch_len = 24, n_heads = 8, n_attn_layers = 2, embedding_dim = 16, dropout = 0.2, mode = "full"):

        super().__init__()
        if mode not in {"full", "stage1", "stage2"}:
            raise ValueError(f"mode must be full|stage1|stage2, got {mode!r}.")
        if mode != "stage2" and (history % patch_len or (history + horizon) % patch_len):
            raise ValueError(
                f"patch_len={patch_len} must divide both history={history} and "
                f"history+horizon={history + horizon}."
            )
        self.mode = mode
        self.history = history
        self.horizon = horizon
        self.d_model = d_model
        self.n_tokens = 1 + N_COVARIATES + 1  # target + covariates + static

        self.series_embedding = nn.Embedding(n_series, embedding_dim)
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static + embedding_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        if mode == "stage2":
            self.target_embed = nn.Linear(history * 2, d_model)
            self.covariate_embed = nn.Linear((history + horizon) * 2, d_model)
        else:
            self.encoder = PatchTCNEncoder(d_model, patch_len, dropout=dropout)

        if mode == "stage1":
            self.channel_mixer = nn.Sequential(
                nn.Linear(self.n_tokens * d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.cross_variate = nn.TransformerEncoder(layer, num_layers=n_attn_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def _split_channels(self, past_dynamic, past_target, future_dynamic):
        batch = past_dynamic.shape[0]

        past_values = past_dynamic[..., :N_COVARIATES]
        future_values = future_dynamic[..., :N_COVARIATES]
        covariate_values = torch.cat([past_values, future_values], dim=1) 

        past_mask = torch.ones_like(past_values)
        future_mask = torch.ones_like(future_values)
        past_mask[..., FIRST_MASKED_COVARIATE:] = past_dynamic[..., N_COVARIATES:]
        future_mask[..., FIRST_MASKED_COVARIATE:] = future_dynamic[..., N_COVARIATES:]
        covariate_mask = torch.cat([past_mask, future_mask], dim=1) 

        covariate_series = torch.stack([covariate_values, covariate_mask], dim=-1)
        covariate_series = covariate_series.permute(0, 2, 1, 3).contiguous() 

        target_series = torch.cat(
            [past_target, torch.ones_like(past_target)], dim=-1
        ) 
        return target_series, covariate_series

    def forward(self,past_dynamic, past_target, future_dynamic, static, series_index):

        batch = past_dynamic.shape[0]
        target_series, covariate_series = self._split_channels(
            past_dynamic, past_target, future_dynamic
        )
        length = covariate_series.shape[2]

        static_token = self.static_encoder(
            torch.cat([static, self.series_embedding(series_index)], dim=-1)
        )
        if self.mode == "stage2":
            target_token = self.target_embed(target_series.reshape(batch, -1))
            covariate_tokens = self.covariate_embed(
                covariate_series.reshape(batch, N_COVARIATES, -1)
            )
        else:
            target_token = self.encoder(target_series)  # (B, d_model)
            covariate_tokens = self.encoder(
                covariate_series.reshape(batch * N_COVARIATES, length, 2)
            ).reshape(batch, N_COVARIATES, self.d_model)

       
        tokens = torch.cat(
            [target_token.unsqueeze(1), covariate_tokens, static_token.unsqueeze(1)], dim=1
        ) 

        if self.mode == "stage1":
            merged = self.channel_mixer(tokens.reshape(batch, -1))
            return self.head(merged)

        enriched = self.cross_variate(tokens) 
        return self.head(enriched[:, 0])  
