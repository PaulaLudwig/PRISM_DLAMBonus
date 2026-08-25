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

    def __init__(
        self,
        d_model,
        patch_len,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()

        if pooling not in {"mean", "attention", "patches"}:
            raise ValueError(
                f"pooling must be mean|attention|patches, got {pooling!r}"
            )

        self.patch_len = patch_len
        self.pooling = pooling

        self.patch_embed = nn.Linear(patch_len * 2, d_model)

        self.tcn = nn.Sequential(
            *[
                TemporalBlock(d_model, kernel_size, d, dropout)
                for d in dilations
            ]
        )

        if pooling == "attention":
            self.pool_score = nn.Linear(d_model, 1, bias=False)

            # Start with uniform attention, i.e. exactly like mean pooling.
            nn.init.zeros_(self.pool_score.weight)

    def forward(self, series):
        n, length, _ = series.shape

        n_patches = length // self.patch_len

        patches = series.reshape(
            n,
            n_patches,
            self.patch_len * 2
        )

        tokens = self.patch_embed(patches)

        # (N, patches, d_model)
        # -> (N, d_model, patches)
        encoded = self.tcn(tokens.transpose(1, 2))

        if self.pooling == "mean":
            return encoded.mean(dim=2)

        # Attention and patch-preserving modes use:
        # (N, n_patches, d_model)
        encoded = encoded.transpose(1, 2)

        if self.pooling == "attention":
            scores = self.pool_score(encoded).squeeze(-1)
            weights = torch.softmax(scores, dim=1)

            return torch.sum(
                encoded * weights.unsqueeze(-1),
                dim=1,
            )

        # pooling == "patches"
        # Do not collapse the temporal patch dimension.
        return encoded


class PRISMForecaster(nn.Module):
   

    def __init__(
        self,
        n_static, n_series, history = 168, horizon = 336, d_model = 128, patch_len = 24, n_heads = 8, n_attn_layers = 2, embedding_dim = 16, dropout = 0.2, mode = "full",pooling="mean"):

        super().__init__()

        if mode not in {"full", "stage1", "stage2"}:
            raise ValueError(
                f"mode must be full|stage1|stage2, got {mode!r}."
            )

        if mode != "stage2" and (
                history % patch_len
                or (history + horizon) % patch_len
        ):
            raise ValueError(
                f"patch_len={patch_len} must divide both history={history} and "
                f"history+horizon={history + horizon}."
            )

        if pooling not in {"mean", "attention", "patches"}:
            raise ValueError(
                f"pooling must be mean|attention|patches, got {pooling!r}."
            )

        if pooling == "patches" and mode != "full":
            raise ValueError(
                "pooling='patches' is only supported with mode='full'."
            )

        self.mode = mode
        self.pooling = pooling
        self.history = history
        self.horizon = horizon
        self.d_model = d_model
        self.patch_len = patch_len
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
            self.encoder = PatchTCNEncoder(
                d_model,
                patch_len,
                dropout=dropout,
                pooling=pooling,
            )

        if mode == "full" and pooling == "patches":
            self.max_patches = (history + horizon) // patch_len

            # 0 = target, 1...18 = covariates
            self.variable_embedding = nn.Embedding(
                1 + N_COVARIATES,
                d_model,
            )

            # Tells the Transformer whether a token came from
            # temporal patch 0, 1, ..., 20.
            self.patch_position_embedding = nn.Embedding(
                self.max_patches,
                d_model,
            )

            # A dedicated token that will collect information
            # from all target/covariate patches for forecasting.
            self.forecast_token = nn.Parameter(
                torch.zeros(1, 1, d_model)
            )

            nn.init.normal_(
                self.forecast_token,
                mean=0.0,
                std=0.02,
            )
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
            # Stage-2-only baseline:
            # embed the complete trajectory directly into one token.
            target_token = self.target_embed(
                target_series.reshape(batch, -1)
            )

            covariate_tokens = self.covariate_embed(
                covariate_series.reshape(
                    batch,
                    N_COVARIATES,
                    -1,
                )
            )

        elif self.pooling == "patches":
            # =========================================================
            # PATCH-PRESERVING BRIDGE
            # =========================================================

            # Target history:
            # (B, 168, 2)
            # -> encoder
            # -> (B, 7, d_model)
            target_patches = self.encoder(target_series)

            # Covariates:
            # start as (B, 18, 504, 2)
            #
            # Merge B and variable dimensions because the same
            # channel-independent TCN processes every covariate:
            #
            # (B*18, 504, 2)
            # -> encoder
            # -> (B*18, 21, d_model)
            covariate_patches = self.encoder(
                covariate_series.reshape(
                    batch * N_COVARIATES,
                    length,
                    2,
                )
            )

            n_cov_patches = covariate_patches.shape[1]

            # Restore the variable dimension:
            #
            # (B*18, 21, d_model)
            # ->
            # (B, 18, 21, d_model)
            covariate_patches = covariate_patches.reshape(
                batch,
                N_COVARIATES,
                n_cov_patches,
                self.d_model,
            )

            # =========================================================
            # ADD IDENTITY TO TARGET PATCHES
            # =========================================================

            n_target_patches = target_patches.shape[1]

            # Variable ID 0 is the target.
            target_var_ids = torch.zeros(
                n_target_patches,
                dtype=torch.long,
                device=target_patches.device,
            )

            # Temporal positions 0...6.
            target_pos_ids = torch.arange(
                n_target_patches,
                device=target_patches.device,
            )

            target_patches = (
                target_patches
                + self.variable_embedding(
                    target_var_ids
                ).unsqueeze(0)
                + self.patch_position_embedding(
                    target_pos_ids
                ).unsqueeze(0)
            )

            # =========================================================
            # ADD IDENTITY TO COVARIATE PATCHES
            # =========================================================

            # IDs 1...18 identify the covariates.
            covariate_var_ids = torch.arange(
                1,
                N_COVARIATES + 1,
                device=covariate_patches.device,
            )

            # Temporal positions 0...20.
            covariate_pos_ids = torch.arange(
                n_cov_patches,
                device=covariate_patches.device,
            )

            covariate_patches = (
                covariate_patches
                + self.variable_embedding(
                    covariate_var_ids
                )[None, :, None, :]
                + self.patch_position_embedding(
                    covariate_pos_ids
                )[None, None, :, :]
            )

            # Flatten variable × patch into one Transformer token axis:
            #
            # (B, 18, 21, d_model)
            # ->
            # (B, 378, d_model)
            covariate_patches = covariate_patches.reshape(
                batch,
                N_COVARIATES * n_cov_patches,
                self.d_model,
            )

            # =========================================================
            # BUILD STAGE-2 TOKEN SEQUENCE
            # =========================================================

            forecast_token = self.forecast_token.expand(
                batch,
                -1,
                -1,
            )

            tokens = torch.cat(
                [
                    forecast_token,              # 1
                    target_patches,              # 7
                    covariate_patches,           # 378
                    static_token.unsqueeze(1),   # 1
                ],
                dim=1,
            )

            # Total:
            # 1 + 7 + 378 + 1 = 387 tokens
            enriched = self.cross_variate(tokens)

            # Forecast token is at index 0.
            return self.head(enriched[:, 0])

        else:
            # =========================================================
            # EXISTING MEAN / ATTENTION POOLING PATH
            # =========================================================

            target_token = self.encoder(
                target_series
            )

            covariate_tokens = self.encoder(
                covariate_series.reshape(
                    batch * N_COVARIATES,
                    length,
                    2,
                )
            ).reshape(
                batch,
                N_COVARIATES,
                self.d_model,
            )

        # =============================================================
        # EXISTING SINGLE-TOKEN PATH
        # Used by stage2, mean pooling and attention pooling.
        # The patches branch above has already returned.
        # =============================================================

        tokens = torch.cat(
            [
                target_token.unsqueeze(1),
                covariate_tokens,
                static_token.unsqueeze(1),
            ],
            dim=1,
        )

        if self.mode == "stage1":
            merged = self.channel_mixer(
                tokens.reshape(batch, -1)
            )
            return self.head(merged)

        enriched = self.cross_variate(tokens)

        return self.head(enriched[:, 0])
