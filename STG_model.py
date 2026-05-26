import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TimeTokenEncoder(nn.Module):
    """Token-level time encoder: R^2 -> R^D."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        return self.net(time_features)


class LowRankActivityBias(nn.Module):
    """Low-rank activity-pair bias. Intended as optional enhancement only."""

    def __init__(self, vocab_size: int, num_heads: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.num_heads = num_heads
        self.activity_factor = nn.Embedding(vocab_size, rank)
        self.head_scale = nn.Parameter(torch.ones(num_heads))
        nn.init.xavier_uniform_(self.activity_factor.weight)

    def forward(self, act_seq: torch.Tensor) -> torch.Tensor:
        # act_seq: [B, T]
        p = self.activity_factor(act_seq)  # [B, T, R]
        b_act = torch.matmul(p, p.transpose(1, 2)) / math.sqrt(self.rank)  # [B, T, T]
        return b_act.unsqueeze(1) * self.head_scale.view(1, self.num_heads, 1, 1)


class MultiModalFusionLayer(nn.Module):
    """
    Stage-2 fusion:
    1) Act <- Res cross attention
    2) Res <- Act cross attention
    3) Token-wise channel-wise dynamic gates
    4) Concatenate and project to trunk representation
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, use_dynamic_gating: bool = True):
        super().__init__()
        self.use_dynamic_gating = use_dynamic_gating
        self.cross_act_res = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_res_act = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.gate_act = nn.Linear(2 * d_model, d_model)
        self.gate_res = nn.Linear(2 * d_model, d_model)

        self.norm_act = nn.LayerNorm(d_model)
        self.norm_res = nn.LayerNorm(d_model)

        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        self.norm_fusion = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_act0: torch.Tensor,
        x_res0: torch.Tensor,
        time_tok: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # key_padding_mask in MHA: True means padding.
        key_padding_mask = (padding_mask == 0) if padding_mask is not None else None

        if self.use_dynamic_gating:
            act_update, _ = self.cross_act_res(
                query=x_act0,
                key=x_res0,
                value=x_res0,
                key_padding_mask=key_padding_mask,
            )
            res_update, _ = self.cross_res_act(
                query=x_res0,
                key=x_act0,
                value=x_act0,
                key_padding_mask=key_padding_mask,
            )

            g_act = torch.sigmoid(self.gate_act(torch.cat([x_act0, act_update], dim=-1)))
            g_res = torch.sigmoid(self.gate_res(torch.cat([x_res0, res_update], dim=-1)))

            x_act1 = self.norm_act(x_act0 + g_act * self.dropout(act_update))
            x_res1 = self.norm_res(x_res0 + g_res * self.dropout(res_update))
        else:
            x_act1 = x_act0
            x_res1 = x_res0

        x_fused = self.fusion_proj(torch.cat([x_act1, x_res1, time_tok], dim=-1))
        return self.norm_fusion(x_fused)


class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        vocab_size: int,
        use_time_bias: bool = True,
        use_act_lowrank_bias: bool = False,
        use_time_value_gate: bool = True,
        act_bias_rank: int = 16,
        time_tau0: float = 1.0,
        time_eps: float = 1e-6,
        time_lambda_init: float = 0.8,
        act_gamma_init: float = -3.0,
        pair_count_threshold: int = 20,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.use_time_bias = use_time_bias
        self.use_act_lowrank_bias = use_act_lowrank_bias
        self.use_time_value_gate = use_time_value_gate

        self.time_tau0 = float(max(time_tau0, time_eps))
        self.time_eps = time_eps
        self.pair_count_threshold = int(pair_count_threshold)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Time bias projection: b_time(h, i, j) = w_h * u_hat(i,j) + c_h
        self.time_w = nn.Parameter(torch.ones(num_heads))
        self.time_c = nn.Parameter(torch.zeros(num_heads))
        self.time_lambda = nn.Parameter(torch.full((num_heads,), float(time_lambda_init)))

        # Optional activity low-rank bias with learnable on/off gate gamma in [0,1]
        if self.use_act_lowrank_bias:
            self.act_bias = LowRankActivityBias(vocab_size, num_heads, rank=act_bias_rank)
            self.theta_act_bias = nn.Parameter(torch.tensor(float(act_gamma_init)))

        # Time value gate: g_time(h, i)
        self.eta_time = nn.Parameter(torch.ones(num_heads))
        self.kappa_time = nn.Parameter(torch.zeros(num_heads))

        self.register_buffer("pair_count_matrix", torch.empty(0, dtype=torch.long), persistent=False)

    def set_activity_pair_count_matrix(self, pair_count_matrix: Optional[torch.Tensor]) -> None:
        if pair_count_matrix is None:
            self.pair_count_matrix = torch.empty(0, dtype=torch.long, device=self.time_w.device)
            return
        self.pair_count_matrix = pair_count_matrix.to(device=self.time_w.device, dtype=torch.long)

    def gamma_l1_penalty(self) -> torch.Tensor:
        if not self.use_act_lowrank_bias:
            return torch.tensor(0.0, device=self.time_w.device)
        gamma = torch.sigmoid(self.theta_act_bias)
        return torch.abs(gamma)

    def gamma_value(self) -> Optional[torch.Tensor]:
        if not self.use_act_lowrank_bias:
            return None
        return torch.sigmoid(self.theta_act_bias)

    def _normalize_time(
        self,
        time_matrix: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Accept [B, T, T] or [B, T, T, 1]
        if time_matrix.dim() == 4:
            time_matrix = time_matrix[..., 0]

        u = torch.log1p(torch.clamp(time_matrix, min=0.0) / self.time_tau0)

        if padding_mask is None:
            mean = u.mean(dim=(1, 2), keepdim=True)
            std = u.std(dim=(1, 2), keepdim=True, unbiased=False).clamp(min=self.time_eps)
            return torch.clamp((u - mean) / std, min=-6.0, max=6.0)

        valid = (padding_mask > 0).to(dtype=u.dtype)
        valid_pair = valid.unsqueeze(1) * valid.unsqueeze(2)  # [B, T, T]
        denom = valid_pair.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)

        mean = (u * valid_pair).sum(dim=(1, 2), keepdim=True) / denom
        var = (((u - mean) ** 2) * valid_pair).sum(dim=(1, 2), keepdim=True) / denom
        std = torch.sqrt(var).clamp(min=self.time_eps)

        u_hat = (u - mean) / std
        u_hat = u_hat * valid_pair
        return torch.clamp(u_hat, min=-6.0, max=6.0)

    def _activity_pair_valid_mask(self, act_seq: torch.Tensor) -> Optional[torch.Tensor]:
        if self.pair_count_threshold <= 0:
            return None
        if self.pair_count_matrix.numel() == 0:
            return None

        bsz, seq_len = act_seq.size()
        idx_i = act_seq.unsqueeze(2).expand(-1, -1, seq_len)
        idx_j = act_seq.unsqueeze(1).expand(-1, seq_len, -1)
        pair_counts = self.pair_count_matrix[idx_i, idx_j]
        return (pair_counts >= self.pair_count_threshold).unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        act_seq: Optional[torch.Tensor] = None,
        time_matrix: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()

        if padding_mask is None:
            padding_mask = torch.ones(bsz, seq_len, dtype=torch.long, device=x.device)

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        u_hat = None
        if self.use_time_bias and time_matrix is not None:
            u_hat = self._normalize_time(time_matrix, padding_mask=padding_mask)  # [B, T, T]
            b_time = u_hat.unsqueeze(1) * self.time_w.view(1, self.num_heads, 1, 1)
            b_time = b_time + self.time_c.view(1, self.num_heads, 1, 1)
            b_time = b_time * self.time_lambda.view(1, self.num_heads, 1, 1)
            scores = scores + b_time

        if self.use_act_lowrank_bias and act_seq is not None:
            b_act = self.act_bias(act_seq)
            valid_pair_mask = self._activity_pair_valid_mask(act_seq)
            if valid_pair_mask is not None:
                b_act = b_act * valid_pair_mask.to(dtype=b_act.dtype)
            gamma = torch.sigmoid(self.theta_act_bias)
            scores = scores + gamma * b_act

        key_mask = (padding_mask == 0).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, v)  # [B, H, T, d_h]

        if self.use_time_value_gate and u_hat is not None:
            valid = (padding_mask > 0).to(u_hat.dtype)
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            u_mean = (u_hat * valid.unsqueeze(1)).sum(dim=-1) / denom  # [B, T]
            gate = torch.sigmoid(
                self.eta_time.view(1, self.num_heads, 1) * u_mean.unsqueeze(1)
                + self.kappa_time.view(1, self.num_heads, 1)
            )
            out = out * gate.unsqueeze(-1)

        out = out * (padding_mask > 0).unsqueeze(1).unsqueeze(-1).to(out.dtype)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class STGTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        vocab_size: int,
        dropout: float = 0.1,
        use_time_bias: bool = True,
        use_act_lowrank_bias: bool = False,
        use_time_value_gate: bool = True,
        act_bias_rank: int = 16,
        time_tau0: float = 1.0,
        time_lambda_init: float = 0.8,
        act_gamma_init: float = -3.0,
        pair_count_threshold: int = 20,
    ):
        super().__init__()
        self.self_attn = SpatioTemporalAttention(
            d_model=d_model,
            num_heads=num_heads,
            vocab_size=vocab_size,
            use_time_bias=use_time_bias,
            use_act_lowrank_bias=use_act_lowrank_bias,
            use_time_value_gate=use_time_value_gate,
            act_bias_rank=act_bias_rank,
            time_tau0=time_tau0,
            time_lambda_init=time_lambda_init,
            act_gamma_init=act_gamma_init,
            pair_count_threshold=pair_count_threshold,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        act_seq: torch.Tensor,
        time_matrix: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.self_attn(
            x=x_norm,
            act_seq=act_seq,
            time_matrix=time_matrix,
            padding_mask=padding_mask,
        )
        y = x + self.dropout1(attn_out)

        y_norm = self.norm2(y)
        ff = self.linear2(self.ffn_dropout(F.relu(self.linear1(y_norm))))
        z = y + self.dropout2(ff)

        if padding_mask is not None:
            z = z * (padding_mask > 0).unsqueeze(-1).to(z.dtype)
        return z


class AttnPool(nn.Module):
    """Masked attention pooling to replace last-step readout."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.score(torch.tanh(self.proj(x))).squeeze(-1)
        s = s.masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(s, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return torch.sum(alpha.unsqueeze(-1) * x, dim=1)


class DistributionHead(nn.Module):
    """Predict mean and uncertainty for heteroscedastic regression."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.sigma_head(h).squeeze(-1)) + 1e-6
        return mu, sigma


class IsoFormerSTGTransformer(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_resources: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        model_type: str = "full",
        use_dynamic_fusion: bool = True,
        use_time_bias: bool = True,
        use_time_value_gate: bool = True,
        use_act_lowrank_bias: bool = False,
        act_bias_rank: int = 16,
        time_tau0: float = 1.0,
        time_lambda_init: float = 0.8,
        act_gamma_init: float = -3.0,
        pair_count_threshold: int = 20,
        use_attn_pool: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        # Backward compatibility for legacy model_type flag.
        if model_type == "time_only":
            use_time_bias = True
            use_act_lowrank_bias = False
        elif model_type == "space_only":
            use_time_bias = False
            use_act_lowrank_bias = True
        elif model_type == "original":
            use_time_bias = True
            use_act_lowrank_bias = False

        self.use_attn_pool = use_attn_pool

        self.act_encoder = nn.Embedding(num_activities, d_model, padding_idx=0)
        self.res_encoder = nn.Embedding(num_resources, d_model, padding_idx=0)
        self.time_encoder = TimeTokenEncoder(d_model)

        self.init_norm_act = nn.LayerNorm(d_model)
        self.init_norm_res = nn.LayerNorm(d_model)

        self.fusion_module = MultiModalFusionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_dynamic_gating=use_dynamic_fusion,
        )

        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                STGTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    vocab_size=num_activities,
                    dropout=dropout,
                    use_time_bias=use_time_bias,
                    use_act_lowrank_bias=use_act_lowrank_bias,
                    use_time_value_gate=use_time_value_gate,
                    act_bias_rank=act_bias_rank,
                    time_tau0=time_tau0,
                    time_lambda_init=time_lambda_init,
                    act_gamma_init=act_gamma_init,
                    pair_count_threshold=pair_count_threshold,
                )
                for _ in range(num_layers)
            ]
        )

        self.pool = AttnPool(d_model)
        self.dist_head = DistributionHead(d_model, dropout=dropout)

    def set_activity_pair_count_matrix(self, pair_count_matrix: Optional[torch.Tensor]) -> None:
        for layer in self.layers:
            layer.self_attn.set_activity_pair_count_matrix(pair_count_matrix)

    def gamma_l1_penalty(self) -> torch.Tensor:
        penalties = [layer.self_attn.gamma_l1_penalty() for layer in self.layers]
        if not penalties:
            return torch.tensor(0.0, device=self.act_encoder.weight.device)
        return torch.stack(penalties).sum()

    def gamma_values(self) -> List[float]:
        values: List[float] = []
        for layer in self.layers:
            gamma = layer.self_attn.gamma_value()
            if gamma is None:
                values.append(0.0)
            else:
                values.append(float(gamma.detach().cpu().item()))
        return values

    def forward(
        self,
        act_seq: torch.Tensor,
        res_seq: torch.Tensor,
        time_features: torch.Tensor,
        time_matrix: Optional[torch.Tensor] = None,
        graph_matrix: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dist: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        del graph_matrix  # Graph bias path is intentionally disabled in this version.

        if padding_mask is None:
            padding_mask = (act_seq != 0).long()

        act_emb = self.act_encoder(act_seq)
        res_emb = self.res_encoder(res_seq)

        # Keep token-level time encoder simple and stable.
        time_tok = self.time_encoder(time_features)

        x_act0 = self.init_norm_act(act_emb + time_tok)
        x_res0 = self.init_norm_res(res_emb + time_tok)

        x = self.fusion_module(x_act0, x_res0, time_tok, padding_mask=padding_mask)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, act_seq, time_matrix, padding_mask)

        if self.use_attn_pool:
            h = self.pool(x, padding_mask)
        else:
            last_idx = torch.clamp(padding_mask.sum(dim=1) - 1, min=0)
            h = x[torch.arange(x.size(0), device=x.device), last_idx, :]

        mu, sigma = self.dist_head(h)
        if return_dist:
            return mu, sigma
        return mu


# ------------------------- quick local smoke test -------------------------
def train_example() -> None:
    num_activities = 100
    num_resources = 50
    d_model = 128
    batch_size = 16
    seq_len = 40
    epochs = 3
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsoFormerSTGTransformer(
        num_activities=num_activities,
        num_resources=num_resources,
        d_model=d_model,
        num_heads=8,
        num_layers=4,
        use_act_lowrank_bias=False,
        pair_count_threshold=20,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()

        act_seq = torch.randint(1, num_activities, (batch_size, seq_len), device=device)
        res_seq = torch.randint(1, num_resources, (batch_size, seq_len), device=device)
        time_features = torch.rand(batch_size, seq_len, 2, device=device)
        time_matrix = torch.rand(batch_size, seq_len, seq_len, 1, device=device)

        lengths = torch.randint(seq_len // 2, seq_len + 1, (batch_size,), device=device)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        for i, ln in enumerate(lengths):
            padding_mask[i, : int(ln.item())] = 1

        y_true = torch.rand(batch_size, device=device) * 200.0
        variant_freq = torch.randint(1, 50, (batch_size,), device=device).float()

        mu, sigma = model(
            act_seq,
            res_seq,
            time_features,
            time_matrix=time_matrix,
            padding_mask=padding_mask,
            return_dist=True,
        )

        w = torch.pow(variant_freq + 1e-6, -0.5)
        w = w / w.mean().clamp(min=1e-6)

        l1 = (w * torch.abs(y_true - mu)).mean()
        nll = (w * (((y_true - mu) ** 2) / (2.0 * sigma ** 2) + torch.log(sigma))).mean()
        loss = 0.6 * l1 + 0.4 * nll + 2e-4 * model.gamma_l1_penalty()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Epoch {epoch + 1:02d} | Loss: {loss.item():.4f} | MAE: {l1.item():.4f}")


if __name__ == "__main__":
    train_example()
