import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

# ==========================================
# Baseline 1: 经典长短期记忆网络 (LSTM)
# ==========================================
class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size_act, vocab_size_res, d_model=64, hidden_size=128):
        super().__init__()
        self.act_emb = nn.Embedding(vocab_size_act, d_model, padding_idx=0)
        self.res_emb = nn.Embedding(vocab_size_res, d_model, padding_idx=0)
        self.time_proj = nn.Linear(2, d_model)

        # LSTM 输入为三个模态拼接
        input_dim = d_model * 3

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()
        )

    def forward(self, act_seq, res_seq, time_seq, mask):
        act_out = self.act_emb(act_seq)
        res_out = self.res_emb(res_seq)
        time_out = self.time_proj(time_seq)

        x = torch.cat([act_out, res_out, time_out], dim=-1)
        lstm_out, _ = self.lstm(x)

        # 优化点：利用 mask 找到每个序列真实的最后一个事件，而不是盲目取 -1 (那可能是 Padding)
        batch_size = x.size(0)
        last_idx = mask.sum(dim=1) - 1
        last_idx = torch.clamp(last_idx, min=0)

        last_hidden = lstm_out[torch.arange(batch_size), last_idx, :]
        return self.fc(last_hidden).squeeze(-1)


# ==========================================
# Baseline 2: 传统原生 Transformer (Vanilla)
# ==========================================
class VanillaTransformerBaseline(nn.Module):
    """
    不包含时空图自适应偏置的纯净版 Transformer，用于证明 STG 模块的有效性。
    """

    def __init__(self, vocab_size_act, vocab_size_res, d_model=64, num_heads=4, num_layers=2, dim_feedforward=256,
                 dropout=0.1):
        super().__init__()
        self.act_emb = nn.Embedding(vocab_size_act, d_model, padding_idx=0)
        self.res_emb = nn.Embedding(vocab_size_res, d_model, padding_idx=0)
        self.time_proj = nn.Linear(2, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        # PyTorch 原生 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )

    def forward(self, act_seq, res_seq, time_seq, mask):
        # 【核心修复】：对极端长的时间特征进行对数平滑，防止 Attention 点积爆炸
        # clamp 确保没有负数，log1p = ln(1 + x) 负责压缩数值边界
        time_seq_safe = torch.log1p(torch.clamp(time_seq, min=0.0))

        # 1. 特征直接加和融合
        act_out = self.act_emb(act_seq)
        res_out = self.res_emb(res_seq)
        time_out = self.time_proj(time_seq_safe)  # 使用平滑后的时间特征

        x = act_out + res_out + time_out

        # 2. 加入位置编码
        x = self.pos_encoder(x)

        # 3. 处理掩码
        pad_mask = (mask == 0)

        # 4. 进入 Transformer
        out = self.transformer_encoder(x, src_key_padding_mask=pad_mask)

        # 5. 提取序列的最后一个有效事件用于预测
        batch_size = x.size(0)
        last_idx = mask.sum(dim=1) - 1
        last_idx = torch.clamp(last_idx, min=0)

        last_out = out[torch.arange(batch_size), last_idx, :]
        return self.fc(last_out).squeeze(-1)


class GRUBaseline(nn.Module):
    def __init__(self, vocab_size_act, vocab_size_res, d_model=64, hidden_size=128):
        super().__init__()
        self.act_emb = nn.Embedding(vocab_size_act, d_model, padding_idx=0)
        self.res_emb = nn.Embedding(vocab_size_res, d_model, padding_idx=0)
        self.time_proj = nn.Linear(2, d_model)

        input_dim = d_model * 3
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus(),
        )

    def forward(self, act_seq, res_seq, time_seq, mask):
        act_out = self.act_emb(act_seq)
        res_out = self.res_emb(res_seq)
        time_out = self.time_proj(time_seq)

        x = torch.cat([act_out, res_out, time_out], dim=-1)
        gru_out, _ = self.gru(x)

        batch_size = x.size(0)
        last_idx = torch.clamp(mask.sum(dim=1) - 1, min=0)
        last_hidden = gru_out[torch.arange(batch_size), last_idx, :]
        return self.fc(last_hidden).squeeze(-1)


class TemporalCNNBaseline(nn.Module):
    def __init__(self, vocab_size_act, vocab_size_res, d_model=64, hidden_size=128, dropout=0.1):
        super().__init__()
        self.act_emb = nn.Embedding(vocab_size_act, d_model, padding_idx=0)
        self.res_emb = nn.Embedding(vocab_size_res, d_model, padding_idx=0)
        self.time_proj = nn.Linear(2, d_model)

        input_dim = d_model * 3
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus(),
        )

    def forward(self, act_seq, res_seq, time_seq, mask):
        act_out = self.act_emb(act_seq)
        res_out = self.res_emb(res_seq)
        time_out = self.time_proj(time_seq)

        x = torch.cat([act_out, res_out, time_out], dim=-1)
        x = x.transpose(1, 2)
        conv_out = self.conv(x).transpose(1, 2)

        batch_size = conv_out.size(0)
        last_idx = torch.clamp(mask.sum(dim=1) - 1, min=0)
        last_hidden = conv_out[torch.arange(batch_size), last_idx, :]
        return self.fc(last_hidden).squeeze(-1)


