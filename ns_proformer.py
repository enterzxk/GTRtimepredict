import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MotifBPETokenizer:
    """Process-aware BPE tokenizer that merges frequent adjacent activity tokens."""

    def __init__(
        self,
        max_vocab_size: int = 4000,
        min_pair_count: int = 20,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
    ) -> None:
        self.max_vocab_size = max_vocab_size
        self.min_pair_count = min_pair_count

        self.pad_token = pad_token
        self.unk_token = unk_token

        self.pad_id = 0
        self.unk_id = 1

        self.activity2id: Dict[str, int] = {pad_token: self.pad_id, unk_token: self.unk_id}
        self.id2activity: Dict[int, str] = {self.pad_id: pad_token, self.unk_id: unk_token}

        self.token_to_atomic_seq: Dict[int, Tuple[int, ...]] = {
            self.pad_id: tuple(),
            self.unk_id: tuple(),
        }
        self.merge_rules: List[Tuple[int, int, int]] = []
        self.vocab_size = 2

    def _merge_once(self, seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        left, right = pair
        merged: List[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
                merged.append(new_id)
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        return merged

    def fit(self, activity_sequences: Sequence[Sequence[str]]) -> None:
        """Fit BPE merges on training sequences."""
        # Reset learned vocabulary so repeated fit calls stay deterministic.
        self.activity2id = {self.pad_token: self.pad_id, self.unk_token: self.unk_id}
        self.id2activity = {self.pad_id: self.pad_token, self.unk_id: self.unk_token}
        self.token_to_atomic_seq = {self.pad_id: tuple(), self.unk_id: tuple()}
        self.merge_rules = []
        self.vocab_size = 2

        for seq in activity_sequences:
            for act in seq:
                if act not in self.activity2id:
                    new_id = len(self.activity2id)
                    self.activity2id[act] = new_id
                    self.id2activity[new_id] = act
                    self.token_to_atomic_seq[new_id] = (new_id,)

        # Ensure motif ids never collide with existing atomic ids.
        self.vocab_size = len(self.activity2id)

        corpus: List[List[int]] = []
        for seq in activity_sequences:
            ids = [self.activity2id.get(act, self.unk_id) for act in seq]
            if ids:
                corpus.append(ids)

        while self.vocab_size < self.max_vocab_size:
            pair_counts: Counter = Counter()
            for seq in corpus:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            (left, right), count = pair_counts.most_common(1)[0]
            if count < self.min_pair_count:
                break

            new_id = self.vocab_size
            self.vocab_size += 1

            self.merge_rules.append((left, right, new_id))
            left_atoms = self.token_to_atomic_seq.get(left, tuple())
            right_atoms = self.token_to_atomic_seq.get(right, tuple())
            self.token_to_atomic_seq[new_id] = left_atoms + right_atoms
            self.id2activity[new_id] = f"Motif_{new_id}"

            corpus = [self._merge_once(seq, (left, right), new_id) for seq in corpus]

        # Keep internal vocab size consistent after fitting.
        if self.vocab_size < len(self.activity2id):
            self.vocab_size = len(self.activity2id)

    def encode(self, activities: Sequence[str], max_len: Optional[int] = None) -> Tuple[List[int], List[List[int]]]:
        """Encode one sequence to motif token ids and token-to-event spans."""
        token_ids = [self.activity2id.get(act, self.unk_id) for act in activities]
        spans = [[i] for i in range(len(token_ids))]

        for left, right, new_id in self.merge_rules:
            merged_tokens: List[int] = []
            merged_spans: List[List[int]] = []

            i = 0
            while i < len(token_ids):
                if i < len(token_ids) - 1 and token_ids[i] == left and token_ids[i + 1] == right:
                    merged_tokens.append(new_id)
                    merged_spans.append(spans[i] + spans[i + 1])
                    i += 2
                else:
                    merged_tokens.append(token_ids[i])
                    merged_spans.append(spans[i])
                    i += 1

            token_ids = merged_tokens
            spans = merged_spans

        if max_len is not None and len(token_ids) > max_len:
            token_ids = token_ids[-max_len:]
            spans = spans[-max_len:]

        return token_ids, spans

    @staticmethod
    def build_time_aggregates(
        spans: Sequence[Sequence[int]],
        time_since_last: Sequence[float],
        time_since_start: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Build motif-level aggregated temporal features."""
        rows: List[List[float]] = []
        for span in spans:
            if not span:
                rows.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            durations = np.asarray([max(0.0, float(time_since_last[i])) for i in span], dtype=np.float32)
            total = float(durations.sum())
            mean = float(durations.mean())
            var = float(durations.var())
            length = float(len(span))

            if time_since_start is not None:
                first_t = float(time_since_start[span[0]])
                last_t = float(time_since_start[span[-1]])
                first_last_interval = max(0.0, last_t - first_t)
            else:
                first_last_interval = total

            rows.append([total, mean, var, length, first_last_interval])

        if not rows:
            return np.zeros((0, 5), dtype=np.float32)

        return np.asarray(rows, dtype=np.float32)


class ProcessStructurePrior:
    """Builds reachability prior and marking vectors from training traces."""

    def __init__(
        self,
        tokenizer: MotifBPETokenizer,
        activity_sequences: Sequence[Sequence[str]],
        reachability_mode: str = "direct",
        reachability_hops: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.reachability_mode = reachability_mode
        self.reachability_hops = max(1, int(reachability_hops))

        self.num_atomic = max(tokenizer.activity2id.values()) + 1
        adjacency = np.zeros((self.num_atomic, self.num_atomic), dtype=bool)

        for seq in activity_sequences:
            ids = [tokenizer.activity2id.get(act, tokenizer.unk_id) for act in seq]
            for a, b in zip(ids[:-1], ids[1:]):
                if a <= tokenizer.unk_id or b <= tokenizer.unk_id:
                    continue
                adjacency[a, b] = True

        np.fill_diagonal(adjacency, True)
        self.atomic_adjacency = adjacency

        if self.reachability_mode == "transitive":
            self.atomic_reachability = self._transitive_closure(adjacency)
        elif self.reachability_mode == "direct":
            self.atomic_reachability = adjacency.copy()
        elif self.reachability_mode == "k_hop":
            self.atomic_reachability = self._k_hop_reachability(adjacency, self.reachability_hops)
        else:
            raise ValueError(
                f"Unsupported reachability_mode: {self.reachability_mode}. "
                f"Expected one of ['direct', 'k_hop', 'transitive']."
            )

        self.atomic_enabled = self._build_enabled_vectors(adjacency)
        self.marking_dim = self.atomic_enabled.shape[1]

        self.token_reachability = self._build_token_reachability()

    @staticmethod
    def _transitive_closure(adjacency: np.ndarray) -> np.ndarray:
        reach = adjacency.copy()
        n = reach.shape[0]
        for k in range(n):
            reach = reach | (reach[:, [k]] & reach[[k], :])
        return reach

    @staticmethod
    def _k_hop_reachability(adjacency: np.ndarray, hops: int) -> np.ndarray:
        n = adjacency.shape[0]
        reach = np.eye(n, dtype=bool)

        adj_int = adjacency.astype(np.int32)
        power = adjacency.copy()

        for _ in range(hops):
            reach |= power
            power = (power.astype(np.int32) @ adj_int) > 0

        np.fill_diagonal(reach, True)
        return reach

    def _build_enabled_vectors(self, adjacency: np.ndarray) -> np.ndarray:
        enabled = np.zeros_like(adjacency, dtype=np.float32)
        for i in range(adjacency.shape[0]):
            row = adjacency[i].copy()
            if not row.any():
                row[i] = True
            enabled[i] = row.astype(np.float32)
        return enabled

    def _build_token_reachability(self) -> np.ndarray:
        vocab_size = self.tokenizer.vocab_size
        table = np.zeros((vocab_size, vocab_size), dtype=bool)

        for i in range(vocab_size):
            seq_i = self.tokenizer.token_to_atomic_seq.get(i, tuple())
            if not seq_i:
                table[i, i] = True
                continue
            end_i = seq_i[-1]

            for j in range(vocab_size):
                seq_j = self.tokenizer.token_to_atomic_seq.get(j, tuple())
                if not seq_j:
                    continue
                start_j = seq_j[0]
                if self.atomic_reachability[end_i, start_j]:
                    table[i, j] = True

            table[i, i] = True

        return table

    def build_reachability_mask(self, token_ids: Sequence[int]) -> np.ndarray:
        if len(token_ids) == 0:
            return np.zeros((0, 0), dtype=bool)

        idx = np.asarray(token_ids, dtype=np.int64)
        mask = self.token_reachability[np.ix_(idx, idx)].copy()
        np.fill_diagonal(mask, True)
        return mask

    def build_marking_vector(self, last_activity: str) -> np.ndarray:
        last_id = self.tokenizer.activity2id.get(last_activity, self.tokenizer.unk_id)
        if 0 <= last_id < self.atomic_enabled.shape[0]:
            mark = self.atomic_enabled[last_id].copy()
        else:
            mark = np.zeros((self.marking_dim,), dtype=np.float32)

        if mark.sum() == 0:
            fallback = min(max(last_id, 0), self.marking_dim - 1)
            mark[fallback] = 1.0

        return mark.astype(np.float32)


class NSProFormerPrefixDataset(Dataset):
    """Prefix dataset that outputs motif tokens, masks, marking and RRT target."""

    def __init__(
        self,
        df,
        tokenizer: MotifBPETokenizer,
        prior: ProcessStructurePrior,
        max_seq_len: int = 80,
        max_token_len: int = 128,
        max_prefixes_per_case: Optional[int] = None,
        activity_col: str = "Activity",
        normalization_stats: Optional[Dict[str, np.ndarray]] = None,
        fit_normalization: bool = True,
        normalization_eps: float = 1e-6,
    ) -> None:
        self.activity_col = activity_col
        self.normalization_eps = normalization_eps

        required = {"CaseID", self.activity_col, "TimeSinceLast", "TimeSinceStart", "Remaining_Time"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

        self.tokenizer = tokenizer
        self.prior = prior
        self.max_seq_len = max_seq_len
        self.max_token_len = max_token_len
        self.max_prefixes_per_case = max_prefixes_per_case

        self.samples: List[Dict[str, np.ndarray]] = []

        working_df = df.copy()
        if "Timestamp" in working_df.columns:
            working_df = working_df.sort_values(["CaseID", "Timestamp"])
        else:
            working_df = working_df.sort_values(["CaseID"]).reset_index(drop=True)

        self._build_samples(working_df)
        self.normalization_stats = self._init_normalization_stats(
            normalization_stats=normalization_stats,
            fit_normalization=fit_normalization,
        )

    def _init_normalization_stats(
        self,
        normalization_stats: Optional[Dict[str, np.ndarray]] = None,
        fit_normalization: bool = True,
    ) -> Dict[str, np.ndarray]:
        if normalization_stats is not None:
            scale = np.asarray(normalization_stats.get("time_agg_scale", np.ones((5,), dtype=np.float32)), dtype=np.float32)
            scale = np.maximum(scale, self.normalization_eps)
            return {"time_agg_scale": scale}

        if fit_normalization:
            return self._build_normalization_stats_from_samples()

        return {"time_agg_scale": np.ones((5,), dtype=np.float32)}

    def _build_normalization_stats_from_samples(self) -> Dict[str, np.ndarray]:
        rows: List[np.ndarray] = []
        for item in self.samples:
            g = item["time_agg"]
            if g.ndim == 2 and g.shape[0] > 0:
                rows.append(g)

        if not rows:
            return {"time_agg_scale": np.ones((5,), dtype=np.float32)}

        all_g = np.concatenate(rows, axis=0)
        scale = np.std(all_g, axis=0).astype(np.float32)
        scale = np.maximum(scale, self.normalization_eps)
        return {"time_agg_scale": scale}

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        return {"time_agg_scale": self.normalization_stats["time_agg_scale"].copy()}

    def _build_samples(self, df) -> None:
        grouped = df.groupby("CaseID", sort=False)

        for _, group in grouped:
            acts = group[self.activity_col].astype(str).tolist()
            time_last = group["TimeSinceLast"].astype(float).tolist()
            time_start = group["TimeSinceStart"].astype(float).tolist()
            rem_times = group["Remaining_Time"].astype(float).tolist()

            total_events = len(acts)
            if total_events == 0:
                continue

            start_idx = 1
            if self.max_prefixes_per_case and total_events > self.max_prefixes_per_case:
                start_idx = total_events - self.max_prefixes_per_case + 1

            for i in range(start_idx, total_events + 1):
                left = max(0, i - self.max_seq_len)

                prefix_acts = acts[left:i]
                prefix_time_last = time_last[left:i]
                prefix_time_start = time_start[left:i]

                token_ids, spans = self.tokenizer.encode(prefix_acts, max_len=self.max_token_len)
                if not token_ids:
                    continue

                time_agg = self.tokenizer.build_time_aggregates(spans, prefix_time_last, prefix_time_start)
                reach_mask = self.prior.build_reachability_mask(token_ids)
                marking = self.prior.build_marking_vector(prefix_acts[-1])

                self.samples.append(
                    {
                        "token_ids": np.asarray(token_ids, dtype=np.int64),
                        "time_agg": time_agg.astype(np.float32),
                        "reachability_mask": reach_mask.astype(bool),
                        "marking": marking.astype(np.float32),
                        "target_rem_time": np.float32(rem_times[i - 1]),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        token_ids = item["token_ids"]
        time_agg = item["time_agg"]
        reach = item["reachability_mask"]

        scale = self.normalization_stats["time_agg_scale"]
        time_agg = (time_agg / scale).astype(np.float32)

        seq_len = min(len(token_ids), self.max_token_len)
        pad_len = self.max_token_len - seq_len

        padded_ids = np.pad(token_ids[:seq_len], (0, pad_len), constant_values=self.tokenizer.pad_id)

        g_dim = time_agg.shape[1] if time_agg.ndim == 2 and time_agg.shape[0] > 0 else 5
        padded_g = np.zeros((self.max_token_len, g_dim), dtype=np.float32)
        if seq_len > 0:
            padded_g[:seq_len] = time_agg[:seq_len]

        padded_reach = np.zeros((self.max_token_len, self.max_token_len), dtype=bool)
        if seq_len > 0:
            padded_reach[:seq_len, :seq_len] = reach[:seq_len, :seq_len]
            for i in range(seq_len):
                padded_reach[i, i] = True

        valid_mask = np.zeros((self.max_token_len,), dtype=bool)
        valid_mask[:seq_len] = True

        return {
            "token_ids": torch.tensor(padded_ids, dtype=torch.long),
            "time_agg": torch.tensor(padded_g, dtype=torch.float32),
            "reachability_mask": torch.tensor(padded_reach, dtype=torch.bool),
            "marking": torch.tensor(item["marking"], dtype=torch.float32),
            "mask": torch.tensor(valid_mask, dtype=torch.bool),
            "target_rem_time": torch.tensor(item["target_rem_time"], dtype=torch.float32),
        }


def split_by_case(df, train_ratio: float = 0.8, seed: int = 42):
    """Split dataframe by case id to avoid case-level leakage."""
    case_ids = df["CaseID"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(case_ids)

    split_idx = int(len(case_ids) * train_ratio)
    train_cases = set(case_ids[:split_idx])
    val_cases = set(case_ids[split_idx:])

    train_df = df[df["CaseID"].isin(train_cases)].copy()
    val_df = df[df["CaseID"].isin(val_cases)].copy()
    return train_df, val_df


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ReachabilityGuidedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        reachability_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if reachability_mask is None:
            allow = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=x.device)
        else:
            allow = reachability_mask.bool().unsqueeze(1)

        if padding_mask is not None:
            valid = padding_mask.bool()
            key_valid = valid.unsqueeze(1).unsqueeze(2)
            query_valid = valid.unsqueeze(1).unsqueeze(-1)
            allow = allow & key_valid & query_valid

        diag = torch.eye(seq_len, dtype=torch.bool, device=x.device).unsqueeze(0).unsqueeze(0)
        allow = allow | diag

        scores = scores.masked_fill(~allow, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(out.dtype)

        return out


class ReachabilityGuidedTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = ReachabilityGuidedSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        reachability_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, reachability_mask, padding_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(x.dtype)

        return x


class GaussianMDNHead(nn.Module):
    def __init__(self, d_model: int, num_components: int = 5, sigma_eps: float = 1e-6) -> None:
        super().__init__()
        self.num_components = num_components
        self.sigma_eps = sigma_eps

        self.pi_layer = nn.Linear(d_model, num_components)
        self.mu_layer = nn.Linear(d_model, num_components)
        self.sigma_layer = nn.Linear(d_model, num_components)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        pi_logits = self.pi_layer(h)
        mu = self.mu_layer(h)
        sigma = F.softplus(self.sigma_layer(h)) + self.sigma_eps
        pi = torch.softmax(pi_logits, dim=-1)

        mean = torch.sum(pi * mu, dim=-1)
        variance = torch.sum(pi * (sigma.pow(2) + mu.pow(2)), dim=-1) - mean.pow(2)
        variance = torch.clamp(variance, min=1e-8)
        eta = torch.clamp(mean, min=0.0)

        return {
            "pi_logits": pi_logits,
            "pi": pi,
            "mu": mu,
            "sigma": sigma,
            "mean": mean,
            "eta": eta,
            "variance": variance,
        }

    @staticmethod
    def nll_loss(
        target: torch.Tensor,
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        y = target.unsqueeze(-1)
        log_pi = torch.log_softmax(pi_logits, dim=-1)
        log_sigma = torch.log(sigma)

        log_prob = -0.5 * (((y - mu) / sigma).pow(2) + 2.0 * log_sigma + math.log(2.0 * math.pi))
        log_mix = log_pi + log_prob
        nll = -torch.logsumexp(log_mix, dim=-1)

        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        return nll.mean()

    @staticmethod
    def sample(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, num_samples: int = 200) -> torch.Tensor:
        """Draw samples from the Gaussian mixture for interval estimation."""
        batch_size, num_components = pi.shape
        comp_idx = torch.multinomial(pi, num_samples=num_samples, replacement=True)

        mu_sel = mu.gather(1, comp_idx)
        sigma_sel = sigma.gather(1, comp_idx)
        eps = torch.randn(batch_size, num_samples, device=mu.device, dtype=mu.dtype)
        return mu_sel + sigma_sel * eps

    @staticmethod
    def quantiles(
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        q: Tuple[float, float] = (0.1, 0.9),
        num_samples: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = GaussianMDNHead.sample(pi, mu, sigma, num_samples=num_samples)
        lower = torch.quantile(samples, q[0], dim=1)
        upper = torch.quantile(samples, q[1], dim=1)
        return lower, upper


class NSProFormer(nn.Module):
    """NS-ProFormer: motif compression + reachability-guided encoder + MDN head."""

    def __init__(
        self,
        vocab_size: int,
        marking_dim: int,
        g_dim: int = 5,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_mixtures: int = 5,
        max_len: int = 256,
        inject_marking_each_layer: bool = True,
    ) -> None:
        super().__init__()
        self.inject_marking_each_layer = inject_marking_each_layer

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.time_proj = nn.Linear(g_dim, d_model)
        self.marking_proj = nn.Linear(marking_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [ReachabilityGuidedTransformerLayer(d_model, num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )

        self.mdn = GaussianMDNHead(d_model=d_model, num_components=num_mixtures)

    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)

        mask_f = mask.unsqueeze(-1).to(x.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (x * mask_f).sum(dim=1) / denom

    def forward(
        self,
        token_ids: torch.Tensor,
        time_agg: torch.Tensor,
        reachability_mask: Optional[torch.Tensor],
        marking: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        time_agg = torch.log1p(torch.clamp(time_agg, min=0.0))

        x = self.token_emb(token_ids) + self.time_proj(time_agg)
        x = self.pos_emb(x)

        state = self.marking_proj(marking)
        x = x + state.unsqueeze(1)
        x = self.input_dropout(x)

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(x.dtype)

        for layer in self.layers:
            if self.inject_marking_each_layer:
                x = x + state.unsqueeze(1)
            x = layer(x, reachability_mask, padding_mask)

        pooled = self._masked_mean_pool(x, padding_mask)
        out = self.mdn(pooled)

        if target is not None:
            out["nll"] = self.mdn.nll_loss(target, out["pi_logits"], out["mu"], out["sigma"], reduction="mean")

        return out


__all__ = [
    "MotifBPETokenizer",
    "ProcessStructurePrior",
    "NSProFormerPrefixDataset",
    "split_by_case",
    "NSProFormer",
    "GaussianMDNHead",
]
