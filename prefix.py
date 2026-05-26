import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter


class ProcessVocab:
    """
    词表构建器：将文本分类变量映射为整数索引
    """

    def __init__(self):
        self.pad_idx = 0
        self.unk_idx = 1
        self.end_idx = 2

        self.act2id = {'[PAD]': 0, '[UNK]': 1, '[END]': 2}
        self.id2act = {0: '[PAD]', 1: '[UNK]', 2: '[END]'}

        self.res2id = {'[PAD]': 0, '[UNK]': 1}
        self.id2res = {0: '[PAD]', 1: '[UNK]'}

    def build_vocab(self, df):
        for act in df['Activity'].unique():
            if act not in self.act2id:
                idx = len(self.act2id)
                self.act2id[act] = idx
                self.id2act[idx] = act

        for res in df['Resource'].unique():
            if res not in self.res2id:
                idx = len(self.res2id)
                self.res2id[res] = idx
                self.id2res[idx] = res

    def get_act_id(self, act):
        return self.act2id.get(act, self.unk_idx)

    def get_res_id(self, res):
        return self.res2id.get(res, self.unk_idx)


class ProcessPrefixDataset(Dataset):
    """
    PyTorch 数据集：动态生成带有时序偏置矩阵的特征序列 (已移除空间拓扑特征)
    """

    def __init__(
        self,
        df,
        vocab,
        max_seq_len=50,
        max_prefixes_per_case=None,
        normalization_stats=None,
        fit_normalization=True,
        normalization_eps=1e-6,
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_prefixes_per_case = max_prefixes_per_case
        self.normalization_eps = normalization_eps
        self.prefixes = []
        self.variant2id = {}
        self.variant_freq_table = {}

        self._generate_prefixes(df)
        self._finalize_variant_stats()

        self.normalization_stats = self._init_normalization_stats(
            df,
            normalization_stats=normalization_stats,
            fit_normalization=fit_normalization,
        )

    def _init_normalization_stats(self, df, normalization_stats=None, fit_normalization=True):
        if normalization_stats is not None:
            return {
                'time_last_scale': max(float(normalization_stats.get('time_last_scale', 1.0)), self.normalization_eps),
                'time_start_scale': max(float(normalization_stats.get('time_start_scale', 1.0)), self.normalization_eps),
                'time_matrix_scale': max(float(normalization_stats.get('time_matrix_scale', 1.0)), self.normalization_eps),
            }

        if fit_normalization:
            return self._build_normalization_stats(df)

        return {
            'time_last_scale': 1.0,
            'time_start_scale': 1.0,
            'time_matrix_scale': 1.0,
        }

    def _build_normalization_stats(self, df):
        if df is None or len(df) == 0:
            return {
                'time_last_scale': 1.0,
                'time_start_scale': 1.0,
                'time_matrix_scale': 1.0,
            }

        time_last = df['TimeSinceLast'].astype(float).to_numpy(dtype=np.float32)
        time_start = df['TimeSinceStart'].astype(float).to_numpy(dtype=np.float32)

        time_last_scale = max(float(np.std(time_last)), self.normalization_eps)
        time_start_scale = max(float(np.std(time_start)), self.normalization_eps)

        return {
            'time_last_scale': time_last_scale,
            'time_start_scale': time_start_scale,
            'time_matrix_scale': time_start_scale,
        }

    def get_normalization_stats(self):
        return dict(self.normalization_stats)

    def _finalize_variant_stats(self):
        if not self.prefixes:
            return

        keys = [item['variant_key'] for item in self.prefixes]
        counter = Counter(keys)

        self.variant2id = {key: idx for idx, key in enumerate(counter.keys())}
        self.variant_freq_table = {
            self.variant2id[key]: float(freq) for key, freq in counter.items()
        }

        for item in self.prefixes:
            variant_id = self.variant2id[item['variant_key']]
            item['variant_id'] = variant_id
            item['variant_freq'] = self.variant_freq_table[variant_id]

    def _generate_prefixes(self, df):
        print("正在生成滑动前缀序列 (Sliding Trace Prefixes)...")
        grouped = df.groupby('CaseID')

        for case_id, group in grouped:
            acts = group['Activity'].tolist()
            ress = group['Resource'].tolist()
            time_lasts = group['TimeSinceLast'].tolist()
            time_starts = group['TimeSinceStart'].tolist()

            next_acts = group['Next_Activity'].tolist()
            next_times = group['Next_Event_Time'].tolist()
            rem_times = group['Remaining_Time'].tolist()

            total_events = len(group)

            # 处理长工单的前缀数量控制，防止单条长序列导致前缀数量爆炸
            start_idx = 1
            if self.max_prefixes_per_case and total_events > self.max_prefixes_per_case:
                start_idx = total_events - self.max_prefixes_per_case + 1

            # 使用滑动窗口提取前缀
            for i in range(start_idx, total_events + 1):
                # 截取滑动窗口，最多保留 max_seq_len 个历史事件
                prefix_acts = acts[max(0, i - self.max_seq_len): i]
                prefix_ress = ress[max(0, i - self.max_seq_len): i]
                prefix_time_lasts = time_lasts[max(0, i - self.max_seq_len): i]
                prefix_time_starts = time_starts[max(0, i - self.max_seq_len): i]

                variant_key = '->'.join([str(a) for a in prefix_acts])

                target_act = next_acts[i - 1]
                target_next_time = next_times[i - 1]
                target_rem_time = rem_times[i - 1]

                self.prefixes.append({
                    'act_seq': [self.vocab.get_act_id(a) for a in prefix_acts],
                    'res_seq': [self.vocab.get_res_id(r) for r in prefix_ress],
                    'time_last_seq': prefix_time_lasts,
                    'time_start_seq': prefix_time_starts,
                    'target_act': self.vocab.get_act_id(target_act),
                    'target_next_time': target_next_time,
                    'target_rem_time': target_rem_time,
                    'variant_key': variant_key,
                })

        print(f" -> 成功生成了 {len(self.prefixes)} 个前缀样本。")

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        item = self.prefixes[idx]
        seq_len = len(item['act_seq'])
        pad_len = self.max_seq_len - seq_len

        time_last_valid = np.asarray(item['time_last_seq'], dtype=np.float32)
        time_start_valid = np.asarray(item['time_start_seq'], dtype=np.float32)

        # 使用训练集拟合的缩放系数进行标准化，消除不同数值特征量级差异。
        time_last_valid = time_last_valid / self.normalization_stats['time_last_scale']
        time_start_valid = time_start_valid / self.normalization_stats['time_start_scale']

        # 1. 序列填充 (Padding)
        act_seq = item['act_seq'] + [self.vocab.pad_idx] * pad_len
        res_seq = item['res_seq'] + [self.vocab.pad_idx] * pad_len
        time_last_seq = time_last_valid.tolist() + [0.0] * pad_len
        time_start_seq = time_start_valid.tolist() + [0.0] * pad_len

        # 注意掩码定义：1 表示有效数据，0 表示 Padding 填充
        mask = [1] * seq_len + [0] * pad_len
        time_features = np.column_stack((time_last_seq, time_start_seq))

        # 2. 动态生成 2D 时间特征矩阵
        time_matrix = np.zeros((self.max_seq_len, self.max_seq_len, 1), dtype=np.float32)

        # 只计算有效序列部分
        for i in range(seq_len):
            for j in range(seq_len):
                # 时间偏置：标准化后的两个事件绝对时间差
                time_matrix[i, j, 0] = abs(time_start_seq[i] - time_start_seq[j])

        return {
            'act_seq': torch.tensor(act_seq, dtype=torch.long),
            'res_seq': torch.tensor(res_seq, dtype=torch.long),
            'time_seq': torch.tensor(time_features, dtype=torch.float32),
            'time_matrix': torch.tensor(time_matrix, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.long),
            'variant_id': torch.tensor(item['variant_id'], dtype=torch.long),
            'variant_freq': torch.tensor(item['variant_freq'], dtype=torch.float32),

            'target_act': torch.tensor(item['target_act'], dtype=torch.long),
            'target_next_time': torch.tensor(item['target_next_time'], dtype=torch.float32),
            'target_rem_time': torch.tensor(item['target_rem_time'], dtype=torch.float32)
        }


if __name__ == '__main__':
    # ==========================================
    # 超参数与执行配置区域 (可直接在 PyCharm 中修改)
    # ==========================================

    # 输入文件路径 (必须是 dataprocessing.py 处理后的文件)
    INPUT_FILE_PATH = "dataset/processed_BPIC2020.csv"

    # 模型输入的最大序列长度
    MAX_SEQ_LENGTH = 20

    # 单个工单允许生成的最大前缀数量 (限制过长工单对显存的影响)
    MAX_PREFIXES = 100

    # DataLoader 的批次大小
    BATCH_SIZE = 16

    # ==========================================
    # 主执行逻辑
    # ==========================================
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"[错误] 找不到输入文件: {INPUT_FILE_PATH}")
    else:
        try:
            print(f"读取清洗后数据: {INPUT_FILE_PATH}")
            df = pd.read_csv(INPUT_FILE_PATH)

            vocab = ProcessVocab()
            vocab.build_vocab(df)

            dataset = ProcessPrefixDataset(
                df=df,
                vocab=vocab,
                max_seq_len=MAX_SEQ_LENGTH,
                max_prefixes_per_case=MAX_PREFIXES
            )

            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            for batch in dataloader:
                print("\n=== DataLoader Batch 测试示例 ===")
                print(f"活动序列 Shape: {batch['act_seq'].shape}")
                print(f"时间矩阵 (time_matrix) Shape: {batch['time_matrix'].shape}")
                print(f"掩码矩阵 (mask) Shape: {batch['mask'].shape}")
                break

        except Exception as e:
            print(f"运行出错: {e}")