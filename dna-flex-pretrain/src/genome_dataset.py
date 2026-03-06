import random
import yaml
import torch
from torch.utils.data import Dataset

from src.tokenization import build_kmer_vocab, encode_sequence_to_ids, encode_sequence_onehot_6x4
from src.flex_features import (
    sequence_to_multi_dinuc_targets,
    sequence_to_multi_trinuc_targets,
)

def load_norm_stats(path: str):
    with open(path, "r") as f:
        stats = yaml.safe_load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    feature_names = stats["feature_names"]
    return feature_names, mean, std

class GenomeWindowDataset(Dataset):
    def __init__(
        self,
        window_txt_path,
        k,
        lookup_data,
        dinuc_feature_names,
        trinuc_feature_names,
        max_rows=None,
        mlm_prob=0.15,
        seed=0,
        norm_stats_path=None,
    ):
        self.window_txt_path = window_txt_path
        self.k = k
        self.lookup_data = lookup_data
        self.dinuc_feature_names = dinuc_feature_names
        self.trinuc_feature_names = trinuc_feature_names
        self.mlm_prob = mlm_prob
        self.rng = random.Random(seed)

        self.feature_names = dinuc_feature_names + trinuc_feature_names

        self.norm_mean = None
        self.norm_std = None
        if norm_stats_path is not None:
            stats_features, mean, std = load_norm_stats(norm_stats_path)
            # safety check: ensure order matches
            assert stats_features == self.feature_names, (
                "Feature order mismatch between config and norm stats.\n"
                f"config: {self.feature_names}\n"
                f"stats:  {stats_features}"
            )
            self.norm_mean = mean
            self.norm_std = std

        self.stoi, self.itos = build_kmer_vocab(k)

        self.seqs = []
        with open(window_txt_path, "r") as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                s = line.strip().upper()
                if s:
                    self.seqs.append(s)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]

        onehot_3d, onehot_flat, kmers = encode_sequence_onehot_6x4(seq, k=self.k)
        L = onehot_flat.shape[0]

        token_ids = encode_sequence_to_ids(seq, self.k, self.stoi, add_cls=False)
        mlm_labels = torch.full((L,), -100, dtype=torch.long)

        mask_positions = []
        for pos in range(L):
            if self.rng.random() < self.mlm_prob:
                mask_positions.append(pos)
        if len(mask_positions) == 0:
            mask_positions.append(self.rng.randrange(L))

        x_masked = onehot_flat.clone()
        for pos in mask_positions:
            mlm_labels[pos] = token_ids[pos]
            x_masked[pos, :] = 0.0

        if len(self.dinuc_feature_names) > 0:
            kmers_di, targets_di = sequence_to_multi_dinuc_targets(
                seq=seq, k=self.k, lookup_data=self.lookup_data, feature_names=self.dinuc_feature_names
            )
            assert kmers == kmers_di
        else:
            targets_di = [[] for _ in range(L)]

        if len(self.trinuc_feature_names) > 0:
            kmers_tri, targets_tri = sequence_to_multi_trinuc_targets(
                seq=seq, k=self.k, lookup_data=self.lookup_data, feature_names=self.trinuc_feature_names
            )
            assert kmers == kmers_tri
        else:
            targets_tri = [[] for _ in range(L)]

        combined_targets = [row_di + row_tri for row_di, row_tri in zip(targets_di, targets_tri)]
        flex = torch.tensor(combined_targets, dtype=torch.float32)  # [L, F]

        # GLOBAL normalization if provided, else fallback to per-window zscore
        if self.norm_mean is not None:
            flex = (flex - self.norm_mean) / self.norm_std
        else:
            m = flex.mean(dim=0, keepdim=True)
            s = flex.std(dim=0, keepdim=True, unbiased=False)
            s[s == 0] = 1.0
            flex = (flex - m) / s

        attention_mask = torch.ones((L,), dtype=torch.long)

        return {
            "x": x_masked,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "flex_targets": flex,
            "length": L,
            "seq": seq,
        }
