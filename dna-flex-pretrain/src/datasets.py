import torch
from torch.utils.data import Dataset

from src.tokenization import build_kmer_vocab, encode_sequence_to_ids, encode_sequence_onehot_6x4
from src.flex_features import (
    sequence_to_multi_dinuc_targets,
    sequence_to_multi_trinuc_targets,
)

def zscore_columns(targets):
    x = torch.tensor(targets, dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False)
    std[std == 0] = 1.0
    return (x - mean) / std

def pad_2d(x, target_len, pad_value=0.0):
    L, D = x.shape
    if L == target_len:
        return x
    pad = torch.full((target_len - L, D), pad_value, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

def pad_1d(x, target_len, pad_value=0):
    L = x.shape[0]
    if L == target_len:
        return x
    pad = torch.full((target_len - L,), pad_value, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

class TinyFlexDataset(Dataset):
    def __init__(self, sequences, k, lookup_data, dinuc_feature_names, trinuc_feature_names):
        self.sequences = sequences
        self.k = k
        self.lookup_data = lookup_data
        self.dinuc_feature_names = dinuc_feature_names
        self.trinuc_feature_names = trinuc_feature_names

        self.stoi, self.itos = build_kmer_vocab(k)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        onehot_3d, onehot_flat, kmers = encode_sequence_onehot_6x4(seq, k=self.k)
        L = onehot_flat.shape[0]

        token_ids = encode_sequence_to_ids(seq, self.k, self.stoi, add_cls=False)
        mlm_labels = torch.full((L,), -100, dtype=torch.long)

        mask_pos = min(2, L - 1)
        mlm_labels[mask_pos] = token_ids[mask_pos]

        x_masked = onehot_flat.clone()
        x_masked[mask_pos, :] = 0.0

        combined_targets = []

        if len(self.dinuc_feature_names) > 0:
            kmers_di, targets_di = sequence_to_multi_dinuc_targets(
                seq=seq,
                k=self.k,
                lookup_data=self.lookup_data,
                feature_names=self.dinuc_feature_names
            )
            assert kmers == kmers_di
        else:
            targets_di = [[] for _ in range(L)]

        if len(self.trinuc_feature_names) > 0:
            kmers_tri, targets_tri = sequence_to_multi_trinuc_targets(
                seq=seq,
                k=self.k,
                lookup_data=self.lookup_data,
                feature_names=self.trinuc_feature_names
            )
            assert kmers == kmers_tri
        else:
            targets_tri = [[] for _ in range(L)]

        for row_di, row_tri in zip(targets_di, targets_tri):
            combined_targets.append(row_di + row_tri)

        flex_targets = zscore_columns(combined_targets)
        attention_mask = torch.ones((L,), dtype=torch.long)

        return {
            "x": x_masked,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "flex_targets": flex_targets,
            "length": L,
            "seq": seq,
        }

def tinyflex_collate_fn(batch):
    max_len = max(item["length"] for item in batch)
    num_features = batch[0]["flex_targets"].shape[1]

    batch_x = []
    batch_attention_mask = []
    batch_mlm_labels = []
    batch_flex_targets = []
    batch_flex_valid = []
    batch_lengths = []
    batch_seqs = []

    for item in batch:
        L = item["length"]

        x_pad = pad_2d(item["x"], max_len, pad_value=0.0)
        attention_pad = pad_1d(item["attention_mask"], max_len, pad_value=0)
        labels_pad = pad_1d(item["mlm_labels"], max_len, pad_value=-100)
        flex_pad = pad_2d(item["flex_targets"], max_len, pad_value=0.0)

        flex_valid = torch.zeros((max_len, num_features), dtype=torch.bool)
        flex_valid[:L, :] = True

        batch_x.append(x_pad)
        batch_attention_mask.append(attention_pad)
        batch_mlm_labels.append(labels_pad)
        batch_flex_targets.append(flex_pad)
        batch_flex_valid.append(flex_valid)
        batch_lengths.append(L)
        batch_seqs.append(item["seq"])

    return {
        "x": torch.stack(batch_x, dim=0),
        "attention_mask": torch.stack(batch_attention_mask, dim=0),
        "mlm_labels": torch.stack(batch_mlm_labels, dim=0),
        "flex_targets": torch.stack(batch_flex_targets, dim=0),
        "flex_valid": torch.stack(batch_flex_valid, dim=0),
        "lengths": batch_lengths,
        "seqs": batch_seqs,
    }
