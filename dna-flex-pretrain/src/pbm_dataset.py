import random
import torch
from torch.utils.data import Dataset

from src.tokenization import encode_sequence_onehot_6x4

class PBMDataset(Dataset):
    def __init__(self, path, k=6, seed=0):
        self.k = k
        self.items = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seq, score = line.split("\t")
                seq = seq.upper().strip()
                score = float(score)
                self.items.append((seq, score))

        rnd = random.Random(seed)
        rnd.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, y = self.items[idx]

        # onehot_flat: [L, 24], where L = len(seq) - k + 1
        _, onehot_flat, _ = encode_sequence_onehot_6x4(seq, k=self.k)

        x = onehot_flat  # [L,24]
        attention_mask = torch.ones((x.shape[0],), dtype=torch.long)

        return {
            "x": x,
            "attention_mask": attention_mask,
            "y": torch.tensor([y], dtype=torch.float32),
            "seq": seq,
        }

def split_dataset(ds, train_frac=0.8, val_frac=0.1):
    n = len(ds)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    idx_train = list(range(0, n_train))
    idx_val = list(range(n_train, n_train + n_val))
    idx_test = list(range(n_train + n_val, n))

    return idx_train, idx_val, idx_test
