from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import yaml

BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


def load_lookup_tables(
    lookup_yaml: str = "data/raw/flex_tables/lookup.yaml",
    config_yaml: Optional[str] = "configs/pretrain.yaml",
):
    """
    Load di- and trinucleotide lookup tables.
    If config_yaml is provided and exists, use the feature order from the config so the
    flex target ordering matches your existing pretrained flex head.
    """
    lookup_yaml = Path(lookup_yaml)
    if not lookup_yaml.exists():
        raise FileNotFoundError(f"lookup yaml not found: {lookup_yaml}")

    data = yaml.safe_load(lookup_yaml.read_text())

    if "dinucleotide" not in data or "trinucleotide" not in data:
        raise ValueError(
            f"{lookup_yaml} must contain top-level keys 'dinucleotide' and 'trinucleotide'"
        )

    di_block = data["dinucleotide"]
    tri_block = data["trinucleotide"]

    # Default feature order: YAML insertion order
    di_names = list(di_block.keys())
    tri_names = list(tri_block.keys())

    # If pretrain config exists, prefer its order for compatibility with current flex head
    if config_yaml is not None:
        config_yaml = Path(config_yaml)
        if config_yaml.exists():
            cfg = yaml.safe_load(config_yaml.read_text())
            try:
                di_names = list(cfg["features"]["dinucleotide"])
                tri_names = list(cfg["features"]["trinucleotide"])
            except Exception:
                pass

    # Ensure all config-requested names exist in the lookup table
    missing_di = [name for name in di_names if name not in di_block]
    missing_tri = [name for name in tri_names if name not in tri_block]
    if missing_di or missing_tri:
        raise ValueError(
            "Feature names requested by config are missing in lookup table. "
            f"Missing dinucleotide: {missing_di}, missing trinucleotide: {missing_tri}"
        )

    di_maps = {
        name: {k.upper(): float(v) for k, v in di_block[name].items()}
        for name in di_names
    }
    tri_maps = {
        name: {k.upper(): float(v) for k, v in tri_block[name].items()}
        for name in tri_names
    }

    return di_names, di_maps, tri_names, tri_maps


def encode_kmer_onehot(kmer: str) -> torch.Tensor:
    """
    Encode one k-mer as flattened kx4 one-hot vector.
    For k=6, output shape is [24].
    """
    kmer = kmer.upper()
    k = len(kmer)
    x = torch.zeros(k * 4, dtype=torch.float32)

    for i, ch in enumerate(kmer):
        j = BASE_TO_ID.get(ch, None)
        if j is not None:
            x[i * 4 + j] = 1.0

    return x


def compute_token_flex_target(
    kmer: str,
    di_names: List[str],
    di_maps: Dict[str, Dict[str, float]],
    tri_names: List[str],
    tri_maps: Dict[str, Dict[str, float]],
) -> torch.Tensor:
    """
    Build one 12-dimensional flex target for a single token (usually a 6-mer).

    IMPORTANT:
    - For each dinucleotide feature: average across all internal 2-mers of this token
    - For each trinucleotide feature: average across all internal 3-mers of this token

    Output ordering:
    - all dinucleotide features first
    - then all trinucleotide features

    This should match the ordering used when your pretrained flex head was trained.
    """
    kmer = kmer.upper()
    k = len(kmer)

    vals: List[float] = []

    # Dinucleotide features
    if k >= 2:
        for feat in di_names:
            fmap = di_maps[feat]
            local_vals = []
            for i in range(k - 1):
                word = kmer[i : i + 2]
                local_vals.append(float(fmap.get(word, 0.0)))
            vals.append(sum(local_vals) / len(local_vals))
    else:
        vals.extend([0.0] * len(di_names))

    # Trinucleotide features
    if k >= 3:
        for feat in tri_names:
            fmap = tri_maps[feat]
            local_vals = []
            for i in range(k - 2):
                word = kmer[i : i + 3]
                local_vals.append(float(fmap.get(word, 0.0)))
            vals.append(sum(local_vals) / len(local_vals))
    else:
        vals.extend([0.0] * len(tri_names))

    return torch.tensor(vals, dtype=torch.float32)


def sequence_to_token_tensors(
    seq: str,
    k: int,
    di_names: List[str],
    di_maps: Dict[str, Dict[str, float]],
    tri_names: List[str],
    tri_maps: Dict[str, Dict[str, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert one sequence into:
      x            : [T, k*4] token one-hot input
      flex_targets : [T, n_flex]
    """
    seq = seq.upper()
    T = len(seq) - k + 1
    if T <= 0:
        raise ValueError(f"Sequence length {len(seq)} is shorter than k={k}: {seq}")

    x_list = []
    flex_list = []

    for t in range(T):
        kmer = seq[t : t + k]
        x_list.append(encode_kmer_onehot(kmer))
        flex_list.append(
            compute_token_flex_target(kmer, di_names, di_maps, tri_names, tri_maps)
        )

    x = torch.stack(x_list, dim=0)               # [T, k*4]
    flex_targets = torch.stack(flex_list, dim=0) # [T, n_flex]
    return x, flex_targets


class BendabilityDataset(Dataset):
    """
    Dataset for BendNet-style bendability files.

    Expected row format:
        sequence   bendability_score   optional_group

    Example:
        TGGAACACGCACTTGACATTCTAGATGCTAATTGGTCAAAAACGGATTT   -2.780076   8
    """
    def __init__(
        self,
        path: str,
        k: int = 6,
        lookup_yaml: str = "data/raw/flex_tables/lookup.yaml",
        config_yaml: Optional[str] = "configs/pretrain.yaml",
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"bendability file not found: {self.path}")

        self.k = k
        self.di_names, self.di_maps, self.tri_names, self.tri_maps = load_lookup_tables(
            lookup_yaml=lookup_yaml,
            config_yaml=config_yaml,
        )

        self.rows: List[Tuple[str, float, Optional[str]]] = []

        with open(self.path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                seq = parts[0].upper()
                try:
                    score = float(parts[1])
                except ValueError:
                    continue

                group = parts[2] if len(parts) >= 3 else None
                self.rows.append((seq, score, group))

        if len(self.rows) == 0:
            raise ValueError(f"No usable rows found in {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        seq, score, group = self.rows[idx]

        x, flex_targets = sequence_to_token_tensors(
            seq=seq,
            k=self.k,
            di_names=self.di_names,
            di_maps=self.di_maps,
            tri_names=self.tri_names,
            tri_maps=self.tri_maps,
        )

        attention_mask = torch.ones(x.shape[0], dtype=torch.long)

        return {
            "seq": seq,
            "x": x,                                  # [T, 24] when k=6
            "attention_mask": attention_mask,        # [T]
            "flex_targets": flex_targets,            # [T, 12]
            "bendability": torch.tensor([score], dtype=torch.float32),  # [1]
            "group": group,                          # keep for metadata only
        }


def collate_bendability_batch(batch: List[dict]) -> dict:
    """
    Pad a batch of bendability examples.
    """
    B = len(batch)
    maxT = max(item["x"].shape[0] for item in batch)
    D = batch[0]["x"].shape[1]
    F = batch[0]["flex_targets"].shape[1]

    x = torch.zeros((B, maxT, D), dtype=torch.float32)
    attention_mask = torch.zeros((B, maxT), dtype=torch.long)
    flex_targets = torch.zeros((B, maxT, F), dtype=torch.float32)
    bendability = torch.zeros((B, 1), dtype=torch.float32)

    seqs: List[str] = []
    groups: List[Optional[str]] = []

    for i, item in enumerate(batch):
        T = item["x"].shape[0]
        x[i, :T] = item["x"]
        attention_mask[i, :T] = 1
        flex_targets[i, :T] = item["flex_targets"]
        bendability[i] = item["bendability"]
        seqs.append(item["seq"])
        groups.append(item["group"])

    return {
        "x": x,
        "attention_mask": attention_mask,
        "flex_targets": flex_targets,
        "bendability": bendability,
        "seq": seqs,
        "group": groups,
    }


def load_bendability_splits(
    split_dir: str = "data/raw/bendability/Data1",
    k: int = 6,
    lookup_yaml: str = "data/raw/flex_tables/lookup.yaml",
    config_yaml: Optional[str] = "configs/pretrain.yaml",
):
    """
    Convenience helper for Data1 / Data2 folder structure.

    IMPORTANT: upstream dataset uses 'vaild_set.txt' spelling.
    """
    split_dir = Path(split_dir)
    train_path = split_dir / "train_set.txt"
    valid_path = split_dir / "vaild_set.txt"
    test_path = split_dir / "test_set.txt"

    train_ds = BendabilityDataset(
        train_path, k=k, lookup_yaml=lookup_yaml, config_yaml=config_yaml
    )
    valid_ds = BendabilityDataset(
        valid_path, k=k, lookup_yaml=lookup_yaml, config_yaml=config_yaml
    )
    test_ds = BendabilityDataset(
        test_path, k=k, lookup_yaml=lookup_yaml, config_yaml=config_yaml
    )

    return train_ds, valid_ds, test_ds
