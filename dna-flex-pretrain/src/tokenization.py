from itertools import product
import torch

from src.coordinates import TokenizedSequence, tokenize_with_coordinates

DNA_ALPHABET = ["A", "C", "G", "T"]

# Base-level one-hot mapping (4 channels)
BASE_TO_ONEHOT = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}

def tokenize_kmers(seq: str, k: int):
    """
    Convert DNA sequence into overlapping k-mers.
    Example: ACGTAC, k=3 -> [ACG, CGT, GTA, TAC]
    """
    seq = seq.upper()
    kmers = []
    for i in range(len(seq) - k + 1):
        kmers.append(seq[i:i+k])
    return kmers

def tokenize_kmers_with_coordinates(seq: str, k: int) -> TokenizedSequence:
    """Tokenize supported k values while retaining canonical coordinates."""

    return tokenize_with_coordinates(seq, k)

def build_kmer_vocab(k: int):
    """
    Keep the vocabulary too (still useful for labels / bookkeeping).
    [PAD], [CLS], [MASK], [UNK] + all possible DNA k-mers
    """
    specials = ["[PAD]", "[CLS]", "[MASK]", "[UNK]"]
    kmers = ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]
    tokens = specials + kmers
    stoi = {tok: i for i, tok in enumerate(tokens)}   # string -> id
    itos = {i: tok for tok, i in stoi.items()}         # id -> string
    return stoi, itos

def encode_sequence_to_ids(seq: str, k: int, stoi: dict, add_cls: bool = False):
    """
    ID encoding (still useful later for MLM labels).
    For now default add_cls=False because CLS is not a real 6-mer.
    """
    kmers = tokenize_kmers(seq, k)
    unk_id = stoi["[UNK]"]
    ids = [stoi.get(km, unk_id) for km in kmers]
    if add_cls:
        ids = [stoi["[CLS]"]] + ids
    return ids

def kmer_to_onehot_6x4(kmer: str):
    """
    Convert one k-mer (e.g. 6-mer) into a [k,4] binary matrix.
    For k=6, output shape is [6,4].
    If a base is unknown (e.g. N), that row becomes [0,0,0,0].
    """
    kmer = kmer.upper()
    rows = []
    for base in kmer:
        if base in BASE_TO_ONEHOT:
            rows.append(BASE_TO_ONEHOT[base])
        else:
            rows.append([0, 0, 0, 0])  # unknown base
    return torch.tensor(rows, dtype=torch.float32)  # [k, 4]

def encode_sequence_onehot_6x4(seq: str, k: int = 6):
    """
    Convert sequence -> overlapping k-mers -> stack of [k,4] matrices.

    Returns:
      onehot_3d: [num_kmers, k, 4]
      onehot_flat: [num_kmers, k*4]  (for feeding into a model)
      kmers: list of k-mer strings
    """
    kmers = tokenize_kmers(seq, k)
    mats = [kmer_to_onehot_6x4(km) for km in kmers]  # each [k,4]
    if len(mats) == 0:
        onehot_3d = torch.zeros((0, k, 4), dtype=torch.float32)
    else:
        onehot_3d = torch.stack(mats, dim=0)  # [L, k, 4]

    onehot_flat = onehot_3d.reshape(onehot_3d.shape[0], k * 4)  # [L, 24] when k=6
    return onehot_3d, onehot_flat, kmers
