from itertools import product

DNA_ALPHABET = ["A", "C", "G", "T"]

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

def build_kmer_vocab(k: int):
    """
    Build a fixed vocabulary:
    special tokens + all possible DNA k-mers.
    """
    specials = ["[PAD]", "[CLS]", "[MASK]", "[UNK]"]
    kmers = ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]
    tokens = specials + kmers
    stoi = {tok: i for i, tok in enumerate(tokens)}  # string -> id
    itos = {i: tok for tok, i in stoi.items()}        # id -> string
    return stoi, itos

def encode_sequence(seq: str, k: int, stoi: dict, add_cls: bool = True):
    """
    Convert sequence -> token IDs using the vocabulary.
    Unknown kmers map to [UNK].
    """
    kmers = tokenize_kmers(seq, k)
    unk_id = stoi["[UNK]"]
    ids = [stoi.get(km, unk_id) for km in kmers]
    if add_cls:
        ids = [stoi["[CLS]"]] + ids
    return ids
