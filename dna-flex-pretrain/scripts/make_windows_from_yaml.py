import random
from Bio import SeqIO
from src.utils import load_yaml

def is_valid_window(s: str) -> bool:
    # keep only A/C/G/T windows
    for ch in s:
        if ch not in ("A", "C", "G", "T"):
            return False
    return True

def main():
    cfg = load_yaml("configs/pretrain.yaml")
    seed = int(cfg.get("seed", 0))
    random.seed(seed)

    data = cfg["data"]
    fasta_path = data["fasta_path"]
    window_bp = int(data["window_bp"])
    num_windows = int(data["num_windows"])
    chroms = set(data["chromosomes"])

    out_path = f"data/raw/hg38_windows_{window_bp}.txt"

    # load requested chromosomes into memory (chr1 only is fine)
    chrom_seq = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        if rec.id in chroms:
            chrom_seq[rec.id] = str(rec.seq).upper()

    if len(chrom_seq) == 0:
        raise ValueError(f"No chromosomes found from {chroms}. Check fasta headers (e.g., chr1).")

    chrom_list = list(chrom_seq.keys())

    collected = 0
    attempts = 0
    max_attempts = num_windows * 500  # high to skip N-rich regions

    with open(out_path, "w") as f:
        while collected < num_windows and attempts < max_attempts:
            attempts += 1
            c = random.choice(chrom_list)
            seq = chrom_seq[c]
            L = len(seq)
            start = random.randint(0, L - window_bp)
            w = seq[start:start + window_bp]
            if is_valid_window(w):
                f.write(w + "\n")
                collected += 1

    print("window_bp =", window_bp)
    print("num_windows requested =", num_windows)
    print("num_windows wrote =", collected)
    print("output file =", out_path)

if __name__ == "__main__":
    main()
