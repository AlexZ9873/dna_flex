import os
import random
from Bio import SeqIO

def is_valid_window(s: str) -> bool:
    for ch in s:
        if ch not in ("A", "C", "G", "T"):
            return False
    return True

def main():
    random.seed(0)

    fasta_path = "data/raw/hg38.fa"
    out_path = "data/raw/hg38_windows_256.txt"

    window_bp = 256
    num_windows = 50000
    chromosomes = {"chr1"}

    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Missing {fasta_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    collected = 0
    with open(out_path, "w") as out_f:
        for record in SeqIO.parse(fasta_path, "fasta"):
            if record.id not in chromosomes:
                continue

            seq = str(record.seq).upper()
            L = len(seq)

            attempts = 0
            max_attempts = num_windows * 400  # high enough to skip N-rich regions

            while collected < num_windows and attempts < max_attempts:
                start = random.randint(0, L - window_bp)
                w = seq[start:start + window_bp]
                attempts += 1
                if is_valid_window(w):
                    out_f.write(w + "\n")
                    collected += 1

            break

    print("wrote windows:", collected)
    print("output file:", out_path)

if __name__ == "__main__":
    main()
