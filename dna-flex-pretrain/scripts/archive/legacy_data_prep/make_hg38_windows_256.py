import os
import random
from Bio import SeqIO

def is_valid_window(s: str) -> bool:
    # Keep only A/C/G/T windows (no N)
    for ch in s:
        if ch not in ("A", "C", "G", "T"):
            return False
    return True

def main():
    random.seed(0)

    fasta_path = "data/raw/hg38.fa"   # <-- put hg38 fasta here
    out_path = "data/raw/hg38_windows_256.txt"

    window_bp = 256
    num_windows = 2000      # small starter set
    chromosomes = {"chr1"}  # start with chr1 only (fast debug)

    if not os.path.exists(fasta_path):
        raise FileNotFoundError(
            f"Missing {fasta_path}. Put hg38 fasta at data/raw/hg38.fa"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    collected = 0
    with open(out_path, "w") as out_f:
        for record in SeqIO.parse(fasta_path, "fasta"):
            name = record.id
            if name not in chromosomes:
                continue

            seq = str(record.seq).upper()
            L = len(seq)

            # sample random windows from this chromosome
            attempts = 0
            while collected < num_windows and attempts < num_windows * 50:
                start = random.randint(0, L - window_bp)
                w = seq[start:start + window_bp]
                attempts += 1

                if is_valid_window(w):
                    out_f.write(w + "\n")
                    collected += 1

            if collected >= num_windows:
                break

    print("wrote windows:", collected)
    print("output file:", out_path)

if __name__ == "__main__":
    main()
