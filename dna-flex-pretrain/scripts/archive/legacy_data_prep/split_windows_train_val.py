import random

def main():
    random.seed(0)

    in_path = "data/raw/hg38_windows_256.txt"
    train_path = "data/raw/hg38_windows_256_train.txt"
    val_path = "data/raw/hg38_windows_256_val.txt"

    val_fraction = 0.1  # 10% validation

    with open(in_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    random.shuffle(lines)

    n_total = len(lines)
    n_val = int(n_total * val_fraction)
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    with open(train_path, "w") as f:
        for s in train_lines:
            f.write(s + "\n")

    with open(val_path, "w") as f:
        for s in val_lines:
            f.write(s + "\n")

    print("total =", n_total)
    print("train =", len(train_lines), "->", train_path)
    print("val   =", len(val_lines), "->", val_path)

if __name__ == "__main__":
    main()
