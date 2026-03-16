import random
from src.utils import load_yaml

def main():
    cfg = load_yaml("configs/pretrain.yaml")
    seed = int(cfg.get("seed", 0))
    random.seed(seed)

    data = cfg["data"]
    window_bp = int(data["window_bp"])
    val_fraction = float(data.get("val_fraction", 0.1))

    in_path = f"data/raw/hg38_windows_{window_bp}.txt"
    train_path = f"data/raw/hg38_windows_{window_bp}_train.txt"
    val_path = f"data/raw/hg38_windows_{window_bp}_val.txt"

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

    print("window_bp =", window_bp)
    print("total =", n_total)
    print("train =", len(train_lines), "->", train_path)
    print("val   =", len(val_lines), "->", val_path)

if __name__ == "__main__":
    main()
