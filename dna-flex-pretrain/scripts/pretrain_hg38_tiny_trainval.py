import os
import csv
import time
import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import load_yaml
from src.flex_features import load_lookup_yaml
from src.genome_dataset import GenomeWindowDataset
from src.collate import genome_collate_fn
from src.model import TinyMultiTaskModelOneHot

LAST_CKPT = "checkpoints/hg38_256_chr1-22_200k_di8_tri4_last.pt"
BEST_CKPT = "checkpoints/hg38_256_chr1-22_200k_di8_tri4_best_by_val_flex.pt"
LOG_CSV  = "logs/hg38_256_chr1-22_200k_di8_tri4_log.csv"

def ensure_dirs():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def save_ckpt(path, model, opt, step, cfg, best_val_flex):
    torch.save(
        {
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "step": step,
            "config": cfg,
            "best_val_flex": best_val_flex,
        },
        path
    )

def load_ckpt_if_exists(model, opt, path):
    if not os.path.exists(path):
        return 0, None
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    opt.load_state_dict(ckpt["opt_state"])
    return int(ckpt["step"]), ckpt.get("best_val_flex", None)

@torch.no_grad()
def eval_losses(model, loader, max_batches=None):
    model.eval()
    mlm_list = []
    flex_list = []
    n = 0
    for batch in loader:
        n += 1
        if (max_batches is not None) and (n > max_batches):
            break

        x = batch["x"]
        attention_mask = batch["attention_mask"]
        mlm_labels = batch["mlm_labels"]
        flex_targets = batch["flex_targets"]
        flex_valid = batch["flex_valid"]

        mlm_logits, flex_pred = model(x, attention_mask)

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100
        )

        flex_loss = F.huber_loss(
            flex_pred[flex_valid],
            flex_targets[flex_valid],
            delta=1.0
        )

        mlm_list.append(float(mlm_loss.item()))
        flex_list.append(float(flex_loss.item()))

    return sum(mlm_list)/len(mlm_list), sum(flex_list)/len(flex_list)

def append_log(row):
    new_file = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "time", "step",
                "train_total", "train_mlm", "train_flex",
                "val_mlm", "val_flex",
                "best_val_flex", "improved"
            ])
        w.writerow(row)

def main():
    ensure_dirs()
    torch.manual_seed(0)

    cfg = load_yaml("configs/pretrain.yaml")

    window_bp = int(cfg['data']['window_bp'])

    k = cfg["tokenizer"]["k"]

    d_model = cfg["model"]["d_model"]
    n_heads = cfg["model"]["n_heads"]
    n_layers = cfg["model"]["n_layers"]
    max_len = cfg["model"]["max_len"]

    tr = cfg["train"]
    lambda_mlm = float(tr["lambda_mlm"])
    lambda_flex = float(tr["lambda_flex"])
    lr = float(tr["lr"])
    weight_decay = float(tr["weight_decay"])
    mlm_prob = float(tr["mlm_prob"])
    steps_per_run = int(tr.get("steps_per_run", 200))
    log_every = int(tr.get("log_every", 50))
    eval_every = int(tr.get("eval_every", 500))
    val_max_batches = int(tr.get("val_max_batches", 50))

    dinuc_feats = cfg["features"]["dinucleotide"]
    trinuc_feats = cfg["features"]["trinucleotide"]
    n_flex = len(dinuc_feats) + len(trinuc_feats)

    lookup = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    train_ds = GenomeWindowDataset(
        window_txt_path=f"data/raw/hg38_windows_{window_bp}_train.txt",
        k=k,
        lookup_data=lookup,
        dinuc_feature_names=dinuc_feats,
        trinuc_feature_names=trinuc_feats,
        mlm_prob=mlm_prob,
        seed=0,
        norm_stats_path="data/processed/flex_norm_stats.yaml"
    )

    val_ds = GenomeWindowDataset(
        window_txt_path=f"data/raw/hg38_windows_{window_bp}_val.txt",
        k=k,
        lookup_data=lookup,
        dinuc_feature_names=dinuc_feats,
        trinuc_feature_names=trinuc_feats,
        mlm_prob=mlm_prob,
        seed=1,
        norm_stats_path="data/processed/flex_norm_stats.yaml"
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=genome_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=genome_collate_fn)

    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=len(train_ds.stoi),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
        n_flex=n_flex
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    step, best_val_flex = load_ckpt_if_exists(model, opt, LAST_CKPT)
    if best_val_flex is None:
        best_val_flex = float("inf")

    print(f"starting | n_flex={n_flex} | steps_per_run={steps_per_run} | log_every={log_every} | eval_every={eval_every}")
    print(f"lambda_mlm={lambda_mlm} lambda_flex={lambda_flex} mlm_prob={mlm_prob}")
    print("dinuc_feats =", dinuc_feats)
    print("trinuc_feats =", trinuc_feats)

    if step > 0:
        print(f"resumed from checkpoint at step {step} | best_val_flex = {best_val_flex:.6f}")

    # cycle so we can do an arbitrary number of steps
    train_iter = itertools.cycle(train_loader)

    # running sums for nicer printing
    run_total = 0.0
    run_mlm = 0.0
    run_flex = 0.0
    count = 0

    end_step = step + steps_per_run
    while step < end_step:
        step += 1
        batch = next(train_iter)

        x = batch["x"]
        attention_mask = batch["attention_mask"]
        mlm_labels = batch["mlm_labels"]
        flex_targets = batch["flex_targets"]
        flex_valid = batch["flex_valid"]

        mlm_logits, flex_pred = model(x, attention_mask)

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100
        )

        flex_loss = F.huber_loss(
            flex_pred[flex_valid],
            flex_targets[flex_valid],
            delta=1.0
        )

        total_loss = lambda_mlm * mlm_loss + lambda_flex * flex_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        run_total += float(total_loss.item())
        run_mlm += float(mlm_loss.item())
        run_flex += float(flex_loss.item())
        count += 1

        if step % log_every == 0:
            avg_total = run_total / count
            avg_mlm = run_mlm / count
            avg_flex = run_flex / count
            print(f"step {step} | train_total={avg_total:.4f} train_mlm={avg_mlm:.4f} train_flex={avg_flex:.4f}")

        # periodic validation
        if step % eval_every == 0:
            val_mlm, val_flex = eval_losses(model, val_loader, max_batches=val_max_batches)

            improved = False
            if val_flex < best_val_flex:
                best_val_flex = val_flex
                save_ckpt(BEST_CKPT, model, opt, step, cfg, best_val_flex)
                improved = True

            save_ckpt(LAST_CKPT, model, opt, step, cfg, best_val_flex)

            # log row
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            avg_total = run_total / count
            avg_mlm = run_mlm / count
            avg_flex = run_flex / count
            append_log([now, step, avg_total, avg_mlm, avg_flex, val_mlm, val_flex, best_val_flex, improved])

            print(f"  VAL @ step {step} | val_mlm={val_mlm:.4f} val_flex={val_flex:.6f} best_val_flex={best_val_flex:.6f} improved={improved}")

            # reset running stats after eval so each segment is readable
            run_total = run_mlm = run_flex = 0.0
            count = 0

    # final save
    save_ckpt(LAST_CKPT, model, opt, step, cfg, best_val_flex)
    print("done. last ckpt =", LAST_CKPT)
    print("best ckpt =", BEST_CKPT)
    print("log file =", LOG_CSV)

if __name__ == "__main__":
    main()
