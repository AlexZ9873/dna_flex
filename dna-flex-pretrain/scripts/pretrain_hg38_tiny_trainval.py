import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import load_yaml
from src.flex_features import load_lookup_yaml
from src.genome_dataset import GenomeWindowDataset
from src.collate import genome_collate_fn
from src.model import TinyMultiTaskModelOneHot

LAST_CKPT = "checkpoints/hg38_tiny_trainval_last.pt"
BEST_CKPT = "checkpoints/hg38_tiny_trainval_best_by_val_flex.pt"

def save_ckpt(path, model, opt, step, cfg, best_val_flex=None):
    os.makedirs("checkpoints", exist_ok=True)
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
def evaluate(model, loader):
    model.eval()
    mlm_losses = []
    flex_losses = []

    for batch in loader:
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

        mlm_losses.append(float(mlm_loss.item()))
        flex_losses.append(float(flex_loss.item()))

    return (
        sum(mlm_losses) / len(mlm_losses),
        sum(flex_losses) / len(flex_losses),
    )

def main():
    torch.manual_seed(0)

    cfg = load_yaml("configs/pretrain.yaml")

    k = cfg["tokenizer"]["k"]
    d_model = cfg["model"]["d_model"]
    n_heads = cfg["model"]["n_heads"]
    n_layers = cfg["model"]["n_layers"]
    max_len = cfg["model"]["max_len"]

    lambda_mlm = cfg["train"]["lambda_mlm"]
    lambda_flex = cfg["train"]["lambda_flex"]
    lr = cfg["train"]["lr"]
    weight_decay = cfg["train"]["weight_decay"]
    mlm_prob = cfg["train"]["mlm_prob"]
    steps_per_run = int(cfg["train"].get("steps_per_run", 200))

    dinuc_feats = cfg["features"]["dinucleotide"]
    trinuc_feats = cfg["features"]["trinucleotide"]
    n_flex = len(dinuc_feats) + len(trinuc_feats)

    lookup = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    train_ds = GenomeWindowDataset(
        window_txt_path="data/raw/hg38_windows_256_train.txt",
        k=k,
        lookup_data=lookup,
        dinuc_feature_names=dinuc_feats,
        trinuc_feature_names=trinuc_feats,
        max_rows=None,
        mlm_prob=mlm_prob,
        seed=0,
        norm_stats_path="data/processed/flex_norm_stats.yaml"
    )

    val_ds = GenomeWindowDataset(
        window_txt_path="data/raw/hg38_windows_256_val.txt",
        k=k,
        lookup_data=lookup,
        dinuc_feature_names=dinuc_feats,
        trinuc_feature_names=trinuc_feats,
        max_rows=None,
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

    start_step, best_val_flex = load_ckpt_if_exists(model, opt, LAST_CKPT)
    if best_val_flex is None:
        best_val_flex = float("inf")

    if start_step > 0:
        print(f"resumed from checkpoint at step {start_step} | best_val_flex = {best_val_flex:.6f}")

    global_step = start_step
    train_total = []
    train_mlm = []
    train_flex = []

    for batch in train_loader:
        global_step += 1

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

        train_total.append(float(total_loss.item()))
        train_mlm.append(float(mlm_loss.item()))
        train_flex.append(float(flex_loss.item()))

        if global_step >= start_step + steps_per_run:
            break

    # Evaluate
    val_mlm, val_flex = evaluate(model, val_loader)

    # Save "best by val flex" (update best_val_flex first)
    improved = False
    if val_flex < best_val_flex:
        best_val_flex = val_flex
        save_ckpt(BEST_CKPT, model, opt, global_step, cfg, best_val_flex=best_val_flex)
        improved = True

    # Save "last" AFTER updating best_val_flex
    save_ckpt(LAST_CKPT, model, opt, global_step, cfg, best_val_flex=best_val_flex)

    print()
    print("mlm_prob =", mlm_prob)
    print("lambda_mlm =", lambda_mlm)
    print("lambda_flex =", lambda_flex)
    print("train_total_loss =", sum(train_total)/len(train_total))
    print("train_mlm_loss =", sum(train_mlm)/len(train_mlm))
    print("train_flex_loss =", sum(train_flex)/len(train_flex))
    print("val_mlm_loss =", val_mlm)
    print("val_flex_loss =", val_flex)
    print("saved last =", LAST_CKPT)
    print("saved best =", BEST_CKPT)
    print("best_val_flex =", best_val_flex)
    print("improved =", improved)

if __name__ == "__main__":
    main()
