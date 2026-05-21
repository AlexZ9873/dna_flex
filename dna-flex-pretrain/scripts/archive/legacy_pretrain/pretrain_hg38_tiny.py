import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import load_yaml
from src.flex_features import load_lookup_yaml
from src.genome_dataset import GenomeWindowDataset
from src.collate import genome_collate_fn
from src.model import TinyMultiTaskModelOneHot

CKPT_PATH = "checkpoints/hg38_tiny_last.pt"

def save_ckpt(model, opt, step, cfg):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "step": step,
            "config": cfg,
        },
        CKPT_PATH
    )

def load_ckpt_if_exists(model, opt):
    if not os.path.exists(CKPT_PATH):
        return 0
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    opt.load_state_dict(ckpt["opt_state"])
    return int(ckpt["step"])

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

    dinuc_feats = cfg["features"]["dinucleotide"]
    trinuc_feats = cfg["features"]["trinucleotide"]
    n_flex = len(dinuc_feats) + len(trinuc_feats)

    lookup = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    dataset = GenomeWindowDataset(
        window_txt_path="data/raw/hg38_windows_256.txt",
        k=k,
        lookup_data=lookup,
        dinuc_feature_names=dinuc_feats,
        trinuc_feature_names=trinuc_feats,
        max_rows=200
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=genome_collate_fn)

    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=len(dataset.stoi),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
        n_flex=n_flex
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_step = load_ckpt_if_exists(model, opt)
    if start_step > 0:
        print(f"resumed from checkpoint at step {start_step}")

    losses = []
    global_step = start_step

    for batch in loader:
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

        loss = lambda_mlm * mlm_loss + lambda_flex * flex_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))

        if global_step % 10 == 0:
            print(f"step {global_step:03d} | loss = {loss.item():.4f} | mlm = {mlm_loss.item():.4f} | flex = {flex_loss.item():.4f}")
            save_ckpt(model, opt, global_step, cfg)

    # Save at end too
    save_ckpt(model, opt, global_step, cfg)

    print()
    print("avg loss =", sum(losses)/len(losses))
    print("min loss =", min(losses))
    print("max loss =", max(losses))
    print("saved checkpoint =", CKPT_PATH)
    print("hg38 tiny pretrain with checkpoint complete")

if __name__ == "__main__":
    main()
