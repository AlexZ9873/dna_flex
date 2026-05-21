import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import load_yaml
from src.flex_features import load_lookup_yaml
from src.datasets import TinyFlexDataset, tinyflex_collate_fn
from src.model import TinyMultiTaskModelOneHot

def evaluate(model, loader, lambda_mlm, lambda_flex):
    model.eval()
    total = 0.0
    count = 0

    with torch.no_grad():
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

            total_loss = lambda_mlm * mlm_loss + lambda_flex * flex_loss
            total += float(total_loss.item())
            count += 1

    return total / count

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

    dinuc_feature_names = cfg["features"]["dinucleotide"]
    trinuc_feature_names = cfg["features"]["trinucleotide"]

    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    train_sequences = [
        "ACGTACGTAA",
        "TGCATGCAAATG",
        "GGGAAACCCC",
        "ATATCGCGTACGTA",
        "CGTACGTTAAAC",
        "TTTGGGCCCAAAT",
        "AACCGGTTAACC",
        "GCGTATATGCGT",
        "AAAACCCCGGGG",
        "TATATATACGCG",
        "CCGGAATTCCGG",
        "GATCGATCGATC",
        "AGCTAGCTAGCT",
        "TGCATGCATGCA",
        "GGATCCGGATCC",
        "CATGCATGCATG",
        "ACACACGTGTGT",
        "GTGTGTACACAC",
        "AAGGTTCCAAGG",
        "CCTTAAGGCCTT",
    ]

    val_sequences = [
        "ATGCATGCATGC",
        "CGCGTATATATA",
        "TTAACCGGTTAA",
        "GGGCCCATAATA",
        "ACGCGTTAACGG",
    ]

    train_dataset = TinyFlexDataset(
        sequences=train_sequences,
        k=k,
        lookup_data=lookup_data,
        dinuc_feature_names=dinuc_feature_names,
        trinuc_feature_names=trinuc_feature_names,
    )

    val_dataset = TinyFlexDataset(
        sequences=val_sequences,
        k=k,
        lookup_data=lookup_data,
        dinuc_feature_names=dinuc_feature_names,
        trinuc_feature_names=trinuc_feature_names,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=tinyflex_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=tinyflex_collate_fn
    )

    n_flex = len(dinuc_feature_names) + len(trinuc_feature_names)

    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=len(train_dataset.stoi),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
        n_flex=n_flex
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    os.makedirs("checkpoints", exist_ok=True)

    train_losses = []
    val_losses = []

    num_epochs = 10
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
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

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_train_loss += float(total_loss.item())
            num_batches += 1

        avg_train_loss = running_train_loss / num_batches
        avg_val_loss = evaluate(model, val_loader, lambda_mlm=lambda_mlm, lambda_flex=lambda_flex)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"epoch {epoch+1:02d} | train_loss = {avg_train_loss:.4f} | val_loss = {avg_val_loss:.4f}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                "checkpoints/best_tinyflex_model.pt"
            )

    print()
    print("d_model =", d_model)
    print("n_heads =", n_heads)
    print("n_layers =", n_layers)
    print("max_len =", max_len)
    print("dinuc_feature_names =", dinuc_feature_names)
    print("trinuc_feature_names =", trinuc_feature_names)
    print("lambda_mlm =", lambda_mlm)
    print("lambda_flex =", lambda_flex)
    print("lr =", lr)
    print("weight_decay =", weight_decay)
    print("n_flex =", n_flex)
    print("start train loss =", train_losses[0])
    print("end train loss =", train_losses[-1])
    print("best train loss =", min(train_losses))
    print("start val loss =", val_losses[0])
    print("end val loss =", val_losses[-1])
    print("best val loss =", min(val_losses))
    print("best epoch =", best_epoch)
    print("checkpoint saved to = checkpoints/best_tinyflex_model.pt")
    print("best-checkpoint train/val epoch loop complete")

if __name__ == "__main__":
    main()
