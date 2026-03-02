import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import load_yaml
from src.flex_features import load_lookup_yaml
from src.datasets import TinyFlexDataset, tinyflex_collate_fn
from src.model import TinyMultiTaskModelOneHot


def evaluate(model, loader):
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

            total_loss = mlm_loss + 0.5 * flex_loss
            total += float(total_loss.item())
            count += 1

    return total / count


def main():
    torch.manual_seed(0)

    # 1) Load config
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    # 2) Load lookup data
    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    # 3) Larger toy train/validation split
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

    # 4) Build datasets
    train_dataset = TinyFlexDataset(
        sequences=train_sequences,
        k=k,
        lookup_data=lookup_data
    )

    val_dataset = TinyFlexDataset(
        sequences=val_sequences,
        k=k,
        lookup_data=lookup_data
    )

    # 5) Build dataloaders
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

    # 6) Smaller model
    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=len(train_dataset.stoi),
        d_model=32,      # smaller hidden size
        n_heads=2,       # fewer heads
        n_layers=1,      # fewer transformer layers
        max_len=512,
        n_flex=4
    )

    # 7) Optimizer with smaller LR + weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-3
    )

    train_losses = []
    val_losses = []

    num_epochs = 10

    # 8) Epoch loop
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

            total_loss = mlm_loss + 0.5 * flex_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_train_loss += float(total_loss.item())
            num_batches += 1

        avg_train_loss = running_train_loss / num_batches
        avg_val_loss = evaluate(model, val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"epoch {epoch+1:02d} | train_loss = {avg_train_loss:.4f} | val_loss = {avg_val_loss:.4f}")

    print()
    print("start train loss =", train_losses[0])
    print("end train loss =", train_losses[-1])
    print("best train loss =", min(train_losses))
    print("start val loss =", val_losses[0])
    print("end val loss =", val_losses[-1])
    print("best val loss =", min(val_losses))
    print("smaller-model train/val epoch loop complete")


if __name__ == "__main__":
    main()
