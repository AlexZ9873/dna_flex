import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import load_yaml
from src.flex_features import load_lookup_yaml
from src.datasets import TinyFlexDataset, tinyflex_collate_fn
from src.model import TinyMultiTaskModelOneHot

def main():
    torch.manual_seed(0)

    # 1) Load config
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    # 2) Example sequences
    sequences = [
        "ACGTACGTAA",
        "TGCATGCAAATG",
        "GGGAAACCCC",
        "ATATCGCGTACGTA",
    ]

    # 3) Lookup data
    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    # 4) Dataset
    dataset = TinyFlexDataset(
        sequences=sequences,
        k=k,
        lookup_data=lookup_data
    )

    # 5) DataLoader with batch_size=2 so we get multiple batches
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=tinyflex_collate_fn
    )

    # 6) Model
    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=len(dataset.stoi),
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=512,
        n_flex=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_losses = []

    # 7) Train for a few epochs over all batches
    num_epochs = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0

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

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.item())
            num_batches += 1

        avg_loss = running_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"epoch {epoch+1:02d} | avg_loss = {avg_loss:.4f}")

    print()
    print("start epoch loss =", epoch_losses[0])
    print("end epoch loss =", epoch_losses[-1])
    print("best epoch loss =", min(epoch_losses))
    print("DataLoader epoch-based training loop complete")

if __name__ == "__main__":
    main()
