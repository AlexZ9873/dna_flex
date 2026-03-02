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

    # 3) Load lookup table data
    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    # 4) Create dataset
    dataset = TinyFlexDataset(
        sequences=sequences,
        k=k,
        lookup_data=lookup_data
    )

    # 5) Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=tinyflex_collate_fn
    )

    # 6) Get one batch from the dataloader
    batch = next(iter(loader))

    x = batch["x"]                         # [B, maxL, 24]
    attention_mask = batch["attention_mask"]   # [B, maxL]
    mlm_labels = batch["mlm_labels"]       # [B, maxL]
    flex_targets = batch["flex_targets"]   # [B, maxL, 4]
    flex_valid = batch["flex_valid"]       # [B, maxL, 4]

    # 7) Create model
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

    # 8) Forward pass
    mlm_logits, flex_pred = model(x, attention_mask)

    # 9) MLM loss
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, mlm_logits.size(-1)),
        mlm_labels.view(-1),
        ignore_index=-100
    )

    # 10) Flex regression loss
    flex_loss = F.huber_loss(
        flex_pred[flex_valid],
        flex_targets[flex_valid],
        delta=1.0
    )

    # 11) Combined loss
    lambda_flex = 0.5
    total_loss = mlm_loss + lambda_flex * flex_loss

    # 12) One optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 13) Print summary
    print("x shape =", tuple(x.shape))
    print("attention_mask shape =", tuple(attention_mask.shape))
    print("mlm_labels shape =", tuple(mlm_labels.shape))
    print("flex_targets shape =", tuple(flex_targets.shape))
    print("flex_valid shape =", tuple(flex_valid.shape))
    print("mlm_logits shape =", tuple(mlm_logits.shape))
    print("flex_pred shape =", tuple(flex_pred.shape))
    print("mlm_loss =", float(mlm_loss.item()))
    print("flex_loss =", float(flex_loss.item()))
    print("total_loss =", float(total_loss.item()))
    print("one DataLoader-based training step complete")

if __name__ == "__main__":
    main()
