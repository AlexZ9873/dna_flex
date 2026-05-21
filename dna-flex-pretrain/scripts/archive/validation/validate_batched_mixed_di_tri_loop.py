import torch
import torch.nn.functional as F

from src.utils import load_yaml
from src.tokenization import build_kmer_vocab, encode_sequence_to_ids, encode_sequence_onehot_6x4
from src.model import TinyMultiTaskModelOneHot
from src.flex_features import (
    load_lookup_yaml,
    sequence_to_multi_dinuc_targets,
    sequence_to_multi_trinuc_targets,
)

def zscore_columns(targets):
    x = torch.tensor(targets, dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False)
    std[std == 0] = 1.0
    return (x - mean) / std

def main():
    torch.manual_seed(0)

    # 1) Load config
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    # 2) Build vocab
    stoi, itos = build_kmer_vocab(k)

    # 3) Small batch of sequences
    sequences = [
        "ACGTACGTAA",
        "TGCATGCAAA",
        "GGGAAACCCC",
        "ATATCGCGTA",
    ]

    dinuc_feature_names = ["twistDisp", "xDisp"]
    trinuc_feature_names = ["NPP", "DNaseI"]
    mixed_feature_names = dinuc_feature_names + trinuc_feature_names

    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    batch_x = []
    batch_attention_mask = []
    batch_mlm_labels = []
    batch_flex_targets = []

    # 4) Build batched tensors
    for seq in sequences:
        onehot_3d, onehot_flat, kmers = encode_sequence_onehot_6x4(seq, k=k)
        x = onehot_flat
        attention_mask = torch.ones((x.shape[0],), dtype=torch.long)

        token_ids = encode_sequence_to_ids(seq, k, stoi, add_cls=False)
        mlm_labels = torch.full((x.shape[0],), -100, dtype=torch.long)

        mask_pos = min(2, x.shape[0] - 1)
        mlm_labels[mask_pos] = token_ids[mask_pos]

        x_masked = x.clone()
        x_masked[mask_pos, :] = 0.0

        kmers_di, targets_di = sequence_to_multi_dinuc_targets(
            seq=seq,
            k=k,
            lookup_data=lookup_data,
            feature_names=dinuc_feature_names
        )

        kmers_tri, targets_tri = sequence_to_multi_trinuc_targets(
            seq=seq,
            k=k,
            lookup_data=lookup_data,
            feature_names=trinuc_feature_names
        )

        assert kmers == kmers_di
        assert kmers == kmers_tri

        combined_targets = []
        for row_di, row_tri in zip(targets_di, targets_tri):
            combined_targets.append(row_di + row_tri)

        flex_targets = zscore_columns(combined_targets)

        batch_x.append(x_masked)
        batch_attention_mask.append(attention_mask)
        batch_mlm_labels.append(mlm_labels)
        batch_flex_targets.append(flex_targets)

    x_batch = torch.stack(batch_x, dim=0)                       # [B, L, 24]
    attention_mask_batch = torch.stack(batch_attention_mask, 0) # [B, L]
    mlm_labels_batch = torch.stack(batch_mlm_labels, 0)         # [B, L]
    flex_targets_batch = torch.stack(batch_flex_targets, 0)     # [B, L, 4]
    flex_valid_batch = torch.ones_like(flex_targets_batch, dtype=torch.bool)

    # 5) Model
    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=len(stoi),
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=512,
        n_flex=len(mixed_feature_names)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lambda_flex = 0.5
    losses = []

    # 6) Train for 50 steps
    for step in range(50):
        mlm_logits, flex_pred = model(x_batch, attention_mask_batch)

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels_batch.view(-1),
            ignore_index=-100
        )

        flex_loss = F.huber_loss(
            flex_pred[flex_valid_batch],
            flex_targets_batch[flex_valid_batch],
            delta=1.0
        )

        total_loss = mlm_loss + lambda_flex * flex_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(float(total_loss.item()))

        if (step + 1) % 10 == 0:
            print(f"step {step+1:02d} | total_loss = {total_loss.item():.4f}")

    print()
    print("number of sequences =", len(sequences))
    print("mixed_feature_names =", mixed_feature_names)
    print("start loss =", losses[0])
    print("end loss =", losses[-1])
    print("best loss =", min(losses))
    print("batched mixed di+tri loop complete")

if __name__ == "__main__":
    main()
