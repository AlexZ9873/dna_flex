import torch
import torch.nn.functional as F

from src.utils import load_yaml
from src.tokenization import build_kmer_vocab, encode_sequence_to_ids, encode_sequence_onehot_6x4
from src.model import TinyMultiTaskModelOneHot
from src.flex_features import load_lookup_yaml, get_feature_table, sequence_to_dinuc_targets

def main():
    torch.manual_seed(0)

    # 1) Load config
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    # 2) Build vocab for MLM labels
    stoi, itos = build_kmer_vocab(k)

    # 3) Example sequence
    seq = "ACGTACGTAA"

    # 4) Input features: one-hot 6x4 flattened -> [L, 24]
    onehot_3d, onehot_flat, kmers = encode_sequence_onehot_6x4(seq, k=k)
    x = onehot_flat.unsqueeze(0)   # [1, L, 24]
    attention_mask = torch.ones((1, x.shape[1]), dtype=torch.long)

    # 5) MLM labels
    token_ids = encode_sequence_to_ids(seq, k, stoi, add_cls=False)
    mlm_labels = torch.full((1, x.shape[1]), -100, dtype=torch.long)

    mask_pos = 2
    mlm_labels[0, mask_pos] = token_ids[mask_pos]

    # mask input by zeroing one token
    x_masked = x.clone()
    x_masked[0, mask_pos, :] = 0.0

    # 6) Real twistDisp targets
    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")
    twist_table = get_feature_table(lookup_data, "dinucleotide", "twistDisp")
    _, twist_targets = sequence_to_dinuc_targets(seq, k, twist_table)

    flex_targets = torch.tensor([twist_targets], dtype=torch.float32)
    flex_valid = torch.ones_like(flex_targets, dtype=torch.bool)

    # 7) Model
    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=len(stoi),
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=512
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lambda_flex = 0.5

    # 8) Train for 50 steps
    losses = []

    for step in range(50):
        mlm_logits, flex_pred = model(x_masked, attention_mask)

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

        total_loss = mlm_loss + lambda_flex * flex_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(float(total_loss.item()))

        if (step + 1) % 10 == 0:
            print(f"step {step+1:02d} | total_loss = {total_loss.item():.4f}")

    print()
    print("start loss =", losses[0])
    print("end loss =", losses[-1])
    print("best loss =", min(losses))

if __name__ == "__main__":
    main()
