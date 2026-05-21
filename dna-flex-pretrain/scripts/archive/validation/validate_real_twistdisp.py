import torch
import torch.nn.functional as F

from src.utils import load_yaml
from src.tokenization import build_kmer_vocab, encode_sequence_to_ids, encode_sequence_onehot_6x4
from src.model import TinyMultiTaskModelOneHot
from src.flex_features import load_lookup_yaml, get_feature_table, sequence_to_dinuc_targets

def main():
    # 1) Load config
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    # 2) Build vocab for MLM labels
    stoi, itos = build_kmer_vocab(k)

    # 3) Example DNA sequence
    seq = "ACGTACGTAA"

    # 4) Build one-hot 6x4 input features
    onehot_3d, onehot_flat, kmers = encode_sequence_onehot_6x4(seq, k=k)
    x = onehot_flat.unsqueeze(0)   # [1, L, 24]
    attention_mask = torch.ones((1, x.shape[1]), dtype=torch.long)

    # 5) Build MLM labels (predict one masked 6-mer)
    token_ids = encode_sequence_to_ids(seq, k, stoi, add_cls=False)
    mlm_labels = torch.full((1, x.shape[1]), -100, dtype=torch.long)

    mask_pos = 2
    mlm_labels[0, mask_pos] = token_ids[mask_pos]

    # Mask the input token by zeroing its 24 features
    x_masked = x.clone()
    x_masked[0, mask_pos, :] = 0.0

    # 6) Load real lookup table and build real regression targets
    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")
    twist_table = get_feature_table(lookup_data, "dinucleotide", "twistDisp")

    target_kmers, twist_targets = sequence_to_dinuc_targets(seq, k, twist_table)
    flex_targets = torch.tensor([twist_targets], dtype=torch.float32)  # [1, L]
    flex_valid = torch.ones_like(flex_targets, dtype=torch.bool)

    # 7) Create multitask model
    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,        # 24 for 6-mer
        vocab_size=len(stoi),   # 4100
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=512
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 8) Forward pass
    mlm_logits, flex_pred = model(x_masked, attention_mask)

    # 9) Compute two losses
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

    # 10) Combine losses
    lambda_flex = 0.5
    total_loss = mlm_loss + lambda_flex * flex_loss

    # 11) One optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 12) Print summary
    print("kmers =", kmers)
    print("real twistDisp targets =", twist_targets)
    print("masked position =", mask_pos)
    print("masked true k-mer =", kmers[mask_pos])
    print("mlm_loss =", float(mlm_loss.item()))
    print("flex_loss (real twistDisp) =", float(flex_loss.item()))
    print("total_loss =", float(total_loss.item()))
    print("one multitask training step complete with REAL twistDisp target")

if __name__ == "__main__":
    main()
