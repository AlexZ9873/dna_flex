import torch
import torch.nn.functional as F

from src.utils import load_yaml
from src.tokenization import build_kmer_vocab, encode_sequence_to_ids, encode_sequence_onehot_6x4
from src.model import TinyMultiTaskModelOneHot
from src.flex_features import load_lookup_yaml, sequence_to_multi_dinuc_targets

def zscore_columns(targets):
    """
    targets: list of lists, shape [num_tokens, num_features]
    returns normalized version with per-feature z-score normalization
    """
    x = torch.tensor(targets, dtype=torch.float32)   # [L, F]

    mean = x.mean(dim=0, keepdim=True)               # [1, F]
    std = x.std(dim=0, keepdim=True, unbiased=False) # [1, F]

    # avoid divide-by-zero if a feature has zero variance
    std[std == 0] = 1.0

    x_norm = (x - mean) / std
    return x_norm, mean, std

def main():
    torch.manual_seed(0)

    # 1) Load config
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    # 2) Build vocab for MLM labels
    stoi, itos = build_kmer_vocab(k)

    # 3) Example sequence
    seq = "ACGTACGTAA"

    # 4) Build one-hot 6x4 input
    onehot_3d, onehot_flat, kmers = encode_sequence_onehot_6x4(seq, k=k)
    x = onehot_flat.unsqueeze(0)   # [1, L, 24]
    attention_mask = torch.ones((1, x.shape[1]), dtype=torch.long)

    # 5) MLM labels
    token_ids = encode_sequence_to_ids(seq, k, stoi, add_cls=False)
    mlm_labels = torch.full((1, x.shape[1]), -100, dtype=torch.long)

    mask_pos = 2
    mlm_labels[0, mask_pos] = token_ids[mask_pos]

    # mask one token
    x_masked = x.clone()
    x_masked[0, mask_pos, :] = 0.0

    # 6) Load 4 real features
    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")
    feature_names = ["twistDisp", "xDisp", "stifness", "bendingstiffness"]

    target_kmers, combined_targets = sequence_to_multi_dinuc_targets(
        seq=seq,
        k=k,
        lookup_data=lookup_data,
        feature_names=feature_names
    )

    assert kmers == target_kmers

    # 7) Normalize each feature separately
    flex_targets_norm, mean, std = zscore_columns(combined_targets)

    flex_targets = flex_targets_norm.unsqueeze(0)   # [1, L, 4]
    flex_valid = torch.ones_like(flex_targets, dtype=torch.bool)

    # 8) Create model
    model = TinyMultiTaskModelOneHot(
        input_dim=k * 4,
        vocab_size=len(stoi),
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=512,
        n_flex=len(feature_names)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 9) Forward
    mlm_logits, flex_pred = model(x_masked, attention_mask)

    # 10) Losses
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

    lambda_flex = 0.5
    total_loss = mlm_loss + lambda_flex * flex_loss

    # 11) One update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 12) Print useful checks
    print("feature_names =", feature_names)
    print("raw first target =", combined_targets[0])
    print("normalized first target =", flex_targets[0, 0].tolist())
    print("mean per feature =", mean.squeeze(0).tolist())
    print("std per feature =", std.squeeze(0).tolist())
    print("flex_targets shape =", tuple(flex_targets.shape))
    print("flex_pred shape =", tuple(flex_pred.shape))
    print("mlm_loss =", float(mlm_loss.item()))
    print("flex_loss (normalized 4 targets) =", float(flex_loss.item()))
    print("total_loss =", float(total_loss.item()))
    print("one multitask training step complete with NORMALIZED four-target regression")

if __name__ == "__main__":
    main()
