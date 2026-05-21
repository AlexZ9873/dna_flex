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
    """
    targets: list of lists, shape [num_tokens, num_features]
    Normalize each feature column separately.
    """
    x = torch.tensor(targets, dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False)
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

    # Mask one token by zeroing its 24 features
    x_masked = x.clone()
    x_masked[0, mask_pos, :] = 0.0

    # 6) Load lookup YAML
    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    # 7) Build dinucleotide targets
    dinuc_feature_names = ["twistDisp", "xDisp"]
    kmers_di, targets_di = sequence_to_multi_dinuc_targets(
        seq=seq,
        k=k,
        lookup_data=lookup_data,
        feature_names=dinuc_feature_names
    )

    # 8) Build trinucleotide targets
    trinuc_feature_names = ["NPP", "DNaseI"]
    kmers_tri, targets_tri = sequence_to_multi_trinuc_targets(
        seq=seq,
        k=k,
        lookup_data=lookup_data,
        feature_names=trinuc_feature_names
    )

    # 9) Sanity check: all k-mer lists should match
    assert kmers == kmers_di
    assert kmers == kmers_tri

    # 10) Combine di + tri targets into one mixed target vector
    combined_targets = []
    for row_di, row_tri in zip(targets_di, targets_tri):
        combined_targets.append(row_di + row_tri)

    mixed_feature_names = dinuc_feature_names + trinuc_feature_names

    # 11) Normalize each feature separately
    flex_targets_norm, mean, std = zscore_columns(combined_targets)

    flex_targets = flex_targets_norm.unsqueeze(0)   # [1, L, 4]
    flex_valid = torch.ones_like(flex_targets, dtype=torch.bool)

    # 12) Create model
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

    # 13) Forward pass
    mlm_logits, flex_pred = model(x_masked, attention_mask)

    # 14) Losses
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

    # 15) One optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 16) Print summary
    print("mixed_feature_names =", mixed_feature_names)
    print("combined_targets =", combined_targets)
    print("flex_targets shape =", tuple(flex_targets.shape))
    print("flex_pred shape =", tuple(flex_pred.shape))
    print("masked true k-mer =", kmers[mask_pos])
    print("mlm_loss =", float(mlm_loss.item()))
    print("flex_loss =", float(flex_loss.item()))
    print("total_loss =", float(total_loss.item()))
    print("one multitask training step complete with MIXED dinucleotide + trinucleotide targets")

if __name__ == "__main__":
    main()
