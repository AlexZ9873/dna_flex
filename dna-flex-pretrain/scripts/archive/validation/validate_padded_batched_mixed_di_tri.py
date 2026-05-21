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

def pad_2d(x, target_len, pad_value=0.0):
    # x: [L, D]
    L, D = x.shape
    if L == target_len:
        return x
    pad = torch.full((target_len - L, D), pad_value, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

def pad_1d(x, target_len, pad_value=0):
    # x: [L]
    L = x.shape[0]
    if L == target_len:
        return x
    pad = torch.full((target_len - L,), pad_value, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

def main():
    torch.manual_seed(0)

    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]
    stoi, itos = build_kmer_vocab(k)

    sequences = [
        "ACGTACGTAA",       # 5 tokens
        "TGCATGCAAATG",     # 7 tokens
        "GGGAAACCCC",       # 5 tokens
        "ATATCGCGTACGTA",   # 9 tokens
    ]

    dinuc_feature_names = ["twistDisp", "xDisp"]
    trinuc_feature_names = ["NPP", "DNaseI"]
    mixed_feature_names = dinuc_feature_names + trinuc_feature_names

    lookup_data = load_lookup_yaml("data/raw/flex_tables/lookup.yaml")

    examples = []
    max_tokens = 0

    # build each example first
    for seq in sequences:
        onehot_3d, onehot_flat, kmers = encode_sequence_onehot_6x4(seq, k=k)   # [L, 24]
        L = onehot_flat.shape[0]
        max_tokens = max(max_tokens, L)

        token_ids = encode_sequence_to_ids(seq, k, stoi, add_cls=False)
        mlm_labels = torch.full((L,), -100, dtype=torch.long)

        mask_pos = min(2, L - 1)
        mlm_labels[mask_pos] = token_ids[mask_pos]

        x_masked = onehot_flat.clone()
        x_masked[mask_pos, :] = 0.0

        kmers_di, targets_di = sequence_to_multi_dinuc_targets(
            seq=seq, k=k, lookup_data=lookup_data, feature_names=dinuc_feature_names
        )
        kmers_tri, targets_tri = sequence_to_multi_trinuc_targets(
            seq=seq, k=k, lookup_data=lookup_data, feature_names=trinuc_feature_names
        )

        assert kmers == kmers_di
        assert kmers == kmers_tri

        combined_targets = []
        for row_di, row_tri in zip(targets_di, targets_tri):
            combined_targets.append(row_di + row_tri)

        flex_targets = zscore_columns(combined_targets)   # [L, 4]

        examples.append({
            "x": x_masked,
            "mlm_labels": mlm_labels,
            "flex_targets": flex_targets,
            "length": L,
        })

    # pad to max length
    batch_x = []
    batch_attention_mask = []
    batch_mlm_labels = []
    batch_flex_targets = []
    batch_flex_valid = []

    for ex in examples:
        L = ex["length"]

        x_pad = pad_2d(ex["x"], max_tokens, pad_value=0.0)                    # [maxL, 24]
        labels_pad = pad_1d(ex["mlm_labels"], max_tokens, pad_value=-100)     # [maxL]
        flex_pad = pad_2d(ex["flex_targets"], max_tokens, pad_value=0.0)      # [maxL, 4]

        attention_mask = torch.zeros((max_tokens,), dtype=torch.long)
        attention_mask[:L] = 1

        flex_valid = torch.zeros((max_tokens, len(mixed_feature_names)), dtype=torch.bool)
        flex_valid[:L, :] = True

        batch_x.append(x_pad)
        batch_attention_mask.append(attention_mask)
        batch_mlm_labels.append(labels_pad)
        batch_flex_targets.append(flex_pad)
        batch_flex_valid.append(flex_valid)

    x_batch = torch.stack(batch_x, dim=0)                         # [B, maxL, 24]
    attention_mask_batch = torch.stack(batch_attention_mask, 0)   # [B, maxL]
    mlm_labels_batch = torch.stack(batch_mlm_labels, 0)           # [B, maxL]
    flex_targets_batch = torch.stack(batch_flex_targets, 0)       # [B, maxL, 4]
    flex_valid_batch = torch.stack(batch_flex_valid, 0)           # [B, maxL, 4]

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

    total_loss = mlm_loss + 0.5 * flex_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("sequence lengths (tokens) =", [ex["length"] for ex in examples])
    print("max_tokens =", max_tokens)
    print("x_batch shape =", tuple(x_batch.shape))
    print("attention_mask_batch shape =", tuple(attention_mask_batch.shape))
    print("mlm_labels_batch shape =", tuple(mlm_labels_batch.shape))
    print("flex_targets_batch shape =", tuple(flex_targets_batch.shape))
    print("flex_valid_batch shape =", tuple(flex_valid_batch.shape))
    print("mlm_logits shape =", tuple(mlm_logits.shape))
    print("flex_pred shape =", tuple(flex_pred.shape))
    print("mlm_loss =", float(mlm_loss.item()))
    print("flex_loss =", float(flex_loss.item()))
    print("total_loss =", float(total_loss.item()))
    print("padded batched mixed di+tri validation complete")

if __name__ == "__main__":
    main()
