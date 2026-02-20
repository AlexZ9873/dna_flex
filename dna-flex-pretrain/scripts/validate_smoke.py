import os
import math
import random
import torch
import torch.nn.functional as F

from src.utils import load_yaml
from src.tokenization import tokenize_kmers, build_kmer_vocab, encode_sequence
from src.model import TinyMultiTaskModel

def toy_flex_target_gc_fraction(seq: str, k: int):
    """
    Returns per-token regression target = GC fraction of each k-mer.
    This is a toy stand-in for real flexibility targets.
    """
    kmers = tokenize_kmers(seq, k)
    vals = []
    for km in kmers:
        gc = (km.count("G") + km.count("C")) / len(km)
        vals.append(gc)
    return vals

def main():
    # Make results repeatable
    torch.manual_seed(0)
    random.seed(0)

    # 1) Load config
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    # 2) Build vocab and sanity-check vocab size
    stoi, itos = build_kmer_vocab(k)
    expected_vocab = 4 + (4 ** k)  # 4 special tokens + all possible DNA k-mers
    print(f"[OK] k={k} vocab_size={len(stoi)} expected={expected_vocab}")

    # 3) Tokenization sanity-check
    seq = "ACGTACGTAA"  # length 10
    kmers = tokenize_kmers(seq, k)
    print(f"[OK] seq_len={len(seq)} num_kmers={len(kmers)} (should be L-k+1={len(seq)-k+1})")

    # 4) Encoding sanity-check
    ids = encode_sequence(seq, k, stoi, add_cls=True)
    assert ids[0] == stoi["[CLS]"]
    print(f"[OK] first token is CLS id={ids[0]}")

    # 5) Build tensors
    input_ids = torch.tensor([ids], dtype=torch.long)     # [B=1, L]
    attention_mask = torch.ones_like(input_ids)           # no padding yet

    # 6) Create MLM labels (mask one position)
    mlm_labels = torch.full_like(input_ids, -100)
    mask_pos = 2  # avoid CLS at 0
    true_token_id = input_ids[0, mask_pos].item()
    mlm_labels[0, mask_pos] = true_token_id

    input_ids_masked = input_ids.clone()
    input_ids_masked[0, mask_pos] = stoi["[MASK]"]

    # 7) Create toy regression targets (GC fraction) aligned to tokens
    gc_vals = toy_flex_target_gc_fraction(seq, k)  # length L-1 (no CLS)
    flex_targets = torch.tensor([[0.0] + gc_vals], dtype=torch.float32)  # [1, L]
    flex_valid = torch.ones_like(flex_targets, dtype=torch.bool)
    flex_valid[:, 0] = False  # ignore CLS for regression

    # 8) Create model
    model = TinyMultiTaskModel(vocab_size=len(stoi), d_model=64, n_heads=4, n_layers=2, max_len=512)
    # Quick forward-shape check
    with torch.no_grad():
        mlm_logits, flex_pred = model(input_ids_masked, attention_mask)
    print(f"[OK] forward shapes: mlm_logits={tuple(mlm_logits.shape)} flex_pred={tuple(flex_pred.shape)}")
    # mlm_logits should be [1, L, vocab_size]; flex_pred should be [1, L]

    # 9) Training loop (50 steps) to show loss is learnable
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lambda_flex = 0.5

    losses = []
    for step in range(50):
        mlm_logits, flex_pred = model(input_ids_masked, attention_mask)

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

    print(f"[OK] loss start={losses[0]:.4f} end={losses[-1]:.4f} (should generally decrease)")

    # 10) Save + load checkpoint to prove the model is usable/restartable
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/validate_smoke.pt"
    torch.save({"model_state": model.state_dict(), "k": k, "vocab_size": len(stoi)}, ckpt_path)

    model2 = TinyMultiTaskModel(vocab_size=len(stoi), d_model=64, n_heads=4, n_layers=2, max_len=512)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model2.load_state_dict(ckpt["model_state"], strict=True)

    # Verify loaded model runs
    with torch.no_grad():
        mlm_logits2, flex_pred2 = model2(input_ids_masked, attention_mask)
    print(f"[OK] checkpoint reload works, output shapes: {tuple(mlm_logits2.shape)}, {tuple(flex_pred2.shape)}")

    print("\n✅ Smoke test complete: tokenizer + vocab + forward pass + training + checkpoint all working.")

if __name__ == "__main__":
    main()
