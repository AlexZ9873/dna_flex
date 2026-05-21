import torch
import torch.nn.functional as F

from src.utils import load_yaml
from src.tokenization import build_kmer_vocab, encode_sequence_to_ids, encode_sequence_onehot_6x4
from src.model import TinyMultiTaskModelOneHot

def main():
    # Make results repeatable
    torch.manual_seed(0)

    # 1) Load config
    cfg = load_yaml("configs/pretrain.yaml")
    k = cfg["tokenizer"]["k"]

    # 2) Build vocab (used for MLM labels)
    stoi, itos = build_kmer_vocab(k)

    # 3) Example sequence
    seq = "ACGTACGTAA"

    # 4) Input features: one-hot 6x4 flattened -> [L, 24]
    onehot_3d, onehot_flat, kmers = encode_sequence_onehot_6x4(seq, k=k)
    x = onehot_flat.unsqueeze(0)   # [1, L, 24]
    attention_mask = torch.ones((1, x.shape[1]), dtype=torch.long)

    # 5) MLM labels: only one masked position contributes to loss
    token_ids = encode_sequence_to_ids(seq, k, stoi, add_cls=False)
    labels = torch.full((1, x.shape[1]), -100, dtype=torch.long)

    mask_pos = 2
    true_token_id = token_ids[mask_pos]
    true_kmer = kmers[mask_pos]
    labels[0, mask_pos] = true_token_id

    # Mask input by zeroing the 24-d token vector
    x_masked = x.clone()
    x_masked[0, mask_pos, :] = 0.0

    # 6) Toy regression target (GC fraction per k-mer)
    gc_vals = []
    for km in kmers:
        gc = (km.count("G") + km.count("C")) / len(km)
        gc_vals.append(gc)

    flex_targets = torch.tensor([gc_vals], dtype=torch.float32)   # [1, L]
    flex_valid = torch.ones_like(flex_targets, dtype=torch.bool)

    # 7) Model + optimizer
    model = TinyMultiTaskModelOneHot(
        input_dim=k*4,            # 24 for 6-mer
        vocab_size=len(stoi),     # 4100
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=512
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lambda_flex = 0.5

    # 8) Check masked-token prediction BEFORE training
    model.eval()
    with torch.no_grad():
        mlm_logits0, flex_pred0 = model(x_masked, attention_mask)
        pred_id_before = int(mlm_logits0[0, mask_pos].argmax().item())
        pred_kmer_before = itos[pred_id_before]
        start_total = (
            F.cross_entropy(
                mlm_logits0.view(-1, mlm_logits0.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            + lambda_flex * F.huber_loss(
                flex_pred0[flex_valid],
                flex_targets[flex_valid],
                delta=1.0
            )
        ).item()

    # 9) Train for multiple steps on the same tiny example (intentional overfit test)
    total_losses = []
    mlm_losses = []
    flex_losses = []

    model.train()
    for step in range(50):
        mlm_logits, flex_pred = model(x_masked, attention_mask)

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            labels.view(-1),
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

        total_losses.append(float(total_loss.item()))
        mlm_losses.append(float(mlm_loss.item()))
        flex_losses.append(float(flex_loss.item()))

        if (step + 1) % 10 == 0:
            print(f"step {step+1:02d} | total={total_loss.item():.4f} | mlm={mlm_loss.item():.4f} | flex={flex_loss.item():.4f}")

    # 10) Check masked-token prediction AFTER training
    model.eval()
    with torch.no_grad():
        mlm_logits1, flex_pred1 = model(x_masked, attention_mask)
        pred_id_after = int(mlm_logits1[0, mask_pos].argmax().item())
        pred_kmer_after = itos[pred_id_after]

        final_mlm = F.cross_entropy(
            mlm_logits1.view(-1, mlm_logits1.size(-1)),
            labels.view(-1),
            ignore_index=-100
        ).item()

        final_flex = F.huber_loss(
            flex_pred1[flex_valid],
            flex_targets[flex_valid],
            delta=1.0
        ).item()

        final_total = final_mlm + lambda_flex * final_flex

    # 11) Summary
    print("\n--- Summary ---")
    print("num_kmers =", len(kmers))
    print("masked position =", mask_pos)
    print("true masked k-mer =", true_kmer)
    print("pred before training =", pred_kmer_before)
    print("pred after training  =", pred_kmer_after)
    print(f"start total loss = {start_total:.4f}")
    print(f"end total loss   = {final_total:.4f}")
    print(f"best total loss  = {min(total_losses):.4f}")

    if final_total < start_total:
        print("✅ Validation passed: total loss decreased over training.")
    else:
        print("⚠️ Total loss did not decrease this run (can happen occasionally). Re-run once to check trend.")

if __name__ == "__main__":
    main()
