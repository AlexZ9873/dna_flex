import torch

def pad_2d(x, target_len, pad_value=0.0):
    L, D = x.shape
    if L == target_len:
        return x
    pad = torch.full((target_len - L, D), pad_value, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

def pad_1d(x, target_len, pad_value=0):
    L = x.shape[0]
    if L == target_len:
        return x
    pad = torch.full((target_len - L,), pad_value, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

def genome_collate_fn(batch):
    max_len = max(item["length"] for item in batch)
    num_features = batch[0]["flex_targets"].shape[1]

    xs, ams, labels, flexs, flex_valids = [], [], [], [], []
    lengths = []

    for item in batch:
        L = item["length"]
        xs.append(pad_2d(item["x"], max_len, 0.0))
        ams.append(pad_1d(item["attention_mask"], max_len, 0))
        labels.append(pad_1d(item["mlm_labels"], max_len, -100))
        flexs.append(pad_2d(item["flex_targets"], max_len, 0.0))

        fv = torch.zeros((max_len, num_features), dtype=torch.bool)
        fv[:L, :] = True
        flex_valids.append(fv)

        lengths.append(L)

    return {
        "x": torch.stack(xs, 0),
        "attention_mask": torch.stack(ams, 0),
        "mlm_labels": torch.stack(labels, 0),
        "flex_targets": torch.stack(flexs, 0),
        "flex_valid": torch.stack(flex_valids, 0),
        "lengths": lengths,
    }
