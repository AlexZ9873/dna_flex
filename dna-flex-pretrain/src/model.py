import torch
import torch.nn as nn

class TinyDNAEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, max_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        key_padding_mask = (attention_mask == 0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return h

class TinyMultiTaskModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, max_len: int = 512):
        super().__init__()
        self.encoder = TinyDNAEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len
        )
        self.mlm_head = nn.Linear(d_model, vocab_size)  # token prediction
        self.flex_head = nn.Linear(d_model, 1)          # one scalar per token (toy flex target)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        h = self.encoder(input_ids, attention_mask)      # [B, L, D]
        mlm_logits = self.mlm_head(h)                    # [B, L, V]
        flex_pred = self.flex_head(h).squeeze(-1)        # [B, L]
        return mlm_logits, flex_pred
