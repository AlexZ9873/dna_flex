import torch
import torch.nn as nn

class TinyDNAEncoderOneHot(nn.Module):
    def __init__(self, input_dim: int = 24, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)   # 24 -> d_model
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x_onehot_flat: torch.Tensor, attention_mask: torch.Tensor):
        """
        x_onehot_flat: [B, L, 24]
        attention_mask: [B, L] (1=real token, 0=padding)
        """
        B, L, _ = x_onehot_flat.shape

        # 1) Project 24-d one-hot token features into hidden space
        x = self.input_proj(x_onehot_flat)  # [B, L, d_model]

        # 2) Add positional embeddings
        pos = torch.arange(L, device=x_onehot_flat.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos)

        # 3) Transformer padding mask (True means ignore)
        key_padding_mask = (attention_mask == 0)

        # 4) Contextual encoding
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, d_model]
        return h

class TinyMultiTaskModelOneHot(nn.Module):
    def __init__(self, input_dim: int = 24, vocab_size: int = 4100, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, max_len: int = 512):
        super().__init__()
        self.encoder = TinyDNAEncoderOneHot(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len
        )
        self.mlm_head = nn.Linear(d_model, vocab_size)  # predict k-mer ID
        self.flex_head = nn.Linear(d_model, 1)          # predict one scalar per token

    def forward(self, x_onehot_flat: torch.Tensor, attention_mask: torch.Tensor):
        h = self.encoder(x_onehot_flat, attention_mask)   # [B, L, d_model]
        mlm_logits = self.mlm_head(h)                     # [B, L, vocab_size]
        flex_pred = self.flex_head(h).squeeze(-1)         # [B, L]
        return mlm_logits, flex_pred
