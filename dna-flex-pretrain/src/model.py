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
        attention_mask: [B, L]
        """
        B, L, _ = x_onehot_flat.shape

        # Project flattened 6x4 one-hot token into hidden space
        x = self.input_proj(x_onehot_flat)   # [B, L, d_model]

        # Add positional embedding
        pos = torch.arange(L, device=x_onehot_flat.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos)

        # Transformer padding mask
        key_padding_mask = (attention_mask == 0)

        # Contextual encoding
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, d_model]
        return h

class TinyMultiTaskModelOneHot(nn.Module):
    def __init__(
        self,
        input_dim: int = 24,
        vocab_size: int = 4100,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 512,
        n_flex: int = 2
    ):
        super().__init__()
        self.encoder = TinyDNAEncoderOneHot(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len
        )
        self.mlm_head = nn.Linear(d_model, vocab_size)   # predict masked k-mer ID
        self.flex_head = nn.Linear(d_model, n_flex)      # predict multiple regression targets

    def forward(self, x_onehot_flat: torch.Tensor, attention_mask: torch.Tensor, return_hidden: bool = False):
        h = self.encoder(x_onehot_flat, attention_mask)   # [B, L, d_model]
        mlm_logits = self.mlm_head(h)                     # [B, L, vocab_size]
        flex_pred = self.flex_head(h)                     # [B, L, n_flex]
        if return_hidden:
            return mlm_logits, flex_pred, h
        return mlm_logits, flex_pred
