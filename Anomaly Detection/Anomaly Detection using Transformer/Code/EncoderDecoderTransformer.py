# transformer_model.py
import math
import torch
import torch.nn as nn

# ----------------------
# Positional Encoding
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[: x.size(0)]


# ----------------------
# Multi-Head Attention
# ----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        len_q, B, _ = q.size()
        len_k, _, _ = k.size()
        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)
        # reshape & permute to (B, H, seq_len, d_k)
        Q = Q.view(len_q, B, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        K = K.view(len_k, B, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        V = V.view(len_k, B, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, V)
        ctx = ctx.permute(2, 0, 1, 3).contiguous()
        ctx = ctx.view(len_q, B, self.num_heads * self.d_k)
        return self.Wo(ctx)


# ----------------------
# Transformer Encoder Block
# ----------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attended)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


# ----------------------
# Transformer Decoder Block
# ----------------------
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # self-attention
        tgt2 = self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt), tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        # cross-attention
        tgt2 = self.cross_attn(self.norm2(tgt), memory, memory, memory_mask)
        tgt = tgt + self.dropout(tgt2)
        # feed forward
        ff_out = self.ff(self.norm3(tgt))
        tgt = tgt + self.dropout(ff_out)
        return tgt


# ----------------------
# Encoder-Decoder Transformer
# ----------------------
class ImprovedEncodeDecoderTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pred_len=20,
    ):
        super().__init__()
        # projections
        self.input_proj = nn.Linear(input_size, d_model)
        self.target_embedding = nn.Linear(output_size, d_model)
        self.output_proj = nn.Linear(d_model, output_size)
        # positional encoding
        self.pos_enc = PositionalEncoding(d_model)
        # encoder/decoder stacks
        self.enc_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.dec_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.pred_len = pred_len
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def create_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return ~mask

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        # src: (B, S, F_in), tgt: (B, T, F_out)
        B, S, _ = src.size()
        device = src.device
        # encode
        memory = self.input_proj(src).permute(1, 0, 2)
        memory = self.pos_enc(memory)
        for enc in self.enc_layers:
            memory = enc(memory)
        # training vs inference
        if self.training and tgt is not None:
            return self._forward_training(memory, tgt, teacher_forcing_ratio)
        else:
            return self._forward_inference(memory, B, device)

    # Note: _forward_training, _forward_inference omitted here for brevity
    # They should be implemented exactly as before or imported if needed


# Expose module API
__all__ = [
    "PositionalEncoding",
    "MultiHeadAttention",
    "TransformerBlock",
    "TransformerDecoderBlock",
    "ImprovedEncodeDecoderTransformer",
]
