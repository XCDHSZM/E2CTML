"""
Transformer building blocks for NMT.
Implements MultiHeadAttention, PositionWiseFFN, PositionalEncoding,
EncoderLayer, DecoderLayer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        # Create positional encoding matrix: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            x with positional encoding added: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # Extend positional encoding dynamically
            return self.dropout(x + self._extend_pe(seq_len))
        return self.dropout(x + self.pe[:, :seq_len])

    def _extend_pe(self, seq_len: int) -> torch.Tensor:
        """Dynamically compute positional encodings beyond max_len."""
        pe = torch.zeros(1, seq_len, self.pe.size(-1), device=self.pe.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.pe.size(-1), 2).float()
            * (-math.log(10000.0) / self.pe.size(-1))
        ).to(self.pe.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe[:, :seq_len]


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, q_len, d_model)
            key:   (batch_size, k_len, d_model)
            value: (batch_size, v_len, d_model)
            mask:  (batch_size, 1, q_len, k_len) — True where masked (don't attend)

        Returns:
            (batch_size, q_len, d_model)
        """
        B = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask shape: (B, 1, q_len, k_len) or broadcastable
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, n_heads, q_len, d_k)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        return self.w_o(out)


class PositionWiseFFN(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Masked self-attention
        self_attn_out = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        # Cross-attention to encoder output
        cross_out = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout2(cross_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x
