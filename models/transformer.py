"""
Full Transformer model for Neural Machine Translation (English → Chinese).
Encoder-Decoder architecture with shared embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import PositionalEncoding, EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    """Transformer Encoder: stack of EncoderLayers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.d_model = d_model
        self.scale = d_model ** 0.5

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            src: (B, S) source token IDs.
            src_mask: (B, 1, 1, S) True where padded.

        Returns:
            memory: (B, S, d_model)
        """
        x = self.embedding(src) * self.scale
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """Transformer Decoder: stack of DecoderLayers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.d_model = d_model
        self.scale = d_model ** 0.5

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (B, T) target token IDs.
            memory: (B, S, d_model) encoder output.
            self_mask: (B, T, T) True where masked (causal + padding).
            memory_mask: (B, 1, 1, S) True where source is padding.

        Returns:
            (B, T, d_model)
        """
        x = self.embedding(tgt) * self.scale
        x = self.pos_encoding(x)
        if torch.isnan(x).any():
            print(f"  [DEBUG] Decoder embedding/pos_encoding NaN!")
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, self_mask, memory_mask)
            if torch.isnan(x).any():
                print(f"  [DEBUG] Decoder layer {i} NaN! "
                      f"self_mask shape={self_mask.shape if self_mask is not None else None}, "
                      f"x min/max={x[~torch.isnan(x)].min().item():.4f}/{x[~torch.isnan(x)].max().item():.4f}")
                break
        return x


class Transformer(nn.Module):
    """
    Full Transformer model for sequence-to-sequence translation.
    English (source) → Chinese (target).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        share_embeddings: bool = True,
    ):
        """
        Args:
            vocab_size: Shared vocabulary size.
            d_model: Model dimension.
            n_layers: Number of encoder/decoder layers.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
            share_embeddings: Whether to share embedding weights between
                              encoder, decoder, and output projection.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.encoder = Encoder(
            vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len
        )
        self.decoder = Decoder(
            vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len
        )

        # Output generation layer
        self.generator = nn.Linear(d_model, vocab_size)

        # Tie embeddings
        if share_embeddings:
            self.generator.weight = self.decoder.embedding.weight
            self.encoder.embedding.weight = self.decoder.embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (B, S) source token IDs.
            tgt: (B, T) target token IDs (teacher forcing).
            src_mask: (B, 1, 1, S) True where source is padding.
            tgt_mask: (B, T, T) True where target should be masked.
            memory_mask: (B, 1, 1, S) True where source is padding (for cross-attention).

        Returns:
            logits: (B, T, vocab_size)
        """
        memory = self.encoder(src, src_mask)
        if torch.isnan(memory).any():
            print(f"  [DEBUG] Encoder output has NaN! src shape={src.shape}, src_mask shape={src_mask.shape}")
            print(f"  [DEBUG] src sample: {src[0,:20]}")
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        if torch.isnan(output).any():
            print(f"  [DEBUG] Decoder output has NaN! tgt shape={tgt.shape}")
        logits = self.generator(output)
        if torch.isnan(logits).any():
            print(f"  [DEBUG] Generator output has NaN!")
        return logits

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode source sequence. Useful for beam search reuse."""
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode one step."""
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        logits = self.generator(output)
        return logits

    def greedy_decode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        bos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.

        Args:
            src: Source tokens (1, S) or (B, S).
            src_mask: Source mask.
            max_len: Maximum generation length.
            bos_id: Beginning-of-sequence token ID.
            eos_id: End-of-sequence token ID.

        Returns:
            Generated token IDs (B, gen_len)
        """
        self.eval()
        B = src.size(0)
        memory = self.encode(src, src_mask)

        # Start with BOS token
        ys = torch.full((B, 1), bos_id, dtype=src.dtype, device=src.device)

        for step in range(max_len):
            # Create causal mask for current sequence
            t_len = ys.size(1)
            tgt_mask = torch.triu(
                torch.ones(t_len, t_len, device=src.device), diagonal=1
            ).bool().unsqueeze(0)  # (1, T, T)

            # Decode
            logits = self.decode(
                ys, memory, tgt_mask, src_mask
            )  # (B, T, vocab_size)

            # Get next token (last position)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            ys = torch.cat([ys, next_token], dim=1)

            # Check if all sequences have EOS
            if (next_token == eos_id).all():
                break

        return ys[:, 1:]  # Remove BOS

    def beam_search_decode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        bos_id: int,
        eos_id: int,
        beam_size: int = 4,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        """
        Beam search decoding.

        Args:
            src: Source tokens (1, S).
            src_mask: Source mask (1, 1, 1, S).
            max_len: Maximum generation length.
            bos_id: BOS token ID.
            eos_id: EOS token ID.
            beam_size: Number of beams.
            length_penalty: Length penalty coefficient.

        Returns:
            Best generated token IDs (1, gen_len).
        """
        self.eval()
        device = src.device

        with torch.no_grad():
            memory = self.encode(src, src_mask)  # (1, S, d_model)
            # Expand memory for beam search
            memory = memory.expand(beam_size, -1, -1)  # (beam, S, d_model)
            mem_mask = src_mask.expand(beam_size, -1, -1, -1)

            # Initialize beams: (beam_size, 1)
            ys = torch.full((1, 1), bos_id, dtype=src.dtype, device=device)
            # Scores: (beam_size,)
            scores = torch.zeros(1, device=device)
            finished = torch.zeros(1, dtype=torch.bool, device=device)

            for step in range(max_len):
                # Current sequence length
                t_len = ys.size(1)

                # Causal mask
                tgt_mask = torch.triu(
                    torch.ones(t_len, t_len, device=device), diagonal=1
                ).bool().unsqueeze(0).expand(ys.size(0), -1, -1)

                # Decode
                logits = self.decode(
                    ys, memory[:ys.size(0)], tgt_mask, mem_mask[:ys.size(0)]
                )  # (current_beam, T, vocab)

                # Next token log-probs from last position
                next_log_probs = F.log_softmax(
                    logits[:, -1, :], dim=-1
                )  # (current_beam, vocab)

                # Calculate cumulative scores
                cum_scores = scores.unsqueeze(1) + next_log_probs  # (beam, vocab)

                # For finished beams, only allow EOS (or pad)
                if finished.any():
                    mask = torch.zeros_like(cum_scores).fill_(float("-inf"))
                    mask[~finished, :] = 0  # unfinished beams: all tokens allowed
                    # EOS for finished beams stays at current score
                    finished_idx = torch.nonzero(finished).squeeze(-1)
                    cum_scores[finished] = float("-inf")
                    cum_scores[finished_idx, eos_id] = scores[finished_idx]

                # Flatten and get top candidates
                flat_scores = cum_scores.view(-1)
                top_scores, top_indices = flat_scores.topk(
                    min(beam_size, flat_scores.size(0))
                )

                # Convert flat indices to (beam_idx, vocab_idx)
                beam_indices = top_indices // self.vocab_size
                token_indices = top_indices % self.vocab_size

                # Build new beams
                new_ys = []
                new_scores = []
                new_finished = []
                for i in range(len(top_scores)):
                    beam_idx = beam_indices[i].item()
                    token = token_indices[i].item()

                    new_seq = torch.cat(
                        [ys[beam_idx], torch.tensor([[token]], device=device)], dim=0
                    )
                    new_ys.append(new_seq.unsqueeze(0))
                    new_scores.append(top_scores[i].item() / (len(new_seq) ** length_penalty))
                    new_finished.append(token == eos_id)

                ys = torch.cat(new_ys, dim=0)
                scores = torch.tensor(new_scores, device=device)
                finished = torch.tensor(new_finished, device=device)

                if finished.all():
                    break

            # Return best (highest normalized score)
            best_idx = scores.argmax()
            return ys[best_idx, 1:].unsqueeze(0)  # (1, gen_len), remove BOS
