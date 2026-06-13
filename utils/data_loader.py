"""
Data loader module for NMT Transformer.
Provides dataset and batching utilities for parallel English-Chinese data.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel sentence pairs (English → Chinese).
    """

    def __init__(
        self,
        english_file: str,
        chinese_file: str,
        tokenizer,
        max_len: int = 128,
        alignment_file: str = None,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(english_file, "r", encoding="utf-8") as f:
            self.en_sentences = [line.strip() for line in f if line.strip()]
        with open(chinese_file, "r", encoding="utf-8") as f:
            self.zh_sentences = [line.strip() for line in f if line.strip()]

        if alignment_file is not None and os.path.exists(alignment_file):
            self._apply_alignment(alignment_file)

        min_len = min(len(self.en_sentences), len(self.zh_sentences))
        self.en_sentences = self.en_sentences[:min_len]
        self.zh_sentences = self.zh_sentences[:min_len]

        print(f"TranslationDataset: {min_len} parallel sentence pairs")

    def _apply_alignment(self, alignment_file: str):
        valid_indices = set()
        with open(alignment_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        en_idx = int(parts[0])
                        zh_idx = int(parts[1])
                        if en_idx < len(self.en_sentences) and zh_idx < len(self.zh_sentences):
                            valid_indices.add((en_idx, zh_idx))
                    except ValueError:
                        continue
        if valid_indices:
            aligned_en = []
            aligned_zh = []
            for en_idx, zh_idx in sorted(valid_indices):
                aligned_en.append(self.en_sentences[en_idx])
                aligned_zh.append(self.zh_sentences[zh_idx])
            self.en_sentences = aligned_en
            self.zh_sentences = aligned_zh
            print(f"Alignment filtered: {len(aligned_en)} aligned pairs")

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_sent = self.en_sentences[idx]
        zh_sent = self.zh_sentences[idx]

        enc_ids = self.tokenizer.encode(en_sent, add_bos=True, add_eos=True)
        dec_ids = self.tokenizer.encode(zh_sent, add_bos=True, add_eos=True)

        enc_ids = enc_ids[:self.max_len]
        dec_ids = dec_ids[:self.max_len]

        return {
            "src": torch.tensor(enc_ids, dtype=torch.long),
            "tgt": torch.tensor(dec_ids, dtype=torch.long),
        }


def collate_fn(batch: List[dict], pad_id: int) -> dict:
    """Collate function for padding sequences in a batch."""
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_id
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=pad_id
    )

    # src_padding_mask: (B, 1, 1, S) — True where PAD
    src_padding_mask = (src_padded == pad_id).unsqueeze(1).unsqueeze(2)

    # tgt_padding_mask: (B, 1, 1, T) — True where PAD
    tgt_padding_mask = (tgt_padded == pad_id).unsqueeze(1).unsqueeze(2)

    T = tgt_padded.size(1)
    # causal_mask: (1, T, T) — upper triangular True
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    # Combine: (B, 1, T, T) — True where masked
    # causal_mask (1, T, T) | tgt_padding_mask expanded to (B, 1, T, T)
    tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0) | tgt_padding_mask.transpose(-1, -2).expand(-1, -1, T, -1)

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_padding_mask,
        "tgt_mask": tgt_mask,
    }


def create_dataloader(
    dataset: TranslationDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pad_id: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        pin_memory=True,
        drop_last=True,
    )
