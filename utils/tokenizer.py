"""
Tokenizer module for NMT Transformer.
Uses SentencePiece for subword tokenization with shared English-Chinese vocabulary.
"""

import os
import sentencepiece as spm


class Tokenizer:
    """SentencePiece-based tokenizer for English-Chinese translation."""

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Path to a pre-trained SentencePiece model file.
                        If None, a new tokenizer must be trained via `train()`.
        """
        self.sp = None
        self.vocab_size = 0
        if model_path is not None:
            self.load(model_path)

    def train(
        self,
        english_file: str,
        chinese_file: str,
        model_prefix: str = "tokenizer_data/spm_model",
        vocab_size: int = 16000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
        num_threads: int = 8,
    ):
        """
        Train a SentencePiece model on combined English + Chinese data.

        Args:
            english_file: Path to English text file (one sentence per line).
            chinese_file: Path to Chinese text file (one sentence per line).
            model_prefix: Prefix for output model files.
            vocab_size: Vocabulary size.
            character_coverage: Character coverage for the model.
            model_type: 'bpe' or 'unigram'.
            num_threads: Number of threads for training.
        """
        os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

        # Create a combined corpus file
        combined_file = model_prefix + "_combined.txt"
        with open(english_file, "r", encoding="utf-8") as f_en, \
             open(chinese_file, "r", encoding="utf-8") as f_zh, \
             open(combined_file, "w", encoding="utf-8") as f_out:
            for en_line in f_en:
                f_out.write(en_line.strip() + "\n")
            for zh_line in f_zh:
                f_out.write(zh_line.strip() + "\n")

        # Train SentencePiece
        spm.SentencePieceTrainer.train(
            input=combined_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            num_threads=num_threads,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            user_defined_symbols=[],
        )

        # Clean up combined file
        if os.path.exists(combined_file):
            os.remove(combined_file)

        # Load the trained model
        self.load(model_prefix + ".model")
        print(f"Tokenizer trained. Vocab size: {self.vocab_size}")

    def load(self, model_path: str):
        """Load a pre-trained SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.get_piece_size()

    def save(self, model_path: str):
        """Not needed for SentencePiece (model is saved during training)."""
        pass

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list:
        """
        Encode a text string into token IDs.

        Args:
            text: Input text string.
            add_bos: Whether to prepend <s> token.
            add_eos: Whether to append </s> token.

        Returns:
            List of token IDs.
        """
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: list, remove_special: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs.
            remove_special: Whether to remove special tokens during decoding.

        Returns:
            Decoded text string.
        """
        if remove_special:
            # Filter out special tokens (pad, bos, eos)
            ids = [i for i in ids if i not in {
                self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id()
            }]
        return self.sp.decode(ids)

    def pad_id(self) -> int:
        return self.sp.pad_id()

    def bos_id(self) -> int:
        return self.sp.bos_id()

    def eos_id(self) -> int:
        return self.sp.eos_id()

    def unk_id(self) -> int:
        return self.sp.unk_id()

    def __len__(self):
        return self.vocab_size
