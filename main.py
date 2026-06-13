"""
Main entry point for NMT Transformer experiment.
Supports modes: train_tokenizer, train, eval, test.

Usage:
    # Step 1: Train tokenizer
    python main.py --mode train_tokenizer

    # Step 2: Train model
    python main.py --mode train

    # Step 3: Evaluate on test set
    python main.py --mode test --checkpoint checkpoints/best_model.pt

    # Step 4: Interactive translation
    python main.py --mode interactive --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import argparse
import torch


# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_EN = os.path.join(DATA_DIR, "TM-training-set", "english.txt")
TRAIN_ZH = os.path.join(DATA_DIR, "TM-training-set", "chinese.txt")
ALIGNMENT = os.path.join(DATA_DIR, "TM-training-set", "Alignment.txt")
DEV_EN = os.path.join(DATA_DIR, "Dev-set", "Niu.dev.en.txt")
DEV_ZH = os.path.join(DATA_DIR, "Dev-set", "Niu.dev.ch.txt")
TEST_EN = os.path.join(DATA_DIR, "Test-set", "Niu.test.en_clean.txt")
TEST_ZH = os.path.join(DATA_DIR, "Test-set", "Niu.test.txt")
TOKENIZER_MODEL = os.path.join(BASE_DIR, "tokenizer_data", "spm_model.model")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def get_config(args) -> dict:
    """Build configuration dictionary from parsed arguments."""
    return {
        "train_en": args.train_en or TRAIN_EN,
        "train_zh": args.train_zh or TRAIN_ZH,
        "dev_en": args.dev_en or DEV_EN,
        "dev_zh": args.dev_zh or DEV_ZH,
        "test_en": args.test_en or TEST_EN,
        "test_zh": args.test_zh or TEST_ZH,
        "alignment_file": args.alignment or ALIGNMENT,
        "tokenizer_model": args.tokenizer_model or TOKENIZER_MODEL,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup_steps": args.warmup_steps,
        "label_smoothing": args.label_smoothing,
        "clip_grad": args.clip_grad,
        "use_amp": not args.no_amp,
        "num_workers": args.num_workers,
        "log_interval": args.log_interval,
        "save_dir": args.save_dir or SAVE_DIR,
        "patience": args.patience,
    }


def cmd_train_tokenizer(args):
    """Train the SentencePiece tokenizer."""
    from utils.tokenizer import Tokenizer

    print("=" * 60)
    print("Training Tokenizer...")
    print("=" * 60)

    train_en = args.train_en or TRAIN_EN
    train_zh = args.train_zh or TRAIN_ZH
    model_prefix = os.path.join(BASE_DIR, "tokenizer_data", "spm_model")

    tokenizer = Tokenizer()
    tokenizer.train(
        english_file=train_en,
        chinese_file=train_zh,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.char_coverage,
        model_type=args.tokenizer_type,
        num_threads=args.num_threads,
    )

    print(f"Tokenizer saved to {model_prefix}.model")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"PAD: {tokenizer.pad_id()}, BOS: {tokenizer.bos_id()}, "
          f"EOS: {tokenizer.eos_id()}, UNK: {tokenizer.unk_id()}")

    # Quick test
    test_en = "This is a test sentence ."
    test_zh = "这是一个测试句子。"
    en_ids = tokenizer.encode(test_en)
    zh_ids = tokenizer.encode(test_zh)
    print(f"EN '{test_en}' → {en_ids} → '{tokenizer.decode(en_ids)}'")
    print(f"ZH '{test_zh}' → {zh_ids} → '{tokenizer.decode(zh_ids)}'")


def cmd_train(args):
    """Train the Transformer model."""
    config = get_config(args)
    print_config(config)

    from train import train as train_model
    train_model(config, args.resume)


def cmd_eval(args):
    """Evaluate model on dev set."""
    from utils.tokenizer import Tokenizer
    from utils.data_loader import TranslationDataset, create_dataloader
    from models.transformer import Transformer
    from evaluation import compute_bleu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = Tokenizer(args.tokenizer_model or TOKENIZER_MODEL)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("config", {})
    model = Transformer(
        vocab_size=len(tokenizer),
        d_model=model_config.get("d_model", args.d_model),
        n_layers=model_config.get("n_layers", args.n_layers),
        n_heads=model_config.get("n_heads", args.n_heads),
        d_ff=model_config.get("d_ff", args.d_ff),
        dropout=model_config.get("dropout", args.dropout),
        max_len=model_config.get("max_len", args.max_len),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    # Create dev dataset
    dev_en = args.dev_en or DEV_EN
    dev_zh = args.dev_zh or DEV_ZH
    dev_dataset = TranslationDataset(dev_en, dev_zh, tokenizer, args.max_len)
    dev_loader = create_dataloader(
        dev_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, pad_id=tokenizer.pad_id()
    )

    # Evaluate
    from train import validate
    metrics = validate(
        model, dev_loader, None, tokenizer,
        device, tokenizer.pad_id(), tokenizer.bos_id(), tokenizer.eos_id()
    )
    print(f"Dev BLEU: {metrics['bleu']:.2f}")


def cmd_test(args):
    """Evaluate on test set and compute final BLEU-4."""
    from utils.tokenizer import Tokenizer
    from models.transformer import Transformer
    from evaluation import evaluate_test_set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = Tokenizer(args.tokenizer_model or TOKENIZER_MODEL)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Load checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join(SAVE_DIR, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model_config = checkpoint.get("config", {})
    model = Transformer(
        vocab_size=len(tokenizer),
        d_model=model_config.get("d_model", args.d_model),
        n_layers=model_config.get("n_layers", args.n_layers),
        n_heads=model_config.get("n_heads", args.n_heads),
        d_ff=model_config.get("d_ff", args.d_ff),
        dropout=model_config.get("dropout", args.dropout),
        max_len=model_config.get("max_len", args.max_len),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}, "
          f"Best dev BLEU: {checkpoint.get('best_bleu', 'N/A')}")

    # Evaluate
    test_en = args.test_en or TEST_EN
    test_zh = args.test_zh or TEST_ZH
    output_dir = args.output_dir or OUTPUT_DIR

    results = evaluate_test_set(
        model, tokenizer, test_en, test_zh, output_dir,
        device, args.max_len, args.beam_size
    )

    print(f"\nResults saved to {output_dir}")
    return results


def cmd_interactive(args):
    """Interactive translation mode."""
    from utils.tokenizer import Tokenizer
    from models.transformer import Transformer
    from evaluation import translate_sentence

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(args.tokenizer_model or TOKENIZER_MODEL)

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join(SAVE_DIR, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get("config", {})
    model = Transformer(
        vocab_size=len(tokenizer),
        d_model=model_config.get("d_model", args.d_model),
        n_layers=model_config.get("n_layers", args.n_layers),
        n_heads=model_config.get("n_heads", args.n_heads),
        d_ff=model_config.get("d_ff", args.d_ff),
        dropout=model_config.get("dropout", args.dropout),
        max_len=model_config.get("max_len", args.max_len),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded. Enter English sentences to translate (type 'quit' to exit).")

    while True:
        try:
            text = input("\nEN > ").strip()
            if text.lower() == "quit":
                break
            if not text:
                continue
            translation = translate_sentence(
                model, tokenizer, text, device, args.max_len, args.beam_size
            )
            print(f"ZH > {translation}")
        except (KeyboardInterrupt, EOFError):
            break


def print_config(config: dict):
    """Print training configuration."""
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="NMT Transformer Experiment (English → Chinese)"
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train_tokenizer", "train", "eval", "test", "interactive"],
        help="Operation mode"
    )

    # Paths
    parser.add_argument("--train_en", type=str, help="Path to English training data")
    parser.add_argument("--train_zh", type=str, help="Path to Chinese training data")
    parser.add_argument("--dev_en", type=str, help="Path to English dev data")
    parser.add_argument("--dev_zh", type=str, help="Path to Chinese dev data")
    parser.add_argument("--test_en", type=str, help="Path to English test data")
    parser.add_argument("--test_zh", type=str, help="Path to Chinese test data")
    parser.add_argument("--alignment", type=str, help="Path to alignment file")
    parser.add_argument("--tokenizer_model", type=str, help="Path to tokenizer model")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    # Tokenizer
    parser.add_argument("--vocab_size", type=int, default=16000, help="Vocabulary size")
    parser.add_argument("--char_coverage", type=float, default=0.9999, help="Character coverage")
    parser.add_argument("--tokenizer_type", type=str, default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--num_threads", type=int, default=8, help="Tokenizer training threads")

    # Model
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")

    # Training
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Warmup steps")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval (steps)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Decoding
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for decoding")

    args = parser.parse_args()

    # Execute mode
    if args.mode == "train_tokenizer":
        cmd_train_tokenizer(args)
    elif args.mode == "train":
        cmd_train(args)
    elif args.mode == "eval":
        cmd_eval(args)
    elif args.mode == "test":
        cmd_test(args)
    elif args.mode == "interactive":
        cmd_interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
