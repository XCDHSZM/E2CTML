"""
Training script for NMT Transformer.
Supports multi-GPU training (DataParallel), mixed precision, Noam scheduler.
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from typing import Dict, Optional

from models.transformer import Transformer
from utils.data_loader import TranslationDataset, create_dataloader
from evaluation import compute_bleu


class NoamScheduler:
    """Learning rate scheduler from 'Attention Is All You Need'."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.base_lr = d_model ** (-0.5)

    def step(self):
        self.step_num += 1
        lr = self.base_lr * min(
            self.step_num ** (-0.5),
            self.step_num * (self.warmup_steps ** (-1.5))
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict["step_num"]
        self.optimizer.load_state_dict(state_dict["optimizer"])


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)
            mask = (target == self.padding_idx).unsqueeze(-1)
            true_dist.masked_fill_(mask, 0.0)
        loss = -(true_dist * log_probs).sum(dim=-1)
        pad_mask = (target != self.padding_idx).float()
        loss = (loss * pad_mask).sum() / pad_mask.sum()
        return loss


def train_epoch(
    model,
    dataloader,
    criterion,
    scheduler: NoamScheduler,
    scaler: GradScaler,
    device,
    pad_id: int,
    use_amp: bool = True,
    clip_grad: float = 1.0,
    log_interval: int = 100,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        T = tgt_input.size(1)
        tgt_mask = tgt_mask[:, :, :T, :T]

        scheduler.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(src, tgt_input, src_mask, tgt_mask, src_mask)
                # Debug: check for NaN in logits
                if batch_idx == 0:
                    print(f"  [DEBUG] logits shape={logits.shape}, has_nan={torch.isnan(logits).any().item()}, "
                          f"min={logits.min().item():.4f}, max={logits.max().item():.4f}")
                loss = criterion(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_output.contiguous().view(-1),
                )
            if batch_idx == 0:
                print(f"  [DEBUG] loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(scheduler.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(scheduler.optimizer)
            scaler.update()
        else:
            logits = model(src, tgt_input, src_mask, tgt_mask, src_mask)
            loss = criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_output.contiguous().view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scheduler.optimizer.step()

        scheduler.step()

        n_tokens = (tgt_output != pad_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if batch_idx % log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            current_loss = total_loss / total_tokens
            ppl = math.exp(min(current_loss, 10))
            lr = scheduler.get_lr()
            print(f"  Step {batch_idx:6d}/{len(dataloader):6d} | "
                  f"Loss: {current_loss:.4f} | PPL: {ppl:.2f} | "
                  f"LR: {lr:.6f} | {elapsed:.0f}s")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 10))
    elapsed = time.time() - start_time
    print(f"Epoch complete | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Time: {elapsed:.0f}s")
    return {"loss": avg_loss, "ppl": ppl}


def validate(
    model,
    dataloader,
    criterion,
    tokenizer,
    device,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_samples: int = 200,
) -> Dict[str, float]:
    """Validate the model and compute BLEU."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            T = tgt_input.size(1)
            tgt_mask_adj = tgt_mask[:, :, :T, :T]

            logits = model(src, tgt_input, src_mask, tgt_mask_adj, src_mask)
            loss = criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_output.contiguous().view(-1),
            )
            n_tokens = (tgt_output != pad_id).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            if len(references) < max_samples:
                for i in range(src.size(0)):
                    if len(references) >= max_samples:
                        break
                    ref_ids = tgt[i][tgt[i] != pad_id].tolist()
                    ref_ids = [t for t in ref_ids if t not in {bos_id, eos_id}]
                    ref_text = tokenizer.decode(ref_ids)
                    references.append(ref_text)

                    hyp_ids = model.greedy_decode(
                        src[i:i+1], src_mask[i:i+1], model.max_len, bos_id, eos_id
                    )
                    hyp_ids = hyp_ids[0].tolist()
                    hyp_ids = [t for t in hyp_ids if t not in {bos_id, eos_id, pad_id}]
                    hyp_text = tokenizer.decode(hyp_ids) if hyp_ids else ""
                    hypotheses.append(hyp_text)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 10))
    bleu = compute_bleu(references, hypotheses) if (references and hypotheses) else 0.0

    print(f"Validation | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | BLEU: {bleu:.2f}")
    return {"loss": avg_loss, "ppl": ppl, "bleu": bleu}


def train(
    config: dict,
    state_dict_path: Optional[str] = None,
):
    """Main training loop."""
    train_en = config["train_en"]
    train_zh = config["train_zh"]
    dev_en = config["dev_en"]
    dev_zh = config["dev_zh"]
    alignment_file = config.get("alignment_file", None)
    tokenizer_model = config["tokenizer_model"]
    d_model = config.get("d_model", 512)
    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 8)
    d_ff = config.get("d_ff", 2048)
    dropout = config.get("dropout", 0.1)
    max_len = config.get("max_len", 128)
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 50)
    warmup_steps = config.get("warmup_steps", 4000)
    label_smoothing = config.get("label_smoothing", 0.1)
    clip_grad = config.get("clip_grad", 1.0)
    use_amp = config.get("use_amp", True)
    num_workers = config.get("num_workers", 4)
    log_interval = config.get("log_interval", 100)
    save_dir = config.get("save_dir", "checkpoints")
    patience = config.get("patience", 10)

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using device: {device} | GPUs: {n_gpus}")

    from utils.tokenizer import Tokenizer
    tokenizer = Tokenizer(tokenizer_model)
    pad_id = tokenizer.pad_id()
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")

    train_dataset = TranslationDataset(
        train_en, train_zh, tokenizer, max_len, alignment_file
    )
    dev_dataset = TranslationDataset(dev_en, dev_zh, tokenizer, max_len)

    train_loader = create_dataloader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers, pad_id=pad_id
    )
    dev_loader = create_dataloader(
        dev_dataset, batch_size, shuffle=False, num_workers=num_workers, pad_id=pad_id
    )
    print(f"Train batches: {len(train_loader)} | Dev batches: {len(dev_loader)}")

    model = Transformer(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_ff=d_ff, dropout=dropout, max_len=max_len,
    )
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model, warmup_steps)
    criterion = LabelSmoothingLoss(vocab_size, pad_id, label_smoothing)
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    start_epoch = 0
    best_bleu = 0.0
    epochs_no_improve = 0

    if state_dict_path:
        print(f"Resuming from {state_dict_path}")
        checkpoint = torch.load(state_dict_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint["model_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_bleu = checkpoint.get("best_bleu", 0.0)
        print(f"Resumed at epoch {start_epoch}, best BLEU: {best_bleu:.2f}")

    print("=" * 60)
    print(f"Training — d_model={d_model}, layers={n_layers}, heads={n_heads}, d_ff={d_ff}")
    print(f"Batch={batch_size}, Epochs={epochs}, Warmup={warmup_steps}, Patience={patience}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")

        train_metrics = train_epoch(
            model, train_loader, criterion, scheduler, scaler,
            device, pad_id, use_amp, clip_grad, log_interval,
        )

        dev_metrics = validate(
            model, dev_loader, criterion, tokenizer,
            device, pad_id, bos_id, eos_id
        )

        current_bleu = dev_metrics["bleu"]
        is_best = current_bleu > best_bleu

        if is_best:
            best_bleu = current_bleu
            epochs_no_improve = 0
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint = {
                "epoch": epoch, "model_state": model_state,
                "scheduler_state": scheduler.state_dict(),
                "best_bleu": best_bleu, "config": config,
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
            print(f"  ✓ Saved best model (BLEU: {best_bleu:.2f})")
        else:
            epochs_no_improve += 1

        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            "epoch": epoch, "model_state": model_state,
            "scheduler_state": scheduler.state_dict(), "best_bleu": best_bleu,
        }, os.path.join(save_dir, "latest_checkpoint.pt"))

        print(f"  Train Loss: {train_metrics['loss']:.4f} | Dev BLEU: {dev_metrics['bleu']:.2f} | Best BLEU: {best_bleu:.2f}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

    print("\n" + "=" * 60)
    print(f"Training complete! Best BLEU: {best_bleu:.2f}")
    print("=" * 60)
    return best_bleu
