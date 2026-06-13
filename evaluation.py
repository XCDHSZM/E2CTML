"""
Evaluation module for NMT Transformer.
Provides BLEU computation, translation generation, and test set evaluation.
"""

import os
import torch
from typing import List, Tuple, Optional
from sacrebleu.metrics import BLEU


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute corpus-level BLEU-4 score using sacreBLEU.

    Args:
        references: List of reference strings.
        hypotheses: List of hypothesis strings.

    Returns:
        BLEU-4 score (float).
    """
    if not references or not hypotheses:
        return 0.0

    # sacreBLEU expects list of reference lists
    refs = [[r] for r in references]
    bleu = BLEU()
    result = bleu.corpus_score(hypotheses, refs)
    return result.score


def compute_sentence_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU for a single sentence pair."""
    bleu = BLEU()
    result = bleu.sentence_score(hypothesis, [reference])
    return result.score


def translate_sentence(
    model,
    tokenizer,
    src_sentence: str,
    device: torch.device,
    max_len: int = 128,
    beam_size: int = 4,
) -> str:
    """
    Translate a single English sentence to Chinese.

    Args:
        model: Trained Transformer model.
        tokenizer: Tokenizer instance.
        src_sentence: English source sentence.
        device: torch device.
        max_len: Maximum generation length.
        beam_size: Beam size for decoding.

    Returns:
        Translated Chinese sentence.
    """
    model.eval()
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    pad_id = tokenizer.pad_id()

    # Encode source
    src_ids = tokenizer.encode(src_sentence, add_bos=True, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src_tensor == pad_id).unsqueeze(1).unsqueeze(2)  # (1, 1, 1, S)

    with torch.no_grad():
        if beam_size > 1:
            hyp_ids = model.beam_search_decode(
                src_tensor, src_mask, max_len, bos_id, eos_id, beam_size
            )
        else:
            hyp_ids = model.greedy_decode(
                src_tensor, src_mask, max_len, bos_id, eos_id
            )

    hyp_ids = hyp_ids[0].tolist()
    # Remove special tokens
    hyp_ids = [t for t in hyp_ids if t not in {bos_id, eos_id, pad_id}]
    return tokenizer.decode(hyp_ids) if hyp_ids else ""


def translate_file(
    model,
    tokenizer,
    src_file: str,
    output_file: str,
    device: torch.device,
    max_len: int = 128,
    beam_size: int = 4,
    batch_size: int = 32,
) -> List[str]:
    """
    Translate all sentences in a source file.

    Args:
        model: Trained Transformer model.
        tokenizer: Tokenizer instance.
        src_file: Path to English source file (one sentence per line).
        output_file: Path to save translations.
        device: torch device.
        max_len: Maximum generation length.
        beam_size: Beam size for decoding.
        batch_size: Number of sentences to process at once.

    Returns:
        List of translated Chinese sentences.
    """
    model.eval()
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    pad_id = tokenizer.pad_id()

    # Read source sentences
    with open(src_file, "r", encoding="utf-8") as f:
        src_sentences = [line.strip() for line in f if line.strip()]

    translations = []

    with torch.no_grad():
        for i in range(0, len(src_sentences), batch_size):
            batch_sents = src_sentences[i:i+batch_size]

            # Encode batch
            batch_ids = []
            for sent in batch_sents:
                ids = tokenizer.encode(sent, add_bos=True, add_eos=True)
                ids = ids[:max_len]
                batch_ids.append(ids)

            # Pad to same length
            max_seq_len = max(len(ids) for ids in batch_ids)
            padded = torch.full(
                (len(batch_ids), max_seq_len), pad_id, dtype=torch.long
            )
            for j, ids in enumerate(batch_ids):
                padded[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)

            src_tensor = padded.to(device)
            src_mask = (src_tensor == pad_id).unsqueeze(1).unsqueeze(2)

            # Greedy decode each sentence
            for j in range(src_tensor.size(0)):
                hyp_ids = model.greedy_decode(
                    src_tensor[j:j+1], src_mask[j:j+1],
                    max_len, bos_id, eos_id
                )
                hyp_ids = hyp_ids[0].tolist()
                hyp_ids = [t for t in hyp_ids if t not in {bos_id, eos_id, pad_id}]
                trans = tokenizer.decode(hyp_ids) if hyp_ids else ""
                translations.append(trans)

            if (i // batch_size) % 10 == 0:
                print(f"  Translated {min(i+batch_size, len(src_sentences))}/{len(src_sentences)} sentences")

    # Save translations
    with open(output_file, "w", encoding="utf-8") as f:
        for trans in translations:
            f.write(trans + "\n")

    print(f"Translations saved to {output_file}")
    return translations


def evaluate_test_set(
    model,
    tokenizer,
    test_en_file: str,
    test_zh_ref_file: str,
    output_dir: str,
    device: torch.device,
    max_len: int = 128,
    beam_size: int = 1,
) -> dict:
    """
    Evaluate the model on the test set.

    Args:
        model: Trained Transformer model.
        tokenizer: Tokenizer instance.
        test_en_file: Path to English test file.
        test_zh_ref_file: Path to reference Chinese file.
        output_dir: Directory to save outputs.
        device: torch device.
        max_len: Maximum generation length.
        beam_size: Beam size for decoding.

    Returns:
        Dictionary with BLEU score and output paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read reference
    with open(test_zh_ref_file, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    # Translate
    output_file = os.path.join(output_dir, "test_translations.txt")
    print(f"Translating test set ({test_en_file})...")
    hypotheses = translate_file(
        model, tokenizer, test_en_file, output_file,
        device, max_len, beam_size
    )

    # Align lengths
    min_len = min(len(references), len(hypotheses))
    references = references[:min_len]
    hypotheses = hypotheses[:min_len]

    # Compute BLEU
    bleu_score = compute_bleu(references, hypotheses)
    print(f"\n{'='*60}")
    print(f"Test Set BLEU-4: {bleu_score:.2f}")
    print(f"{'='*60}")

    # Also compute per-sentence average
    sent_bleus = []
    for ref, hyp in zip(references, hypotheses):
        sb = compute_sentence_bleu(ref, hyp)
        sent_bleus.append(sb)
    avg_sent_bleu = sum(sent_bleus) / len(sent_bleus) if sent_bleus else 0.0
    print(f"Avg Sentence BLEU: {avg_sent_bleu:.2f}")

    # Save results
    result_file = os.path.join(output_dir, "bleu_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Corpus BLEU-4: {bleu_score:.2f}\n")
        f.write(f"Avg Sentence BLEU: {avg_sent_bleu:.2f}\n")
        f.write(f"Sentences: {min_len}\n")

    return {
        "bleu": bleu_score,
        "avg_sentence_bleu": avg_sent_bleu,
        "output_file": output_file,
        "result_file": result_file,
        "n_sentences": min_len,
    }
