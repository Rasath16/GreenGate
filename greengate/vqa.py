"""VQA dataset loading and answer scoring for the vision routing evaluation.

Uses VQAv2 validation questions with embedded COCO images
('merve/vqav2-small') — the same image source as LLaVA-Instruct, but with
ground-truth answers, so vision accuracy is objective and free (no LLM
judge). Scoring: normalised exact/containment match against the
majority human answer.
"""

import re
import string


ARTICLES = {"a", "an", "the"}
PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower().strip().translate(PUNCT)
    words = [w for w in text.split() if w not in ARTICLES]
    return " ".join(words)


def vqa_match(model_answer: str, gt_answer: str) -> bool:
    """Correct if the normalised ground truth equals, or appears as a
    whole-word phrase inside, the normalised model answer."""
    ma, gt = normalize(model_answer), normalize(gt_answer)
    if not gt:
        return False
    if ma == gt:
        return True
    return re.search(rf"(?:^|\s){re.escape(gt)}(?:\s|$)", ma) is not None


def load_vqa(n_questions: int = 500, seed: int = 42):
    """Yield dicts: {image: PIL.Image, question: str, answer: str}."""
    from datasets import load_dataset

    ds = load_dataset("merve/vqav2-small", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n_questions, len(ds))))
    for row in ds:
        yield {
            "image": row["image"].convert("RGB"),
            "question": row["question"],
            "answer": row["multiple_choice_answer"],
        }
