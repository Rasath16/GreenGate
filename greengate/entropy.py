import torch
import torch.nn.functional as F


def shannon_entropy(logits: torch.Tensor) -> float:
    """Compute Shannon entropy H(x) = -sum(P(x) * log2(P(x))) over token logits.

    Args:
        logits: Raw logits tensor of shape (vocab_size,) from a single generation step.

    Returns:
        Entropy in bits. Low = confident, high = uncertain.
    """
    probs = F.softmax(logits, dim=-1)
    probs = probs.clamp(min=1e-9)
    entropy = -(probs * probs.log2()).sum(dim=-1)
    return entropy.item()


def mean_token_entropy(all_logits: list[torch.Tensor]) -> float:
    """Average Shannon entropy across all generated tokens."""
    if not all_logits:
        return 0.0
    entropies = [shannon_entropy(logits.squeeze()) for logits in all_logits]
    return sum(entropies) / len(entropies)


def first_token_entropy(all_logits: list[torch.Tensor]) -> float:
    """Shannon entropy of just the first generated token.

    The first token is often the most informative signal for routing:
    a confident model commits early, an uncertain model hedges immediately.
    """
    if not all_logits:
        return 0.0
    return shannon_entropy(all_logits[0].squeeze())
