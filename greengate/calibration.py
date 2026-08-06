"""Temperature scaling calibration (Guo et al., 2017).

Fits a single scalar T on a held-out validation split by minimising
negative log-likelihood. T > 1 softens overconfident distributions.
Reports Expected Calibration Error (ECE) before and after.
"""

import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                    t_min: float = 0.25, t_max: float = 8.0,
                    n_grid: int = 400) -> float:
    """Fit T by fine grid search over NLL (robust, dependency-free).

    Args:
        logits: (n, k) raw choice logits from the validation split
        labels: (n,) correct choice indices
    """
    best_t, best_nll = 1.0, float("inf")
    for i in range(n_grid):
        t = t_min + (t_max - t_min) * i / (n_grid - 1)
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def ece(logits: torch.Tensor, labels: torch.Tensor,
        temperature: float = 1.0, n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    probs = F.softmax(logits / temperature, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()

    total = 0.0
    n = len(labels)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = conf[mask].mean().item()
        avg_acc = correct[mask].mean().item()
        total += (mask.sum().item() / n) * abs(avg_conf - avg_acc)
    return total


def calibrated_entropy(logits: torch.Tensor, temperature: float) -> float:
    """Shannon entropy (bits) of the temperature-scaled distribution."""
    probs = F.softmax(logits / temperature, dim=-1).clamp(min=1e-9)
    return -(probs * probs.log2()).sum(dim=-1).item()
