"""Semantic entropy (Farquhar et al., 2024, Nature) — Ablation A.

Samples the small model k times per query, clusters the answers by
bidirectional NLI entailment, and computes entropy over the cluster
distribution. More faithful to *meaning*-level uncertainty than token
entropy, but costs k extra generations per query — the ablation
quantifies whether the signal gain justifies the energy overhead.
"""

import torch


class NLIClusterer:
    """Bidirectional-entailment clustering with a small NLI model."""

    def __init__(self, model_name: str = "microsoft/deberta-base-mnli",
                 device: str | None = None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        # deberta-base-mnli label order: 0=contradiction, 1=neutral, 2=entailment
        self.entail_idx = 2

    @torch.no_grad()
    def entails(self, a: str, b: str) -> bool:
        inputs = self.tokenizer(a, b, return_tensors="pt",
                                truncation=True, max_length=256).to(self.device)
        pred = self.model(**inputs).logits.argmax(dim=-1).item()
        return pred == self.entail_idx

    def bidirectional(self, a: str, b: str) -> bool:
        return self.entails(a, b) and self.entails(b, a)

    def cluster(self, answers: list[str]) -> list[int]:
        """Greedy semantic clustering; returns a cluster id per answer."""
        cluster_ids: list[int] = []
        representatives: list[str] = []
        for ans in answers:
            assigned = None
            for cid, rep in enumerate(representatives):
                if self.bidirectional(ans, rep):
                    assigned = cid
                    break
            if assigned is None:
                assigned = len(representatives)
                representatives.append(ans)
            cluster_ids.append(assigned)
        return cluster_ids


def semantic_entropy(cluster_ids: list[int]) -> float:
    """Shannon entropy (bits) over the empirical cluster distribution."""
    import math
    n = len(cluster_ids)
    counts: dict[int, int] = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1
    return -sum((k / n) * math.log2(k / n) for k in counts.values())
