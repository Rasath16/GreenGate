"""ShareGPT loader: real-world user queries for text routing evaluation.

Takes the FIRST human turn of each conversation as the query.
Applies an English-length filter and a regex PII scrub (emails, phone
numbers, long digit sequences) before use, per the ethics section.
"""

import re


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
LONGNUM_RE = re.compile(r"\b\d{9,}\b")


def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = LONGNUM_RE.sub("[NUMBER]", text)
    return text


def _mostly_ascii(text: str, threshold: float = 0.9) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= threshold


def load_sharegpt(n_queries: int = 500, seed: int = 42,
                  min_len: int = 15, max_len: int = 600) -> list[str]:
    """Load first-turn user queries from ShareGPT.

    Primary source: 'liyucheng/ShareGPT90K' (parquet, loads with datasets>=5).
    Fallback: 'RyokoAI/ShareGPT52K' raw JSON via hf_hub_download.
    """
    conversations = None
    try:
        from datasets import load_dataset
        ds = load_dataset("liyucheng/ShareGPT90K", split="train")
        conversations = ds["conversations"]
    except Exception as e:
        print(f"  primary ShareGPT source failed ({e}); trying fallback...")
        import json
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("RyokoAI/ShareGPT52K", "sg_90k_part1.json",
                               repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        conversations = [row.get("conversations", []) for row in data]

    queries = []
    for conv in conversations:
        # conv is a list of {"from": "human"/"gpt", "value": ...} turns
        # (ShareGPT90K stores it as a dict of lists — normalise both forms)
        if isinstance(conv, dict):
            froms, values = conv.get("from", []), conv.get("value", [])
            turns = list(zip(froms, values))
        else:
            turns = [(t.get("from", ""), t.get("value", "")) for t in conv]
        for who, text in turns:
            if who != "human":
                continue
            text = (text or "").strip()
            if min_len <= len(text) <= max_len and _mostly_ascii(text):
                queries.append(scrub_pii(text))
            break  # only the first human turn

    # Deterministic shuffle + sample
    import random
    rng = random.Random(seed)
    rng.shuffle(queries)
    return queries[:n_queries]
