"""Multiple-choice evaluator for MMLU-style questions.

Instead of free-text generation, we do a SINGLE forward pass and read the
next-token logits restricted to the four choice letters (A/B/C/D). This gives:

- prediction: argmax over the four letter tokens
- choice entropy: Shannon entropy over the renormalised 4-way distribution
  (0 bits = fully confident, 2 bits = maximally uncertain)
- energy/carbon: measured around the forward pass via CarbonProfiler

One forward pass per question per model keeps the evaluation cheap enough
for CPU smoke tests and free-tier GPUs.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from greengate.mmlu import MMLUQuestion, CHOICE_LETTERS
from greengate.profiler import CarbonProfiler


@dataclass
class ChoiceResult:
    predicted_idx: int
    correct: bool
    entropy: float  # bits, over the 4 choices (max = 2.0)
    energy_joules: float
    carbon_grams: float


class ChoiceEvaluator:
    """Wraps one causal LM for multiple-choice prediction with energy tracking."""

    def __init__(self, model_name: str, device: str | None = None,
                 load_in_4bit: bool = False,
                 carbon_intensity: float = 475.0, pue: float = 1.2):
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            else:
                load_kwargs["dtype"] = torch.float16
        else:
            load_kwargs["dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

        self.profiler = CarbonProfiler(carbon_intensity=carbon_intensity, pue=pue)

        # Token ids for " A", " B", " C", " D" (with leading space, as they
        # follow "Answer:"). Fall back to bare letters if needed.
        self.choice_token_ids = []
        for letter in CHOICE_LETTERS:
            ids = self.tokenizer.encode(" " + letter, add_special_tokens=False)
            if len(ids) != 1:
                ids = self.tokenizer.encode(letter, add_special_tokens=False)
            self.choice_token_ids.append(ids[0])

    @torch.no_grad()
    def evaluate(self, q: MMLUQuestion) -> ChoiceResult:
        prompt = q.to_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        self.profiler.start()
        logits = self.model(**inputs).logits[0, -1, :]
        energy, carbon = self.profiler.stop()

        choice_logits = logits[self.choice_token_ids]
        probs = F.softmax(choice_logits, dim=-1).clamp(min=1e-9)
        entropy = -(probs * probs.log2()).sum().item()
        predicted_idx = int(probs.argmax().item())

        return ChoiceResult(
            predicted_idx=predicted_idx,
            correct=(predicted_idx == q.answer_idx),
            entropy=entropy,
            energy_joules=energy,
            carbon_grams=carbon,
        )
