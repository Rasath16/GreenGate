"""Small vision tier: VLM answer generation with confidence signal + energy.

Confidence signal per the methodology: average token probability of the
generated answer (Chapter 1.7.3). Low mean probability = uncertain =
escalate. Entropy variants recorded too for the signal comparison.

Default model: HuggingFaceTB/SmolVLM-Instruct (2B) — open weights, clean
transformers generate() with output_scores (needed for the probability
signal; Moondream-2B's custom modelling code does not expose logits —
substitution documented in the thesis). SmolVLM-256M works for CPU
smoke tests.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from greengate.profiler import CarbonProfiler


@dataclass
class VisionResult:
    response: str
    avg_token_prob: float     # mean P(chosen token) — primary signal
    entropy_mean: float       # mean full-vocab entropy (bits)
    energy_joules: float
    carbon_grams: float
    latency_s: float
    output_tokens: int


class SmallVisionModel:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
                 max_new_tokens: int = 40, device: str | None = None,
                 load_in_4bit: bool = False):
        from transformers import AutoProcessor
        try:  # transformers >= 5
            from transformers import AutoModelForImageTextToText as VLMClass
        except ImportError:  # transformers 4.x
            from transformers import AutoModelForVision2Seq as VLMClass

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_name)
        kwargs = {}
        if device == "cuda":
            kwargs["device_map"] = "auto"
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            else:
                kwargs["dtype"] = torch.float16
        else:
            kwargs["dtype"] = torch.float32
        self.model = VLMClass.from_pretrained(model_name, **kwargs)
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self.profiler = CarbonProfiler()

    @torch.no_grad()
    def answer(self, image, question: str) -> VisionResult:
        messages = [{
            "role": "user",
            "content": [{"type": "image"},
                        {"type": "text",
                         "text": f"{question} Answer briefly."}],
        }]
        prompt = self.processor.apply_chat_template(messages,
                                                    add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        self.profiler.start()
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            do_sample=False, output_scores=True,
            return_dict_in_generate=True)
        energy, carbon = self.profiler.stop()
        latency = time.perf_counter() - t0

        gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        probs_chosen, entropies = [], []
        for step_logits, tok in zip(out.scores, gen_ids):
            p = F.softmax(step_logits[0], dim=-1).clamp(min=1e-9)
            probs_chosen.append(p[tok].item())
            entropies.append(-(p * p.log2()).sum().item())

        response = self.processor.decode(gen_ids, skip_special_tokens=True)
        return VisionResult(
            response=response.strip(),
            avg_token_prob=sum(probs_chosen) / len(probs_chosen) if probs_chosen else 0.0,
            entropy_mean=sum(entropies) / len(entropies) if entropies else 0.0,
            energy_joules=energy,
            carbon_grams=carbon,
            latency_s=latency,
            output_tokens=len(gen_ids),
        )
