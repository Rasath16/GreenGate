"""Free-text generation with entropy signal + energy for the small tier."""

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from greengate.profiler import CarbonProfiler


@dataclass
class GenResult:
    response: str
    entropy_raw: float        # mean token entropy, T=1 (bits)
    entropy_calibrated: float # mean token entropy at fitted T (bits)
    entropy_first: float      # first-token entropy, T=1 (bits)
    entropy_max: float        # max per-token entropy, T=1 (bits)
    energy_joules: float
    carbon_grams: float
    latency_s: float
    output_tokens: int


class SmallTextModel:
    def __init__(self, model_name: str, temperature_T: float = 1.0,
                 max_new_tokens: int = 200, device: str | None = None,
                 load_in_4bit: bool = False):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.T = temperature_T
        self.max_new_tokens = max_new_tokens
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self.profiler = CarbonProfiler()

    def _chat_wrap(self, query: str) -> str:
        """Use the model's chat template when available (instruct models)."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                tokenize=False, add_generation_prompt=True)
        return query

    @torch.no_grad()
    def generate(self, query: str) -> GenResult:
        prompt = self._chat_wrap(query)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        self.profiler.start()
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            do_sample=False, output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        energy, carbon = self.profiler.stop()
        latency = time.perf_counter() - t0

        def step_entropies(temp: float) -> list[float]:
            ents = []
            for step_logits in out.scores:
                probs = F.softmax(step_logits[0] / temp, dim=-1).clamp(min=1e-9)
                ents.append(-(probs * probs.log2()).sum().item())
            return ents

        ents_raw = step_entropies(1.0)
        ents_cal = step_entropies(self.T)

        gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return GenResult(
            response=response.strip(),
            entropy_raw=sum(ents_raw) / len(ents_raw) if ents_raw else 0.0,
            entropy_calibrated=sum(ents_cal) / len(ents_cal) if ents_cal else 0.0,
            entropy_first=ents_raw[0] if ents_raw else 0.0,
            entropy_max=max(ents_raw) if ents_raw else 0.0,
            energy_joules=energy,
            carbon_grams=carbon,
            latency_s=latency,
            output_tokens=len(gen_ids),
        )
