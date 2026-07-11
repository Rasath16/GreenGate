import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

from greengate.entropy import mean_token_entropy, first_token_entropy
from greengate.profiler import CarbonProfiler, QueryCarbonRecord


@dataclass
class RouteResult:
    """Result of routing a single query through GreenGate."""
    query: str
    decision: str  # "ANSWER" or "ESCALATE"
    response: str
    entropy: float
    threshold: float
    energy_joules: float
    carbon_grams: float
    total_carbon_grams: float  # includes wasted cost if escalated
    gqos: float
    model_used: str


class GreenGateRouter:
    """Confidence-aware cascading router for green AI inference.

    Routes queries based on Shannon entropy of the small model's output.
    Low entropy (confident) -> answer with small model.
    High entropy (uncertain) -> escalate to large model.
    """

    def __init__(
        self,
        small_model: str = "gpt2",
        threshold: float = 2.5,
        max_new_tokens: int = 80,
        entropy_mode: str = "mean",
        carbon_intensity: float = 475.0,
        pue: float = 1.2,
        device: str | None = None,
    ):
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens
        self.entropy_mode = entropy_mode

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.profiler = CarbonProfiler(
            carbon_intensity=carbon_intensity,
            pue=pue,
        )

        self._load_model(small_model)

    def _load_model(self, model_name: str):
        load_kwargs = {}
        if self.device == "cuda":
            try:
                load_kwargs["load_in_4bit"] = True
                load_kwargs["device_map"] = "auto"
            except ImportError:
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["dtype"] = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self.model_name = model_name

    def route(self, query: str) -> RouteResult:
        """Route a single query through the entropy gate."""
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        self.profiler.start()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        small_energy, small_carbon = self.profiler.stop()

        if self.entropy_mode == "first":
            entropy = first_token_entropy(list(outputs.scores))
        else:
            entropy = mean_token_entropy(list(outputs.scores))

        if entropy <= self.threshold:
            decision = "ANSWER"
            energy = small_energy
            carbon = small_carbon
            wasted_energy = 0.0
            wasted_carbon = 0.0
        else:
            decision = "ESCALATE"
            # Full carbon accounting: the small model run is wasted energy
            wasted_energy = small_energy
            wasted_carbon = small_carbon
            # In CPU demo mode, simulate the large model cost as ~3x small model
            energy = small_energy * 3.0
            carbon = small_carbon * 3.0

        total_carbon = carbon + wasted_carbon

        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        proxy_accuracy = max(0.0, 1.0 - (entropy / 10.0))
        gqos = proxy_accuracy / total_carbon if total_carbon > 0 else 0.0

        self.profiler.record(
            query=query,
            model_used=self.model_name if decision == "ANSWER" else f"{self.model_name}->large",
            decision=decision,
            energy_joules=energy,
            carbon_grams=carbon,
            wasted_energy=wasted_energy,
            wasted_carbon=wasted_carbon,
        )

        return RouteResult(
            query=query,
            decision=decision,
            response=response.strip(),
            entropy=entropy,
            threshold=self.threshold,
            energy_joules=energy,
            carbon_grams=carbon,
            total_carbon_grams=total_carbon,
            gqos=gqos,
            model_used=self.model_name if decision == "ANSWER" else f"{self.model_name}->large",
        )

    def route_batch(self, queries: list[str]) -> list[RouteResult]:
        return [self.route(q) for q in queries]

    def summary(self) -> dict:
        return self.profiler.summary()
