"""Large-tier inference via OpenAI API with EcoLogits carbon estimation.

Dual-Mode Carbon Profiler, API side: local models are measured with
pynvml (profiler.py); API models are estimated with EcoLogits
(Rince et al., 2025, JOSS). If EcoLogits is unavailable or fails,
falls back to a documented per-token estimate and flags it.
"""

import os
import time
from dataclasses import dataclass

# Fallback constants (used ONLY if EcoLogits fails) — documented in thesis:
# Jegham et al. (2025) measure ~0.42 Wh for a mean GPT-4o query (~150 output
# tokens). GPT-4o-mini is a distilled/smaller deployment; we conservatively
# assume 0.5x the 4o figure, scaled linearly by output tokens.
FALLBACK_WH_PER_OUTPUT_TOKEN = 0.42 * 0.5 / 150  # = 0.0014 Wh/token
WORLD_CARBON_INTENSITY = 475.0  # gCO2/kWh
DATACENTER_PUE = 1.2


@dataclass
class APIResult:
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    energy_wh: float
    carbon_grams: float
    carbon_source: str  # "ecologits" or "fallback_estimate"


class APILargeTier:
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 300,
                 dry_run: bool = False):
        self.model = model
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        self._ecologits_ok = False

        if dry_run:
            return

        # EcoLogits must be initialised BEFORE the OpenAI client is created.
        try:
            from ecologits import EcoLogits
            EcoLogits.init(providers=["openai"])
            self._ecologits_ok = True
        except Exception as e:
            print(f"  [api_tier] EcoLogits unavailable ({e}) — using documented fallback estimate")

        from openai import OpenAI
        self.client = OpenAI()  # reads OPENAI_API_KEY env var

    def query_vision(self, prompt: str, image) -> APIResult:
        """Vision escalation: PIL image sent as a base64 data URL."""
        if self.dry_run:
            return self.query(prompt)
        import base64
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        return self._chat(content)

    def query(self, prompt: str) -> APIResult:
        if self.dry_run:
            return APIResult(
                response="[DRY RUN — no API call made]",
                model=self.model, input_tokens=len(prompt) // 4,
                output_tokens=50, latency_s=0.5,
                energy_wh=50 * FALLBACK_WH_PER_OUTPUT_TOKEN,
                carbon_grams=50 * FALLBACK_WH_PER_OUTPUT_TOKEN / 1000
                * DATACENTER_PUE * WORLD_CARBON_INTENSITY,
                carbon_source="dry_run",
            )

        return self._chat(prompt)

    def _chat(self, content) -> APIResult:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
        )
        latency = time.perf_counter() - t0

        text = resp.choices[0].message.content or ""
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens

        energy_wh, carbon_g, source = None, None, "fallback_estimate"
        if self._ecologits_ok and hasattr(resp, "impacts"):
            try:
                # EcoLogits attaches .impacts; energy in kWh (value or range)
                e = resp.impacts.energy.value
                energy_kwh = (e.min + e.max) / 2 if hasattr(e, "min") else float(e)
                g = resp.impacts.gwp.value
                gwp_kg = (g.min + g.max) / 2 if hasattr(g, "min") else float(g)
                energy_wh = energy_kwh * 1000
                carbon_g = gwp_kg * 1000
                source = "ecologits"
            except Exception:
                pass
        if energy_wh is None:
            energy_wh = out_tok * FALLBACK_WH_PER_OUTPUT_TOKEN
            carbon_g = energy_wh / 1000 * DATACENTER_PUE * WORLD_CARBON_INTENSITY

        return APIResult(
            response=text.strip(), model=self.model,
            input_tokens=in_tok, output_tokens=out_tok,
            latency_s=latency, energy_wh=energy_wh,
            carbon_grams=carbon_g, carbon_source=source,
        )
