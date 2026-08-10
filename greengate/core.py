"""GreenGate — the public library API.

    import greengate
    gw = greengate.GreenGate(small="Qwen/Qwen2.5-0.5B-Instruct",
                             large="gpt-4o-mini", budget_g=0.5)
    r = gw.route("Summarise this document ...")
    r.response, r.decision, r.carbon_g
    gw.profile()      # session totals: carbon, escalation rate, wasted cost
    gw.calibrate()    # one-time temperature fit for unlisted small models

Tier rules (by design, see thesis Ch.3):
  small — any open-weight transformers model, runs locally (the entropy
          signal needs token logits, which APIs do not expose)
  large — a local transformers model ("org/name") OR an OpenAI API model
          name ("gpt-4o-mini")
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

PRESETS_PATH = Path(__file__).parent / "presets.json"
USER_CALIBRATIONS = Path.home() / ".greengate" / "calibrations.json"

# threshold percentile used while auto-tuning on the user's own traffic
AUTO_THRESHOLD_PERCENTILE = {"green": 80, "balanced": 60, "quality": 35}
WARMUP_QUERIES = 20


@dataclass
class RouteResult:
    response: str
    decision: str            # "LOCAL" or "ESCALATE" (or "LOCAL(budget)")
    signal: float            # entropy (bits) or semantic entropy
    threshold: float | None
    carbon_g: float          # full accounting: includes wasted small run
    wasted_carbon_g: float
    latency_s: float
    small_model: str
    large_model: str


@dataclass
class _Session:
    queries: int = 0
    escalated: int = 0
    budget_blocked: int = 0
    carbon_g: float = 0.0
    wasted_carbon_g: float = 0.0
    energy_j: float = 0.0
    latency_s: float = 0.0


def _load_temperature(model_name: str) -> tuple[float, str]:
    """Preset registry first, then the user's own calibrations, else 1.0."""
    for path, source in [(PRESETS_PATH, "preset"),
                         (USER_CALIBRATIONS, "user-calibrated")]:
        try:
            data = json.loads(path.read_text())
            if model_name in data:
                return float(data[model_name]["temperature"]), source
        except (OSError, json.JSONDecodeError):
            pass
    return 1.0, "uncalibrated"


class GreenGate:
    def __init__(self, small: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 large: str = "gpt-4o-mini",
                 mode: str = "balanced",
                 threshold: float | None = None,
                 budget_g: float | None = None,
                 budget_window_s: float = 3600.0,
                 signal: str = "entropy",
                 small_4bit: bool = False,
                 large_4bit: bool = True,
                 max_new_tokens: int = 200,
                 dry_run_api: bool = False,
                 large_is_api: bool | None = None):
        if mode not in AUTO_THRESHOLD_PERCENTILE:
            raise ValueError(f"mode must be one of {list(AUTO_THRESHOLD_PERCENTILE)}")
        if signal not in ("entropy", "semantic"):
            raise ValueError("signal must be 'entropy' or 'semantic'")

        self.small_name, self.large_name = small, large
        self.mode = mode
        self.signal = signal
        self._fixed_threshold = threshold
        self._entropy_history: list[float] = []

        T, source = _load_temperature(small)
        self.temperature = T
        if source == "uncalibrated":
            print(f"[greengate] no calibration preset for {small} — routing on "
                  f"raw entropy with auto-threshold; run gw.calibrate() to fit one")

        from greengate.textgen import SmallTextModel
        self._small = SmallTextModel(small, temperature_T=T,
                                     max_new_tokens=max_new_tokens,
                                     load_in_4bit=small_4bit)

        if large_is_api is None:  # auto-detect: OpenAI naming vs HF hub id
            large_is_api = large.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))
        self._large_is_api = large_is_api
        if self._large_is_api:
            from greengate.api_tier import APILargeTier
            self._large = APILargeTier(model=large, max_tokens=max_new_tokens,
                                       dry_run=dry_run_api)
        else:
            self._large = SmallTextModel(large, max_new_tokens=max_new_tokens,
                                         load_in_4bit=large_4bit)

        self._budget = None
        if budget_g is not None:
            from greengate.budget import SlidingWindowBudget
            self._budget = SlidingWindowBudget(budget_g, budget_window_s)

        self._semantic = None  # lazy — only if signal="semantic"
        self._session = _Session()

    # ------------------------------------------------------------------ #

    def _threshold(self) -> float | None:
        if self._fixed_threshold is not None:
            return self._fixed_threshold
        if len(self._entropy_history) < WARMUP_QUERIES:
            return None  # warmup: not enough traffic seen yet
        h = sorted(self._entropy_history)
        pct = AUTO_THRESHOLD_PERCENTILE[self.mode]
        return h[int(pct / 100 * (len(h) - 1))]

    def _semantic_entropy(self, query: str, k: int = 3) -> tuple[float, float]:
        """(semantic entropy, extra energy J). EXPERIMENTAL — k extra samples."""
        import torch
        if self._semantic is None:
            from greengate.semantic import NLIClusterer
            self._semantic = NLIClusterer()
        answers, extra_j = [], 0.0
        for _ in range(k):
            with torch.no_grad():
                prompt = self._small._chat_wrap(query)
                inputs = self._small.tokenizer(prompt, return_tensors="pt",
                                               truncation=True, max_length=1024)
                inputs = {kk: v.to(self._small.model.device)
                          for kk, v in inputs.items()}
                self._small.profiler.start()
                out = self._small.model.generate(
                    **inputs, max_new_tokens=80, do_sample=True,
                    temperature=1.0, top_p=0.95,
                    pad_token_id=self._small.tokenizer.pad_token_id)
                e, _ = self._small.profiler.stop()
                extra_j += e
            gen = out[0][inputs["input_ids"].shape[1]:]
            answers.append(self._small.tokenizer.decode(
                gen, skip_special_tokens=True).strip())
        from greengate.semantic import semantic_entropy
        return semantic_entropy(self._semantic.cluster(answers)), extra_j

    # ------------------------------------------------------------------ #

    def route(self, query: str) -> RouteResult:
        t0 = time.perf_counter()
        small_r = self._small.generate(query)

        if self.signal == "semantic":
            sig, extra_j = self._semantic_entropy(query)
            extra_c = extra_j / 3_600_000.0 * 1.2 * 475.0
        else:
            sig = small_r.entropy_calibrated
            extra_c = 0.0
        self._entropy_history.append(sig)

        thr = self._threshold()
        wants_escalation = thr is not None and sig > thr

        decision = "LOCAL"
        response = small_r.response
        carbon = small_r.carbon_grams + extra_c
        wasted = 0.0

        if wants_escalation:
            esc_cost_estimate = carbon * 3  # rough pre-check for the budget
            now = time.monotonic()
            if self._budget is not None and not self._budget.allows(now, esc_cost_estimate):
                decision = "LOCAL(budget)"
                self._session.budget_blocked += 1
            else:
                decision = "ESCALATE"
                if self._large_is_api:
                    large_r = self._large.query(query)
                    large_carbon = large_r.carbon_grams
                    response = large_r.response
                else:
                    large_r = self._large.generate(query)
                    large_carbon = large_r.carbon_grams
                    response = large_r.response
                wasted = small_r.carbon_grams  # full accounting
                carbon = large_carbon + wasted + extra_c

        if self._budget is not None:
            self._budget.record(time.monotonic(), carbon)

        latency = time.perf_counter() - t0
        s = self._session
        s.queries += 1
        s.escalated += decision == "ESCALATE"
        s.carbon_g += carbon
        s.wasted_carbon_g += wasted
        s.latency_s += latency

        return RouteResult(
            response=response, decision=decision, signal=sig, threshold=thr,
            carbon_g=carbon, wasted_carbon_g=wasted, latency_s=latency,
            small_model=self.small_name, large_model=self.large_name)

    def profile(self) -> dict:
        s = self._session
        return {
            "queries": s.queries,
            "escalation_rate": s.escalated / s.queries if s.queries else 0.0,
            "budget_blocked": s.budget_blocked,
            "total_carbon_g": round(s.carbon_g, 6),
            "wasted_carbon_g": round(s.wasted_carbon_g, 6),
            "avg_latency_s": round(s.latency_s / s.queries, 3) if s.queries else 0.0,
            "small_model": self.small_name,
            "large_model": self.large_name,
            "signal": self.signal,
            "threshold": self._threshold(),
        }

    def config(self, threshold: float | None = None,
               budget_g: float | None = None,
               mode: str | None = None):
        if threshold is not None:
            self._fixed_threshold = threshold
        if mode is not None:
            if mode not in AUTO_THRESHOLD_PERCENTILE:
                raise ValueError(f"mode must be one of {list(AUTO_THRESHOLD_PERCENTILE)}")
            self.mode = mode
        if budget_g is not None:
            from greengate.budget import SlidingWindowBudget
            self._budget = SlidingWindowBudget(budget_g, 3600.0)
        return self

    def calibrate(self, n: int = 150, seed: int = 42) -> dict:
        """Fit temperature scaling for this small model on held-out MMLU
        validation (the thesis methodology), save it under ~/.greengate/."""
        import torch
        from greengate.mmlu import load_mmlu
        from greengate.evaluator import ChoiceEvaluator
        from greengate.calibration import fit_temperature, ece

        print(f"[greengate] calibrating {self.small_name} on {n} MMLU "
              f"validation questions...")
        ev = ChoiceEvaluator(self.small_name)
        logits, labels = [], []
        for q in load_mmlu(n_questions=n, seed=seed, split="validation"):
            r = ev.evaluate(q)
            logits.append(r.choice_logits)
            labels.append(q.answer_idx)
        lt, lb = torch.tensor(logits), torch.tensor(labels)
        T = fit_temperature(lt, lb)
        result = {"temperature": round(T, 4),
                  "ece_before": round(ece(lt, lb, 1.0), 5),
                  "ece_after": round(ece(lt, lb, T), 5),
                  "fitted_on": f"MMLU validation n={n}"}

        USER_CALIBRATIONS.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if USER_CALIBRATIONS.exists():
            data = json.loads(USER_CALIBRATIONS.read_text())
        data[self.small_name] = result
        USER_CALIBRATIONS.write_text(json.dumps(data, indent=2))

        self.temperature = T
        self._small.T = T
        print(f"[greengate] T={T:.3f}, ECE {result['ece_before']:.4f} -> "
              f"{result['ece_after']:.4f}, saved to {USER_CALIBRATIONS}")
        return result
