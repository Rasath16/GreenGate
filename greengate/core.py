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
# the break-even verdict is only reported once the large tier has been measured
# often enough for its mean cost to mean anything
MIN_LARGE_CALLS_FOR_VERDICT = 5


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
    # per-tier totals, so the break-even condition can be applied to measured
    # costs rather than to numbers the developer has to work out themselves
    small_carbon_g: float = 0.0     # every small run, discarded ones included
    large_carbon_g: float = 0.0     # large runs only
    large_calls: int = 0


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
                 large_is_api: bool | None = None,
                 carbon_intensity: float | None = None):
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

        # Grid carbon intensity varies by more than twenty times between
        # regions, so a developer's real figure is only as good as this.
        self.carbon_intensity = carbon_intensity

        from greengate.textgen import SmallTextModel
        self._small = SmallTextModel(small, temperature_T=T,
                                     max_new_tokens=max_new_tokens,
                                     load_in_4bit=small_4bit,
                                     carbon_intensity=carbon_intensity)

        if large_is_api is None:  # auto-detect: OpenAI naming vs HF hub id
            large_is_api = large.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))
        self._large_is_api = large_is_api
        if self._large_is_api:
            from greengate.api_tier import APILargeTier
            self._large = APILargeTier(model=large, max_tokens=max_new_tokens,
                                       dry_run=dry_run_api)
        else:
            self._large = SmallTextModel(large, max_new_tokens=max_new_tokens,
                                         load_in_4bit=large_4bit,
                                         carbon_intensity=carbon_intensity)

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
        s.small_carbon_g += small_r.carbon_grams
        if decision == "ESCALATE":
            s.large_carbon_g += large_carbon
            s.large_calls += 1

        return RouteResult(
            response=response, decision=decision, signal=sig, threshold=thr,
            carbon_g=carbon, wasted_carbon_g=wasted, latency_s=latency,
            small_model=self.small_name, large_model=self.large_name)

    def profile(self) -> dict:
        """Session accounting, plus the break-even verdict once enough traffic
        has been seen to support one.

        The verdict compares this cascade against always using the large tier,
        on costs measured from this session rather than assumed. Until the
        large tier has run often enough for its mean cost to be meaningful,
        ``verdict`` says what is still missing instead of guessing.
        """
        s = self._session
        out = {
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

        mean_small = s.small_carbon_g / s.queries if s.queries else 0.0
        mean_large = s.large_carbon_g / s.large_calls if s.large_calls else 0.0
        out["mean_small_carbon_g"] = round(mean_small, 8)
        out["mean_large_carbon_g"] = round(mean_large, 8)

        if s.large_calls >= MIN_LARGE_CALLS_FOR_VERDICT and mean_large > 0:
            from greengate.measure import should_cascade
            out.update(should_cascade(mean_small, mean_large,
                                      out["escalation_rate"]))
        else:
            missing = []
            if s.queries < WARMUP_QUERIES:
                missing.append(f"{WARMUP_QUERIES - s.queries} more queries to "
                               f"set a threshold")
            need = MIN_LARGE_CALLS_FOR_VERDICT - s.large_calls
            if need > 0:
                missing.append(f"{need} more escalation"
                               f"{'s' if need != 1 else ''} to measure the "
                               f"large tier")
            out["verdict"] = ("no verdict yet: needs " + " and ".join(missing)
                              if missing else "no verdict yet")
        return out

    def audit(self, queries: list[str],
              rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7),
              verbose: bool = True) -> dict:
        """Decide whether to cascade at all, before deploying anything.

        Routing in production learns slowly and blindly: the threshold needs a
        warm-up, and the cost of the large tier is only discovered on queries
        that happen to escalate. This runs BOTH tiers over a sample of the
        caller's own queries instead, so the inter-tier cost ratio is measured
        directly and every threshold can be evaluated on the same data.

        The sample should look like real traffic; a few dozen queries is
        usually enough to place the cost ratio, since it varies far less
        between queries than quality does.

        This reports cost only. The library has no judge, so it cannot tell
        you what escalating buys in quality: that has to come from the
        caller's own evaluation. What it does tell you is the ceiling, the
        break-even rate, and the threshold that realises any given escalation
        rate on this traffic.

        Args:
            queries: a representative sample of real queries.
            rates: escalation rates to report thresholds and savings for.
            verbose: print the table as it is computed.

        Returns:
            The measured cost ratio and break-even rate, a row per escalation
            rate with the threshold that achieves it and the predicted saving,
            and a recommendation.
        """
        from greengate.measure import should_cascade

        if not queries:
            raise ValueError("audit needs at least one query")

        signals, small_c, large_c = [], [], []
        for i, q in enumerate(queries, 1):
            small_r = self._small.generate(q)
            signals.append(small_r.entropy_calibrated)
            small_c.append(small_r.carbon_grams)
            if self._large_is_api:
                large_c.append(self._large.query(q).carbon_grams)
            else:
                large_c.append(self._large.generate(q).carbon_grams)
            if verbose and (i % 10 == 0 or i == len(queries)):
                print(f"[greengate] audited {i}/{len(queries)}")

        n = len(queries)
        mean_small = sum(small_c) / n
        mean_large = sum(large_c) / n
        ratio = mean_small / mean_large if mean_large else float("inf")
        ordered = sorted(signals)

        rows = []
        for r in rates:
            if r <= 0:
                thr = None                      # never escalate
            elif r >= 1:
                thr = float("-inf")             # always escalate
            else:
                thr = ordered[min(int((1 - r) * n), n - 1)]
            realised = (sum(1 for s in signals if thr is not None and s > thr) / n
                        if thr is not None else 0.0)
            v = should_cascade(mean_small, mean_large, realised)
            rows.append({"target_rate": r,
                         "threshold": thr,
                         "realised_rate": round(realised, 4),
                         "predicted_saving": v["predicted_saving"]})

        best = max(rows, key=lambda x: x["predicted_saving"])
        if ratio >= 1.0:
            rec = ("the small tier is not cheaper than the large tier on this "
                   "traffic, so do not cascade: use the large tier alone")
        elif best["target_rate"] == 0.0:
            rec = (f"cheapest option is the small tier alone, saving "
                   f"{best['predicted_saving'] * 100:.1f}%. Escalate only as far "
                   f"as your own quality measurements justify; every point of "
                   f"escalation costs about a point of saving, and all saving "
                   f"is gone at {(1 - ratio) * 100:.0f}%")
        else:
            rec = (f"escalate up to {best['target_rate']:.0%} for a "
                   f"{best['predicted_saving'] * 100:.1f}% saving")

        result = {"queries_audited": n,
                  "mean_small_carbon_g": round(mean_small, 8),
                  "mean_large_carbon_g": round(mean_large, 8),
                  "cost_ratio_small_over_large": round(ratio, 4),
                  "break_even_escalation_rate": round(1 - ratio, 4),
                  "rows": rows,
                  "recommendation": rec}

        if verbose:
            print(f"\n  audited {n} queries on your own traffic")
            print(f"  cost ratio small/large   {ratio:.4f}")
            print(f"  break-even escalation    {(1 - ratio) * 100:.1f}%\n")
            print(f"  {'escalate':>9s}{'threshold':>12s}{'saving':>10s}")
            for row in rows:
                t = "never" if row["threshold"] is None else f"{row['threshold']:.3f}"
                print(f"  {row['realised_rate']:>8.0%}{t:>12s}"
                      f"{row['predicted_saving'] * 100:>9.1f}%")
            print(f"\n  {rec}\n")
        return result

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
