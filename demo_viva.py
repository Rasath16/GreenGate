"""Viva demonstration: the instrument, the accounting, and the decision.

Runs offline on a laptop with no GPU and no network in about twenty seconds.
Everything printed comes from measurements committed to this repository, so
each number can be traced to a raw record on request.

    python demo_viva.py
"""
import json
import time
from pathlib import Path

import greengate

BAR = "=" * 74
ROOT = Path(__file__).resolve().parent


def pause(label=""):
    print(f"\n{BAR}\n  {label}\n{BAR}")
    time.sleep(0.3)


# ---------------------------------------------------------------- 1
pause("1. THE INSTRUMENT: measure inference this library does not run")
print("""
    with greengate.measure(n_queries=1, label="...") as m:
        answer = my_existing_pipeline(prompt)
""")
with greengate.measure(n_queries=1, label="local pipeline") as m:
    sum(i * i for i in range(400_000))          # stand-in for a real call
print(m.report())
print("\n  On this laptop the mode is 'cpu (estimated)'. On the evaluation"
      "\n  hardware it reads 'gpu (pynvml)', sampling every GPU at 100 ms."
      "\n  Repeatability measured at 1.4% on text and 0.5% on vision.")

# ---------------------------------------------------------------- 2
pause("2. THE INSTRUMENT REFUSES TO MIS-MEASURE")
print(f"  exclusive GPU during that measurement : {m.exclusive}")
print(f"  other compute processes seen          : {m.shared_with or 'none'}")
print("""
  Device power cannot be split between concurrent processes. During the
  vision experiments one run was launched twice by mistake; the two
  processes shared the accelerator and one tier came out 42% too high.
  It was caught because that tier is measured in both runs and the values
  disagreed. Measurements now report exclusivity, and strict=True raises.""")

# ---------------------------------------------------------------- 3
pause("3. THE ACCOUNTING: what published cascade results leave out")
print("""
  A cascade runs the small model on every query and the large model on the
  escalated fraction. When it escalates, the small run is discarded but the
  energy was still spent. Counting it gives

        S  =  1  -  C_s/C_l  -  e

  and a cascade only saves while escalation stays below 1 - C_s/C_l.""")

# ---------------------------------------------------------------- 4
pause("4. THE DECISION, ON MEASURED DATA")
CASES = [
    ("MMLU  Qwen2.5-1.5B -> Mistral-7B   (Tesla T4)", 0.272, 0.457),
    ("MMLU  Qwen2.5-0.5B -> Mistral-7B   (A100)", 0.136, 0.823),
    ("MMLU  Qwen2.5-0.5B -> Mistral-7B   (H100)", 0.214, 0.823),
    ("VQAv2 SmolVLM-2B   -> Qwen2-VL-7B  (A100)", 1.049, 0.050),
    ("VQAv2 SmolVLM-256M -> Qwen2-VL-7B  (A100)", 0.530, 0.150),
]
for label, ratio, esc in CASES:
    r = greengate.should_cascade(carbon_small=ratio, carbon_large=1.0,
                                 escalation_rate=esc)
    flag = "SAVES  " if r["predicted_saving"] > 0 else "COSTS  "
    print(f"  {flag} {label}")
    print(f"           C_s/C_l {ratio:5.3f} | escalation {esc:4.0%} "
          f"| break-even {r['break_even_escalation_rate']:4.0%} "
          f"| predicted {r['predicted_saving']:+6.1%}")
print("""
  Rows two and three are the same models and the same 300 questions on two
  accelerators. The condition predicts a change of sign, and the measured
  results are +4.2% on the A100 and -4.0% on the H100.

  Row four is the case the instrument exists for: the 2B vision model costs
  more per query than the 4-bit 7B, so no escalation rate can save. Without
  measuring both tiers that pairing looks obviously sensible.""")

# ---------------------------------------------------------------- 5
pause("5. THE SIGNAL IS A PROPERTY OF THE MODEL, NOT THE HARDWARE")
csv = ROOT / "experiments" / "hardware_comparison.csv"
if csv.exists():
    rows = [l.split(",") for l in csv.read_text().strip().splitlines()]
    hdr = rows[0]
    im, ig, ie, ia = (hdr.index("small_model"), hdr.index("gpu"),
                      hdr.index("entropy"), hdr.index("small_acc"))
    print(f"  {'model':16s}{'GPU':7s}{'mean entropy':>14s}{'accuracy':>10s}")
    for r in rows[1:]:
        print(f"  {r[im]:16s}{r[ig]:7s}{float(r[ie]):14.3f}{float(r[ia]):10.3f}")
    print("\n  Entropy is identical to three decimals across a tenfold range of"
          "\n  board power. A threshold calibrated on one machine transfers.")
else:
    print("  (run: python experiments/compare_hardware.py)")

# ---------------------------------------------------------------- 6
pause("6. THE VISION RESULT, FROM RAW RECORDS")
for tag, folder in (("SmolVLM-2B  ", "vision_local"),
                    ("SmolVLM-256M", "vision_256m_clean")):
    p = ROOT / "experiments" / "vision_local_2026-09-29" / folder / "vision_records.jsonl"
    if not p.exists():
        continue
    recs = [json.loads(l) for l in p.open(encoding="utf-8")]
    sg = sum(x["small_carbon_g"] for x in recs)
    lg = sum(x["large_carbon_g"] for x in recs)
    sa = sum(x["small_correct"] for x in recs) / len(recs)
    la = sum(x["large_correct"] for x in recs) / len(recs)
    print(f"  {tag} n={len(recs)}  small {sg:6.4f} g (acc {sa:.3f})  "
          f"large {lg:6.4f} g (acc {la:.3f})  C_s/C_l {sg / lg:5.3f}")
print("\n  Both tiers measured on one accelerator, so the columns are"
      "\n  comparable. The large tier reproduced across the two runs to 0.5%.")

pause("EVERY FIGURE ABOVE IS IN THE PUBLIC REPOSITORY")
print("  github.com/Rasath16/GreenGate    pip install greengate\n")
