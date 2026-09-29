"""Interactive demonstration: ask questions, watch the router decide.

Each query is answered by the small local model first. Its calibrated
entropy is compared against the threshold; if the model is uncertain the
query escalates to the large tier, and the discarded small-model run is
charged to that query under full accounting.

    python demo_live.py                       # small local, large via API
    python demo_live.py --offline             # both tiers local, no network
    python demo_live.py --budget 0.5          # enforce a carbon budget

Type a question and press enter. Type 'profile' for the session report,
'preset' to run a set of prepared questions, or 'quit' to finish.
"""
import argparse
import sys
import time

BAR = "-" * 72

PRESET = [
    "What is the capital of France?",
    "Convert 45 degrees Celsius to Fahrenheit.",
    "What does the HTTP status code 404 mean?",
    "Write a SQL query to find the second highest salary in a table.",
    "Explain why transformer attention is quadratic in sequence length, "
    "and what has been proposed to reduce it.",
    "Design a caching strategy for a news site whose traffic spikes "
    "unpredictably, and justify the eviction policy you choose.",
]


def show(q, r, n):
    esc = r.decision.startswith("ESCALATE")
    who = r.large_model if esc else r.small_model
    print(f"\n{BAR}\n[{n}] {q}")
    print(f"{BAR}")
    print(f"  entropy    {r.signal:.3f} bits   threshold "
          f"{'%.3f' % r.threshold if r.threshold is not None else 'warming up'}")
    print(f"  decision   {r.decision}   ->  answered by {who}")
    if esc:
        print(f"  accounting {r.carbon_g:.6f} g total, of which "
              f"{r.wasted_carbon_g:.6f} g was the discarded small-model run")
    else:
        print(f"  accounting {r.carbon_g:.6f} g total, nothing discarded")
    print(f"  latency    {r.latency_s:.2f} s")
    print(f"\n  {r.response.strip()[:400]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--large", default="gpt-4o-mini")
    ap.add_argument("--offline", action="store_true",
                    help="use a local large tier instead of the API")
    ap.add_argument("--mode", default="balanced",
                    choices=["quality", "balanced", "green"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--budget", type=float, default=None,
                    help="carbon budget in grams per hour")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    if args.offline and args.large == "gpt-4o-mini":
        args.large = "Qwen/Qwen2.5-1.5B-Instruct"

    import greengate
    print(f"\n  GreenGate {greengate.__version__}")
    print(f"  small tier : {args.small}")
    print(f"  large tier : {args.large}"
          f"{' (local)' if args.offline else ' (API, carbon estimated)'}")
    print(f"  mode       : {args.mode}"
          + (f" | budget {args.budget} g/h" if args.budget else ""))
    print("\n  loading models, this takes a moment on first run ...")

    t0 = time.perf_counter()
    gw = greengate.GreenGate(small=args.small, large=args.large,
                             mode=args.mode, threshold=args.threshold,
                             budget_g=args.budget,
                             max_new_tokens=args.max_new_tokens)
    print(f"  ready in {time.perf_counter() - t0:.1f} s")
    print("\n  The first few queries calibrate the automatic threshold from the"
          "\n  entropy it has seen, so early decisions may all be local.")
    print("\n  Commands: 'preset' runs prepared questions, 'profile' shows the"
          "\n  session report, 'quit' exits.\n")

    n = 0
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            q = "quit"
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "profile":
            p = gw.profile()
            print(f"\n{BAR}\n  SESSION PROFILE\n{BAR}")
            for k, v in p.items():
                print(f"  {k:20s} {v}")
            cut = None
            if p["queries"] and p["escalation_rate"] > 0:
                print(f"\n  Of {p['total_carbon_g']} g emitted, "
                      f"{p['wasted_carbon_g']} g was spent on small-model runs "
                      f"that were discarded on escalation. Conventional "
                      f"reporting would omit that.")
            print()
            continue
        if q.lower() == "preset":
            for pq in PRESET:
                n += 1
                show(pq, gw.route(pq), n)
            continue
        n += 1
        show(q, gw.route(q), n)

    p = gw.profile()
    print(f"\n{BAR}\n  FINAL: {p['queries']} queries, "
          f"{p['escalation_rate']:.0%} escalated, "
          f"{p['total_carbon_g']} g CO2 "
          f"({p['wasted_carbon_g']} g of it discarded work)\n{BAR}\n")


if __name__ == "__main__":
    sys.exit(main())
