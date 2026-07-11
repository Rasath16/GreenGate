"""
GreenGate Demo — Shannon Entropy Routing
=========================================
Runs test queries through the GreenGate router to demonstrate
entropy-based routing decisions, carbon tracking, and gQoS computation.

Uses GPT-2 on CPU (zero cost). When moving to GPU, swap the model name
to 'mistralai/Mistral-7B-Instruct-v0.2' — the routing logic is identical.
"""

from greengate import GreenGateRouter


def main():
    print("=" * 65)
    print("  GreenGate — Confidence-Aware Cascading Router Demo")
    print("  Model: GPT-2 (CPU)  |  Threshold: 2.5")
    print("=" * 65)
    print()

    router = GreenGateRouter(
        small_model="gpt2",
        threshold=2.5,
        max_new_tokens=60,
    )

    queries = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Explain the philosophical implications of Godel's incompleteness theorems.",
        "Write a Python function to implement quicksort with detailed comments.",
        "What color is the sky?",
    ]

    print(f"Routing {len(queries)} queries...\n")

    for q in queries:
        result = router.route(q)
        decision_marker = "LOCAL" if result.decision == "ANSWER" else "ESCALATE"
        print(f"Query:    {result.query}")
        print(f"Entropy:  {result.entropy:.4f}  |  Decision: {decision_marker}")
        print(f"Energy:   {result.energy_joules:.2f}J  |  Carbon: {result.total_carbon_grams:.6f}g CO2")
        if result.decision == "ESCALATE":
            wasted = result.total_carbon_grams - result.carbon_grams
            print(f"          (includes {wasted:.6f}g wasted on small model — full accounting)")
        print(f"gQoS:     {result.gqos:.2f} accuracy/gCO2")
        print("-" * 65)

    # Summary
    summary = router.summary()
    print()
    print("=" * 65)
    print("  SESSION SUMMARY")
    print("=" * 65)
    print(f"  Total queries:       {summary['total_queries']}")
    print(f"  Answered locally:    {summary['answered_locally']}")
    print(f"  Escalated:           {summary['escalated']}")
    print(f"  Escalation rate:     {summary['escalation_rate']:.0%}")
    print(f"  Total energy:        {summary['total_energy_joules']:.2f} J")
    print(f"  Total carbon:        {summary['total_carbon_grams']:.6f} g CO2")
    print(f"  Wasted carbon:       {summary['wasted_carbon_grams']:.6f} g CO2")
    print(f"  Profiling mode:      {summary['mode']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
