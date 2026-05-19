"""Compare current eval results to the committed baseline.

Block if any metric regresses beyond the tolerance.
If no baseline exists yet (first run), this is a no-op pass.
"""
import json
import sys
from pathlib import Path

# Tolerance per metric (regression budget)
MAX_REGRESSION = {
    "retrieval_recall_at_3":  0.02,    # accept tiny noise, block real drops
    "answer_correctness":     0.05,
    "p95_latency_ms":         2000,    # +2s p95 is suspicious
    "avg_cost_per_query_usd": 0.001,
}

HIGHER_IS_BETTER = {"retrieval_recall_at_3", "answer_correctness"}

def main():
    here = Path(__file__).parent
    current = json.loads((here / "results.json").read_text())["metrics"]
    baseline_path = here / "baseline.json"

    if not baseline_path.exists():
        print("⚠️  No baseline.json — skipping comparison. The current results will become the baseline after first merge.")
        return 0

    baseline = json.loads(baseline_path.read_text())["metrics"]

    print("\n" + "="*70)
    print(f"{'METRIC':<28} {'BASELINE':>12} {'CURRENT':>12} {'DELTA':>12}")
    print("="*70)

    regressions = []
    for metric, tolerance in MAX_REGRESSION.items():
        b = baseline.get(metric)
        c = current.get(metric)
        if b is None or c is None:
            continue
        delta = c - b

        # Determine whether the delta is a regression
        if metric in HIGHER_IS_BETTER:
            regressed = delta < -tolerance        # dropped too much
        else:
            regressed = delta > tolerance         # grew too much

        icon = "❌" if regressed else "✅"
        print(f"  {icon} {metric:<25} {b:>12.4f} {c:>12.4f} {delta:>+12.4f}")
        if regressed:
            regressions.append({"metric": metric, "baseline": b, "current": c, "delta": delta})

    print("="*70)
    if regressions:
        print(f" {len(regressions)} REGRESSION(S) DETECTED — blocking deployment")
        return 1
    print("✅ No regressions beyond tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
