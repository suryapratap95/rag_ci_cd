"""Run the golden dataset through the RAG agent and emit metrics.

Output: eval/results.json with per-question and aggregate metrics.
Exit code:
  0 = all gates passed (deploy allowed)
  1 = one or more gates failed (block deploy)
"""
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent import answer
from src.retriever import retrieve

# ── Gate thresholds — fail the run if any of these aren't met ──
GATES = {
    "retrieval_recall_at_3":  0.85,    # ≥ 85% of questions get the right doc in top-3
    "answer_correctness":     0.80,    # ≥ 80% of answers contain expected facts
    "forbidden_phrase_rate":  0.0,     # 0% — never allowed
    "p95_latency_ms":         8000,    # ≤ 8s p95
    "avg_cost_per_query_usd": 0.005,   # ≤ $0.005 average
}

# Sonnet 4.5 token prices ($ per million tokens)
PRICE_INPUT_PER_MTOK  = 3.0
PRICE_OUTPUT_PER_MTOK = 15.0


def run_eval():
    dataset = json.loads((Path(__file__).parent / "golden_dataset.json").read_text())
    per_question = []

    for tc in dataset:
        # 1. Retrieval check (cheap — runs locally without LLM)
        retrieved = retrieve(tc["question"], top_k=3)
        retrieved_ids = [d["id"] for d in retrieved]
        recall_at_3 = 1 if tc["expected_doc_id"] in retrieved_ids else 0

        # 2. End-to-end answer
        result = answer(tc["question"])
        ans_lower = result["answer"].lower()

        # Correctness — at least one expected fact must appear
        contains_fact = any(f.lower() in ans_lower for f in tc["expected_facts"])
        # Safety — none of the forbidden phrases may appear
        contains_forbidden = any(p.lower() in ans_lower for p in tc.get("forbidden_phrases", []))

        cost_usd = (
            result["input_tokens"]  * PRICE_INPUT_PER_MTOK  / 1_000_000 +
            result["output_tokens"] * PRICE_OUTPUT_PER_MTOK / 1_000_000
        )

        per_question.append({
            "id":              tc["id"],
            "recall_at_3":     recall_at_3,
            "correct":         int(contains_fact),
            "forbidden_hit":   int(contains_forbidden),
            "latency_ms":      result["latency_ms"],
            "cost_usd":        round(cost_usd, 6),
            "answer_preview":  result["answer"][:80],
            "retrieved":       retrieved_ids,
        })

    # ── Aggregate metrics ──
    n = len(per_question)
    metrics = {
        "retrieval_recall_at_3":  sum(q["recall_at_3"] for q in per_question) / n,
        "answer_correctness":     sum(q["correct"]     for q in per_question) / n,
        "forbidden_phrase_rate":  sum(q["forbidden_hit"] for q in per_question) / n,
        "p50_latency_ms":         int(statistics.median(q["latency_ms"] for q in per_question)),
        "p95_latency_ms":         int(sorted(q["latency_ms"] for q in per_question)[int(0.95*n)-1]),
        "avg_cost_per_query_usd": round(sum(q["cost_usd"] for q in per_question) / n, 6),
        "total_cost_usd":         round(sum(q["cost_usd"] for q in per_question), 6),
        "n_questions":            n,
    }

    # ── Gate evaluation ──
    gate_results = {}
    # Higher-is-better gates: recall, correctness
    for k in ["retrieval_recall_at_3", "answer_correctness"]:
        gate_results[k] = {"value": metrics[k], "threshold": GATES[k],
                           "passed": metrics[k] >= GATES[k]}
    # Lower-is-better gates: forbidden, latency, cost
    for k in ["forbidden_phrase_rate", "p95_latency_ms", "avg_cost_per_query_usd"]:
        gate_results[k] = {"value": metrics[k], "threshold": GATES[k],
                           "passed": metrics[k] <= GATES[k]}

    all_passed = all(g["passed"] for g in gate_results.values())

    report = {
        "timestamp":    os.environ.get("GITHUB_SHA", "local"),
        "metrics":      metrics,
        "gates":        gate_results,
        "all_passed":   all_passed,
        "per_question": per_question,
    }

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(report, indent=2))

    # Print a human-readable summary
    print("\n" + "="*60)
    print("RAG EVALUATION REPORT")
    print("="*60)
    for k, g in gate_results.items():
        icon = "✅" if g["passed"] else "❌"
        op = "≥" if k in ("retrieval_recall_at_3", "answer_correctness") else "≤"
        print(f"  {icon} {k:<28} = {g['value']:<10} (gate {op} {g['threshold']})")
    print("="*60)
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_eval())
