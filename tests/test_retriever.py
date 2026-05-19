"""Unit tests for the retriever. No LLM calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import CORPUS, retrieve


def test_retrieve_returns_list():
    results = retrieve("savings account interest")
    assert isinstance(results, list)
    assert len(results) > 0


def test_retrieve_relevant_doc_for_savings():
    results = retrieve("what is the interest on a savings account")
    ids = [r["id"] for r in results]
    assert "D001" in ids, f"Expected D001 (Savings FAQ) in top results, got {ids}"


def test_retrieve_relevant_doc_for_home_loan():
    results = retrieve("home loan interest rate")
    ids = [r["id"] for r in results]
    assert "D003" in ids, f"Expected D003 (Home Loan) in top results, got {ids}"


def test_retrieve_relevant_doc_for_fraud():
    results = retrieve("fraud transaction reporting")
    ids = [r["id"] for r in results]
    assert "D006" in ids, f"Expected D006 (Card Fraud) in top results, got {ids}"


def test_top_k_caps_results():
    results = retrieve("loan", top_k=2)
    assert len(results) <= 2


def test_low_similarity_filtered():
    # A query totally unrelated to banking should retrieve nothing (above the threshold)
    results = retrieve("butterflies and rainbows in springtime")
    assert len(results) == 0, f"Expected no matches, got {results}"


def test_results_have_scores():
    results = retrieve("FD interest rate")
    assert all("score" in r and r["score"] > 0 for r in results)


def test_results_sorted_by_score():
    results = retrieve("home loan eligibility tenure")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "results must be sorted by score desc"


def test_corpus_well_formed():
    """Sanity check: every corpus entry has the fields the agent expects."""
    for doc in CORPUS:
        assert set(doc.keys()) >= {"id", "title", "text"}
        assert doc["id"].startswith("D")
        assert len(doc["text"]) > 20
