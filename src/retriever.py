"""Tiny TF-IDF retriever for banking policy documents."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS = [
    {"id": "D001", "title": "Savings Account FAQ",
     "text": "Our regular savings account requires a minimum balance of 10000 rupees in metro and 5000 in semi-urban. Interest is paid at 3 percent per annum on balances up to 1 lakh and 3.5 percent above."},
    {"id": "D002", "title": "Fee Schedule",
     "text": "NEFT is free via netbanking. RTGS is free via netbanking. Cheque return for insufficient funds costs 500 inward and 350 outward. Duplicate passbook costs 100 rupees."},
    {"id": "D003", "title": "Home Loan Brochure",
     "text": "Home loans up to 10 crore. Tenure 5 to 30 years. LTV is 80 percent for loans up to 30 lakh and 75 percent above. Interest rate 8.4 percent floating for salaried. Processing fee is 0.5 percent of the loan amount."},
    {"id": "D004", "title": "Personal Loan",
     "text": "Personal loans from 50000 to 25 lakh. Tenure 1 to 5 years. Interest rate 11 percent for credit score 700 to 749. Minimum credit score required is 650. No prepayment charges after 12 months."},
    {"id": "D005", "title": "Fixed Deposit Rates",
     "text": "FD interest rates: 1 year tenure 6.8 percent, 3 year tenure 7 percent, 5 year tenure 7 percent. Senior citizens get an additional 0.5 percent across all tenures."},
    {"id": "D006", "title": "Card Fraud Protocol",
     "text": "Report a fraudulent transaction within 3 working days for zero liability. Beyond 7 days customer is fully liable. Always freeze the card before raising the dispute."},
]

_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_doc_vectors = _vectorizer.fit_transform([d["title"] + " " + d["text"] for d in CORPUS])


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Return the top-k documents ranked by cosine similarity to the query."""
    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _doc_vectors)[0]
    ranked = sorted(enumerate(sims), key=lambda x: -x[1])
    return [{**CORPUS[i], "score": float(s)} for i, s in ranked[:top_k] if s > 0.05]
