from app.providers.gemini_provider import GeminiProvider

if __name__ == "__main__":
    provider = GeminiProvider()

    question = "How does gradient descent converge for convex functions?"
    evidence_chunks = [
        {
            "chunk_id": "ev_1",
            "text": (
                "When the function f is convex, all local minima are also "
                "global minima, so in this case gradient descent can converge "
                "to the global solution."
            ),
        },
        {
            "chunk_id": "ev_2",
            "text": (
                "If the objective is assumed to be strongly convex and "
                "Lipschitz smooth, then gradient descent converges linearly "
                "with a fixed step size."
            ),
        },
    ]

    claims = provider.generate_claims(question, evidence_chunks)

    for claim in claims:
        print(f"- {claim.text}")
        print(f"  evidence_ids: {claim.evidence_ids}")
        print(f"  confidence: {claim.confidence}")
        print()