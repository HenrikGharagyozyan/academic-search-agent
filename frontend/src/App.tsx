import { useState } from "react";
import { askQuestion } from "./api/research";
import type { Answer } from "./types/answer";
import { ClaimText } from "./components/ClaimText";
import { buildCitationNumbers } from "./utils/citations";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      const result = await askQuestion(question);
      setAnswer(result);
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const citationNumbers = answer ? buildCitationNumbers(answer) : new Map();

  return (
    <div
      style={{
        maxWidth: 720,
        margin: "0 auto",
        padding: "64px 24px",
      }}
    >
      <header style={{ textAlign: "center", marginBottom: 40 }}>
        <h1
          style={{
            fontSize: 32,
            fontWeight: 700,
            margin: 0,
            letterSpacing: "-0.02em",
          }}
        >
          Academic Search
        </h1>
        <p style={{ color: "var(--color-text-muted)", marginTop: 8 }}>
          Ask a research question, get a grounded answer with citations.
        </p>
      </header>

      <form onSubmit={handleSubmit}>
        <div
          style={{
            display: "flex",
            gap: 8,
            background: "var(--color-surface)",
            padding: 8,
            borderRadius: "var(--radius)",
            border: "1px solid var(--color-border)",
            boxShadow: "var(--shadow-sm)",
          }}
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. How does gradient descent converge for convex functions?"
            style={{
              flex: 1,
              border: "none",
              outline: "none",
              padding: "10px 12px",
              fontSize: 15,
              background: "transparent",
              color: "var(--color-text)",
            }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              border: "none",
              background: loading ? "var(--color-text-muted)" : "var(--color-accent)",
              color: "white",
              padding: "10px 20px",
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              cursor: loading ? "default" : "pointer",
              transition: "background 0.15s",
            }}
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
      </form>

      {error && (
        <p style={{ color: "#dc2626", marginTop: 16, fontSize: 14 }}>{error}</p>
      )}

      {answer && (
        <div
          style={{
            marginTop: 32,
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius)",
            boxShadow: "var(--shadow-sm)",
            padding: "28px 32px",
          }}
        >
          {answer.claims.length === 0 ? (
            <p style={{ color: "var(--color-text-muted)", margin: 0 }}>
              Not enough evidence was found to answer this question.
            </p>
          ) : (
            answer.claims.map((claim, i) => (
              <ClaimText
                key={i}
                claim={claim}
                evidence={answer.evidence}
                citationNumbers={citationNumbers}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default App;