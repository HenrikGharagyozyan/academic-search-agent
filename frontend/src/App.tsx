import { useState } from "react";
import { streamQuestion } from "./api/research";
import type { ActivityStep, Answer } from "./types/answer";
import { ActivityLog } from "./components/ActivityLog";
import { ClaimText } from "./components/ClaimText";
import { Prose } from "./components/Prose";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  // Collected live so the reader sees progress inside a slow stage. The finished
  // answer carries its own copy, which is what the panel shows afterwards.
  const [activity, setActivity] = useState<ActivityStep[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setStage(null);
    setActivity([]);

    let receivedTerminal = false;

    try {
      for await (const event of streamQuestion(question)) {
        if (event.type === "progress") {
          setStage(event.data.label);
        } else if (event.type === "activity") {
          setActivity((steps) => [...steps, event.data]);
        } else if (event.type === "result") {
          receivedTerminal = true;
          if (!event.data.claims?.length) {
            setError("It was not possible to find reliable enough sources for an answer. Try to reformulate the question.");
          } else {
            setAnswer(event.data);
          }
        } else if (event.type === "error") {
          receivedTerminal = true;
          setError(event.data.detail);
        }
      }
      if (!receivedTerminal) {
        setError("The connection was interrupted before an answer arrived. Please try again.");
      }
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
      setStage(null);
    }
  };

  // Distinct themes in the order they appear, so each section can be numbered.
  const themeOrder = answer
    ? [...new Set(answer.claims.map((c) => c.theme).filter(Boolean))]
    : [];

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
              color: "var(--color-on-accent)",
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
        <p style={{ color: "var(--color-danger)", marginTop: 16, fontSize: 14 }}>{error}</p>
      )}
      {loading && stage && (
        <div style={{ marginTop: 16, textAlign: "center" }}>
          <p style={{ color: "var(--color-text)", margin: 0, fontSize: 14 }}>{stage}…</p>
          {activity.length > 0 && (
            <p
              style={{
                color: "var(--color-text-muted)",
                margin: "4px 0 0",
                fontSize: 13,
                // The step list grows fast; a fixed line keeps the layout still.
                minHeight: 18,
              }}
            >
              {activity[activity.length - 1].label}
            </p>
          )}
        </div>
      )}
      {loading && activity.length > 0 && (
        <ActivityLog steps={activity} defaultOpen={false} />
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
            <>
                {!answer.evidence_sufficient && (
                <div
                  style={{
                    background: "var(--color-warning-bg)",
                    border: "1px solid var(--color-warning-border)",
                    borderRadius: 8,
                    padding: "10px 14px",
                    marginBottom: 20,
                    fontSize: 13,
                    color: "var(--color-warning-text)",
                    lineHeight: 1.5,
                  }}
                >
                  ⚠ The evidence found may not fully support this answer — treat these
                  claims as preliminary and verify against the sources.
                </div>
              )}
              {answer.summary && (
                <p
                  style={{
                    fontSize: 15,
                    color: "var(--color-text-muted)",
                    marginTop: 0,
                    marginBottom: 20,
                    fontStyle: "italic",
                    lineHeight: 1.6,
                  }}
                >
                  <Prose text={answer.summary} />
                </p>
              )}

              {answer.claims.map((claim, i) => {
                // A subheading appears where the theme changes, which turns a
                // flat list of claims into the shape of the field. Claims
                // sharing a theme arrive adjacent, so comparing with the
                // previous one is enough — no regrouping, and the model's
                // ordering is preserved.
                const startsTheme =
                  claim.theme !== "" && claim.theme !== answer.claims[i - 1]?.theme;

                return (
                  <div key={i}>
                    {startsTheme && (
                      <h3
                        style={{
                          fontSize: 18,
                          fontWeight: 650,
                          lineHeight: 1.35,
                          color: "var(--color-text)",
                          margin: i === 0 ? "0 0 10px" : "28px 0 10px",
                        }}
                      >
                        {/* A real space, not a margin: the heading has to read
                            "1. Error rates" when copied, not "1.Error rates". */}
                        <span style={{ color: "var(--color-text-muted)" }}>
                          {themeOrder.indexOf(claim.theme) + 1}.
                        </span>{" "}
                        {claim.theme}
                      </h3>
                    )}
                    <ClaimText
                      claim={claim}
                      evidence={answer.evidence}
                    />
                  </div>
                );
              })}

              {answer.conclusion && (
                <div
                  style={{
                    marginTop: 20,
                    paddingTop: 16,
                    borderTop: "1px solid var(--color-border)",
                  }}
                >
                  <div
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: "var(--color-text-muted)",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      marginBottom: 6,
                    }}
                  >
                    Conclusion
                  </div>
                  <div style={{ fontSize: 16 }}>
                    <Prose text={answer.conclusion} />
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {answer && <ActivityLog steps={answer.activity} />}
    </div>
  );
}

export default App;