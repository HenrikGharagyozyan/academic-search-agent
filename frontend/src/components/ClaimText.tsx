import type { Answer } from "../types/answer";
import { Citation } from "./Citation";

interface ClaimTextProps {
  claim: Answer["claims"][number];
  evidence: Answer["evidence"];
  citationNumbers: Map<string, number>;
}

const CONFIDENCE_STYLES: Record<
  Answer["claims"][number]["confidence"],
  { bg: string; color: string; label: string }
> = {
  high: { bg: "#dcfce7", color: "#166534", label: "High confidence" },
  medium: { bg: "#fef3c7", color: "#92400e", label: "Medium confidence" },
  low: { bg: "#fee2e2", color: "#991b1b", label: "Low confidence" },
};

export function ClaimText({ claim, evidence, citationNumbers }: ClaimTextProps) {
  const confidenceStyle = CONFIDENCE_STYLES[claim.confidence];

  return (
    <p
      style={{
        marginBottom: 18,
        lineHeight: 1.7,
        fontSize: 16,
        color: "var(--color-text)",
      }}
    >
      {claim.confidence !== "high" && (
        <span
          title={confidenceStyle.label}
          style={{
            display: "inline-block",
            fontSize: 11,
            fontWeight: 600,
            padding: "1px 6px",
            borderRadius: 4,
            marginRight: 6,
            background: confidenceStyle.bg,
            color: confidenceStyle.color,
            verticalAlign: "middle",
          }}
        >
          {claim.confidence.toUpperCase()}
        </span>
      )}
      {claim.text}{" "}
      {claim.evidence_ids.map((id) => {
        const item = evidence[id];
        const number = citationNumbers.get(id);
        if (!item || number === undefined) return null;
        return <Citation key={id} index={number} evidence={item} />;
      })}
    </p>
  );
}