import type { Answer } from "../types/answer";
import { Citation } from "./Citation";
import { Prose } from "./Prose";
import { groupBySource } from "../utils/sources";

interface ClaimTextProps {
  claim: Answer["claims"][number];
  evidence: Answer["evidence"];
}

const CONFIDENCE_STYLES: Record<
  Answer["claims"][number]["confidence"],
  { bg: string; color: string; label: string }
> = {
  high: { bg: "#dcfce7", color: "#166534", label: "High confidence" },
  medium: { bg: "#fef3c7", color: "#92400e", label: "Medium confidence" },
  low: { bg: "#fee2e2", color: "#991b1b", label: "Low confidence" },
};

export function ClaimText({ claim, evidence }: ClaimTextProps) {
  const confidenceStyle = CONFIDENCE_STYLES[claim.confidence];
  // One chip per page, not per passage: three passages from one article are
  // one source to the reader, and the popup lists all three.
  const sources = groupBySource(
    claim.evidence_ids.map((id) => evidence[id]).filter((item) => item !== undefined)
  );

  return (
    <div
      style={{
        marginBottom: 18,
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
      <Prose
        text={claim.text}
        trailing={sources.map((passages) => (
          <Citation key={passages[0].source_url} passages={passages} />
        ))}
      />
    </div>
  );
}