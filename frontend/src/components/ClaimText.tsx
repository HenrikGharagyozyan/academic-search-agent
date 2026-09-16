import type { Answer } from "../types/answer";
import { Citation } from "./Citation";

interface ClaimTextProps {
  claim: Answer["claims"][number];
  evidence: Answer["evidence"];
  citationNumbers: Map<string, number>;
}

export function ClaimText({ claim, evidence, citationNumbers }: ClaimTextProps) {
  return (
    <p
      style={{
        marginBottom: 18,
        lineHeight: 1.7,
        fontSize: 16,
        color: "var(--color-text)",
      }}
    >
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