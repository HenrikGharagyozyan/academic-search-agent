import { useState } from "react";
import type { ActivityStep } from "../types/answer";

interface ActivityLogProps {
  steps: ActivityStep[];
  /** Open by default while the run is still going, collapsed once it is done. */
  defaultOpen?: boolean;
}

/** A glyph per kind, so the trail can be skimmed without reading every line. */
const MARK: Record<ActivityStep["kind"], string> = {
  search: "⌕",
  source_found: "·",
  scrape_ok: "✓",
  scrape_failed: "✕",
  collect: "▤",
  select: "▤",
  grade_relevance: "◈",
  generate: "✎",
  verify: "⚖",
  grade_answer: "◉",
  refine: "↻",
};

const FAILED: ActivityStep["kind"][] = ["scrape_failed"];

function markColor(kind: ActivityStep["kind"]): string {
  if (FAILED.includes(kind)) return "#b91c1c";
  if (kind === "scrape_ok") return "#15803d";
  return "var(--color-text-muted)";
}

function StepRow({ step }: { step: ActivityStep }) {
  // A source is worth following; everything else is just a line of text.
  const body = step.url ? (
    <a
      href={step.url}
      target="_blank"
      rel="noopener noreferrer"
      style={{ color: "var(--color-accent)", textDecoration: "none" }}
      title={step.title ?? step.url}
    >
      {step.label}
    </a>
  ) : (
    <span>{step.label}</span>
  );

  return (
    <li style={{ display: "flex", gap: 8, alignItems: "baseline", padding: "3px 0" }}>
      <span
        aria-hidden
        style={{
          color: markColor(step.kind),
          fontSize: 12,
          width: 14,
          flexShrink: 0,
          textAlign: "center",
        }}
      >
        {MARK[step.kind] ?? "·"}
      </span>
      <span style={{ fontSize: 13, lineHeight: 1.5, minWidth: 0 }}>
        {body}
        {step.detail && (
          <span style={{ color: "var(--color-text-muted)" }}> — {step.detail}</span>
        )}
      </span>
    </li>
  );
}

export function ActivityLog({ steps, defaultOpen = false }: ActivityLogProps) {
  const [open, setOpen] = useState(defaultOpen);

  if (steps.length === 0) return null;

  // The agent may search more than once; grouping by pass makes a second
  // attempt legible instead of looking like the first one repeating itself.
  const passes = [...new Set(steps.map((s) => s.attempt))].sort((a, b) => a - b);
  const sources = new Set(steps.filter((s) => s.kind === "source_found").map((s) => s.url));
  const read = steps.filter((s) => s.kind === "scrape_ok").length;

  return (
    <div
      style={{
        marginTop: 20,
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius)",
        background: "var(--color-surface)",
        overflow: "hidden",
      }}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "10px 14px",
          border: "none",
          background: "transparent",
          cursor: "pointer",
          font: "inherit",
          fontSize: 13,
          color: "var(--color-text-muted)",
          textAlign: "left",
        }}
      >
        <span aria-hidden style={{ fontSize: 10, width: 10 }}>
          {open ? "▾" : "▸"}
        </span>
        <span style={{ fontWeight: 600 }}>How this answer was found</span>
        <span>
          {sources.size} source{sources.size === 1 ? "" : "s"} found, {read} read
          {passes.length > 1 && `, ${passes.length} searches`}
        </span>
      </button>

      {open && (
        <div style={{ padding: "0 14px 12px 14px" }}>
          {passes.map((pass) => (
            <div key={pass}>
              {passes.length > 1 && (
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    color: "var(--color-text-muted)",
                    margin: "10px 0 4px",
                  }}
                >
                  Attempt {pass + 1}
                </div>
              )}
              <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
                {steps
                  .filter((s) => s.attempt === pass)
                  .map((step, i) => (
                    <StepRow key={`${pass}-${i}`} step={step} />
                  ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
