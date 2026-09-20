import { useEffect, useRef, useState } from "react";
import type { AnswerEvidence } from "../types/answer";

interface CitationProps {
  index: number;
  evidence: AnswerEvidence;
}

function truncateText(text: string, maxLines: number = 5): { shown: string; truncated: boolean } {
  const lines = text.split("\n");
  if (lines.length <= maxLines) {
    return { shown: text, truncated: false };
  }
  return { shown: lines.slice(0, maxLines).join("\n"), truncated: true };
}

export function Citation({ index, evidence }: CitationProps) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const containerRef = useRef<HTMLSpanElement>(null);

  const { shown, truncated } = truncateText(evidence.text);
  const displayText = expanded ? evidence.text : shown;

  // Close the popup when clicking anywhere outside this component
  useEffect(() => {
    if (!open) return;

    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  return (
    <span ref={containerRef} style={{ position: "relative", display: "inline-block" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          minWidth: 20,
          height: 20,
          padding: "0 5px",
          marginLeft: 3,
          border: "1px solid var(--color-border)",
          borderRadius: 999,
          background: open ? "var(--color-accent)" : "var(--color-accent-bg)",
          color: open ? "white" : "var(--color-accent)",
          fontSize: 11,
          fontWeight: 700,
          cursor: "pointer",
          verticalAlign: "middle",
          transition: "all 0.12s",
        }}
      >
        {index}
      </button>

      {open && (
        <div
          style={{
            position: "absolute",
            bottom: "calc(100% + 8px)",
            left: 0,
            zIndex: 10,
            width: 340,
            background: "var(--color-surface)",
            border: "1px solid var(--color-accent)",
            borderRadius: 10,
            boxShadow: "var(--shadow-md)",
            padding: 14,
            fontSize: 13,
            lineHeight: 1.5,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4, color: "var(--color-text)" }}>
            {evidence.title}
          </div>
          <div style={{ color: "var(--color-text-muted)", marginBottom: 10, fontSize: 12 }}>
            Lines {evidence.start_line}–{evidence.end_line}
          </div>
          <div
            style={{
              whiteSpace: "pre-wrap",
              marginBottom: 10,
              color: "var(--color-text)",
              background: "var(--color-accent-bg)",
              padding: "8px 10px",
              borderRadius: 6,
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
              fontSize: 12,
            }}
          >
            {displayText}
            {truncated && !expanded && "…"}
          </div>
          {truncated && (
            <button
              onClick={() => setExpanded((e) => !e)}
              style={{
                background: "none",
                border: "none",
                color: "var(--color-accent)",
                cursor: "pointer",
                padding: 0,
                fontSize: 12,
                fontWeight: 600,
                marginBottom: 10,
                display: "block",
              }}
            >
              {expanded ? "Show less" : "Show full quote"}
            </button>
          )}
          <a
            href={evidence.source_url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: "var(--color-accent)",
              fontSize: 12,
              fontWeight: 600,
              textDecoration: "none",
            }}
          >
            Open source →
          </a>
        </div>
      )}
    </span>
  );
}