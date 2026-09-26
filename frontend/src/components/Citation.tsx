import { useEffect, useRef, useState } from "react";
import type { AnswerEvidence } from "../types/answer";
import { computePopupPosition, type PopupPosition } from "../utils/popupPosition";

interface CitationProps {
  index: number;
  evidence: AnswerEvidence;
}

/**
 * Scraped markdown keeps the source page's inline HTML — table cells in
 * particular pack their line breaks as literal <br> tags. Rendering the quote
 * verbatim would show the tags, so they become the line breaks they stand for.
 */
function readableQuote(text: string): string {
  return text.replace(/<br\s*\/?>/gi, "\n");
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
  const [position, setPosition] = useState<PopupPosition | null>(null);
  const containerRef = useRef<HTMLSpanElement>(null);

  const quote = readableQuote(evidence.text);
  const { shown, truncated } = truncateText(quote);
  const displayText = expanded ? quote : shown;

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

  useEffect(() => {
    if (!open) return;

    function reposition() {
      const marker = containerRef.current?.getBoundingClientRect();
      if (marker) setPosition(computePopupPosition(marker));
    }

    reposition();
    window.addEventListener("resize", reposition);
    // Capture phase so scrolling inside any ancestor keeps the popup on its marker.
    window.addEventListener("scroll", reposition, true);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
    };
  }, [open, expanded]);

  function toggleOpen() {
    setOpen((wasOpen) => {
      // A reopened popup starts collapsed again.
      if (wasOpen) setExpanded(false);
      return !wasOpen;
    });
  }

  return (
    <span ref={containerRef} style={{ position: "relative", display: "inline-block" }}>
      <button
        onClick={toggleOpen}
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

      {open && position && (
        <div
          style={{
            position: "fixed",
            left: position.left,
            top: position.top,
            bottom: position.bottom,
            zIndex: 10,
            width: position.width,
            maxHeight: position.maxHeight,
            overflowY: "auto",
            overscrollBehavior: "contain",
            background: "var(--color-surface)",
            border: "1px solid var(--color-accent)",
            borderRadius: 10,
            boxShadow: "var(--shadow-md)",
            padding: 14,
            fontSize: 13,
            lineHeight: 1.5,
          }}
        >
          <div
            style={{
              fontWeight: 600,
              marginBottom: 4,
              color: "var(--color-text)",
              overflowWrap: "anywhere",
            }}
          >
            {evidence.title}
          </div>
          <div style={{ color: "var(--color-text-muted)", marginBottom: 10, fontSize: 12 }}>
            Lines {evidence.start_line}–{evidence.end_line}
          </div>
          <div
            style={{
              whiteSpace: "pre-wrap",
              // Raw markdown quotes carry long unbreakable URLs that would
              // otherwise stretch the popup far past its width.
              overflowWrap: "anywhere",
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