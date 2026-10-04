import { useEffect, useRef, useState } from "react";
import type { AnswerEvidence } from "../types/answer";
import { computePopupPosition, type PopupPosition } from "../utils/popupPosition";
import { faviconUrl, siteName } from "../utils/sources";

interface CitationProps {
  /** Every passage the claim cites from one page. */
  passages: AnswerEvidence[];
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

/** One cited passage, collapsed to its first lines until asked for more. */
function Quote({ passage, onResize }: { passage: AnswerEvidence; onResize: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const quote = readableQuote(passage.text);
  const { shown, truncated } = truncateText(quote);

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ color: "var(--color-text-muted)", marginBottom: 4, fontSize: 12 }}>
        Lines {passage.start_line}–{passage.end_line}
      </div>
      <div
        style={{
          whiteSpace: "pre-wrap",
          // Raw markdown quotes carry long unbreakable URLs that would
          // otherwise stretch the popup far past its width.
          overflowWrap: "anywhere",
          color: "var(--color-text)",
          background: "var(--color-accent-bg)",
          padding: "8px 10px",
          borderRadius: 6,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          fontSize: 12,
        }}
      >
        {expanded ? quote : shown}
        {truncated && !expanded && "…"}
      </div>
      {truncated && (
        <button
          onClick={() => {
            setExpanded((e) => !e);
            onResize();
          }}
          style={{
            background: "none",
            border: "none",
            color: "var(--color-accent)",
            cursor: "pointer",
            padding: 0,
            marginTop: 4,
            fontSize: 12,
            fontWeight: 600,
            display: "block",
          }}
        >
          {expanded ? "Show less" : "Show full quote"}
        </button>
      )}
    </div>
  );
}

/** The site's icon, or its initial when the icon cannot be fetched. */
function SiteIcon({ url, name }: { url: string; name: string }) {
  const [failed, setFailed] = useState(false);
  const size = 12;

  if (failed) {
    return (
      <span
        aria-hidden
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: size,
          height: size,
          borderRadius: 3,
          background: "var(--color-border)",
          color: "var(--color-text)",
          fontSize: 8,
          fontWeight: 700,
          flexShrink: 0,
        }}
      >
        {name.charAt(0).toUpperCase()}
      </span>
    );
  }
  return (
    <img
      src={faviconUrl(url)}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      onError={() => setFailed(true)}
      style={{ borderRadius: 2, flexShrink: 0 }}
    />
  );
}

export function Citation({ passages }: CitationProps) {
  const [open, setOpen] = useState(false);
  // Bumped when a quote expands, so the popup is placed again for its new height.
  const [layout, setLayout] = useState(0);
  const [position, setPosition] = useState<PopupPosition | null>(null);
  const containerRef = useRef<HTMLSpanElement>(null);
  const first = passages[0];
  const name = siteName(first.source_url);

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
  }, [open, layout]);

  function toggleOpen() {
    // Closing unmounts the quotes, so a reopened popup starts collapsed again.
    setOpen((wasOpen) => !wasOpen);
  }

  return (
    <span ref={containerRef} style={{ position: "relative", display: "inline-block" }}>
      <button
        onClick={toggleOpen}
        title={first.title || name}
        aria-label={`Source: ${name}`}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 5,
          height: 20,
          maxWidth: 160,
          padding: "0 8px 0 5px",
          marginLeft: 4,
          border: `1px solid ${open ? "var(--color-accent)" : "var(--color-border)"}`,
          borderRadius: 999,
          background: open ? "var(--color-accent-bg)" : "var(--color-chip)",
          color: open ? "var(--color-accent)" : "var(--color-text-muted)",
          fontSize: 11,
          fontWeight: 500,
          lineHeight: 1,
          cursor: "pointer",
          verticalAlign: "middle",
          transition: "all 0.12s",
        }}
      >
        <SiteIcon url={first.source_url} name={name} />
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {name}
        </span>
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
              marginBottom: 8,
              color: "var(--color-text)",
              overflowWrap: "anywhere",
            }}
          >
            {first.title}
          </div>
          {passages.map((passage) => (
            <Quote
              key={passage.chunk_id}
              passage={passage}
              onResize={() => setLayout((n) => n + 1)}
            />
          ))}
          <a
            href={first.source_url}
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