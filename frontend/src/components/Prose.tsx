import { Children, type ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MathText } from "./MathText";

interface ProseProps {
  text: string;
}

/**
 * Renders a claim, summary or conclusion: Markdown for structure, KaTeX for
 * maths, and nothing else.
 *
 * The model is asked for headings, bullet lists, fenced text diagrams and
 * tables when the evidence has that shape, and without a Markdown renderer all
 * of that reached the reader as literal `###` and backticks.
 *
 * Maths stays with MathText rather than moving to remark-math, so that
 * parseMath's rules survive — in particular that "$5 and $10" is currency and
 * not a formula. Every string child is passed through it; element children are
 * left alone, and code blocks are deliberately excluded so a diagram's
 * characters are never read as maths.
 */
function withMath(children: ReactNode): ReactNode {
  return Children.map(children, (child) =>
    typeof child === "string" ? <MathText text={child} /> : child,
  );
}

const MUTED = "var(--color-text-muted)";
const BORDER = "1px solid var(--color-border)";

export function Prose({ text }: ProseProps) {
  return (
    <Markdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => (
          <p style={{ margin: "0 0 10px", lineHeight: 1.7 }}>{withMath(children)}</p>
        ),
        // The claim's own theme is already a section heading, so a heading
        // inside one is a sub-point: rendered smaller than its container, never
        // larger.
        h1: ({ children }) => <SubHeading>{withMath(children)}</SubHeading>,
        h2: ({ children }) => <SubHeading>{withMath(children)}</SubHeading>,
        h3: ({ children }) => <SubHeading>{withMath(children)}</SubHeading>,
        h4: ({ children }) => <SubHeading>{withMath(children)}</SubHeading>,
        ul: ({ children }) => (
          <ul style={{ margin: "0 0 10px", paddingLeft: 22, lineHeight: 1.7 }}>
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol style={{ margin: "0 0 10px", paddingLeft: 22, lineHeight: 1.7 }}>
            {children}
          </ol>
        ),
        li: ({ children }) => <li style={{ marginBottom: 3 }}>{withMath(children)}</li>,
        strong: ({ children }) => <strong>{withMath(children)}</strong>,
        em: ({ children }) => <em>{withMath(children)}</em>,
        // A fenced block is a diagram: monospaced, scrollable, and never
        // touched by the maths parser.
        pre: ({ children }) => (
          <pre
            style={{
              margin: "0 0 12px",
              padding: "12px 14px",
              background: "var(--color-accent-bg)",
              border: BORDER,
              borderRadius: 8,
              fontSize: 12.5,
              lineHeight: 1.5,
              overflowX: "auto",
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            }}
          >
            {children}
          </pre>
        ),
        code: ({ children, className }) =>
          // Fenced blocks carry a language class; a bare `code` is inline.
          className ? (
            <code>{children}</code>
          ) : (
            <code
              style={{
                background: "var(--color-accent-bg)",
                padding: "1px 5px",
                borderRadius: 4,
                fontSize: "0.92em",
              }}
            >
              {children}
            </code>
          ),
        table: ({ children }) => (
          <div style={{ overflowX: "auto", margin: "0 0 12px" }}>
            <table style={{ borderCollapse: "collapse", fontSize: 13.5, width: "100%" }}>
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th
            style={{
              textAlign: "left",
              padding: "6px 10px",
              borderBottom: "2px solid var(--color-border)",
              color: MUTED,
              fontWeight: 600,
            }}
          >
            {withMath(children)}
          </th>
        ),
        td: ({ children }) => (
          <td style={{ padding: "6px 10px", borderBottom: BORDER, verticalAlign: "top" }}>
            {withMath(children)}
          </td>
        ),
        blockquote: ({ children }) => (
          <blockquote
            style={{
              margin: "0 0 10px",
              paddingLeft: 12,
              borderLeft: "3px solid var(--color-border)",
              color: MUTED,
            }}
          >
            {children}
          </blockquote>
        ),
        hr: () => <hr style={{ border: "none", borderTop: BORDER, margin: "14px 0" }} />,
        a: ({ children, href }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--color-accent)" }}
          >
            {children}
          </a>
        ),
      }}
    >
      {text}
    </Markdown>
  );
}

function SubHeading({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: 14, fontWeight: 700, margin: "12px 0 6px" }}>{children}</div>
  );
}
