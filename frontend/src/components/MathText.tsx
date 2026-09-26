import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { parseMath } from "../utils/math";

interface MathTextProps {
  text: string;
}

function renderMath(value: string, display: boolean): string | null {
  try {
    return katex.renderToString(value, {
      displayMode: display,
      throwOnError: false,
      output: "html",
    });
  } catch {
    return null;
  }
}

/** Renders model text by converting LaTeX-delimited segments into formulas. */
export function MathText({ text }: MathTextProps) {
  const segments = useMemo(() => parseMath(text), [text]);

  return (
    <>
      {segments.map((segment, i) => {
        if (segment.type === "text") return <span key={i}>{segment.value}</span>;

        const html = renderMath(segment.value, segment.display);
        // If KaTeX cannot render a formula, show the original LaTeX instead of dropping it.
        if (html === null) {
          return (
            <code key={i} style={{ fontSize: "0.95em" }}>
              {segment.value}
            </code>
          );
        }

        return <span key={i} dangerouslySetInnerHTML={{ __html: html }} />;
      })}
    </>
  );
}
