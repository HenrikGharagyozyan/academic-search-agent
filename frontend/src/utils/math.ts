export type Segment =
  | { type: "text"; value: string }
  | { type: "math"; value: string; display: boolean };

interface Delimiter {
  open: string;
  close: string;
  display: boolean;
}

// Order matters: $$ is checked before $, otherwise a block formula would split into two inline ones.
const DELIMITERS: Delimiter[] = [
  { open: "$$", close: "$$", display: true },
  { open: "\\[", close: "\\]", display: true },
  { open: "$", close: "$", display: false },
  { open: "\\(", close: "\\)", display: false },
];

// Rejects "$5 and $10": in a currency pair, the content starts or ends with whitespace.
// For $...$ formulas, the content also cannot span multiple lines.
function isMath(content: string, delimiter: Delimiter): boolean {
  if (content.trim() === "") return false;
  if (delimiter.open !== "$") return true;
  if (/^\s|\s$/.test(content)) return false;
  return !content.includes("\n");
}

/**
 * Splits text into plain segments and formulas marked with $...$, $$...$$, \(...\), or \[...\].
 * An unmatched or non-formula-like delimiter remains plain text.
 */
export function parseMath(input: string): Segment[] {
  const segments: Segment[] = [];
  let text = "";
  let i = 0;

  while (i < input.length) {
    const delimiter = DELIMITERS.find((d) => input.startsWith(d.open, i));
    if (!delimiter) {
      text += input[i];
      i += 1;
      continue;
    }

    const contentStart = i + delimiter.open.length;
    const closeIndex = input.indexOf(delimiter.close, contentStart);
    if (closeIndex === -1 || !isMath(input.slice(contentStart, closeIndex), delimiter)) {
      text += input[i];
      i += 1;
      continue;
    }

    if (text) {
      segments.push({ type: "text", value: text });
      text = "";
    }
    segments.push({
      type: "math",
      value: input.slice(contentStart, closeIndex).trim(),
      display: delimiter.display,
    });
    i = closeIndex + delimiter.close.length;
  }

  if (text) segments.push({ type: "text", value: text });
  return segments;
}
