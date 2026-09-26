export interface Claim {
  text: string;
  evidence_ids: string[];
  confidence: "high" | "medium" | "low";
}

export interface AnswerEvidence {
  chunk_id: string;
  document_id: string;
  text: string;
  source_url: string;
  title: string;
  start_line: number;
  end_line: number;
}

export type ActivityKind =
  | "search"
  | "source_found"
  | "scrape_ok"
  | "scrape_failed"
  | "collect"
  | "select"
  | "grade_relevance"
  | "generate"
  | "verify"
  | "grade_answer"
  | "refine";

export interface ActivityStep {
  kind: ActivityKind;
  /** Written by the backend, shown as-is. */
  label: string;
  url?: string | null;
  title?: string | null;
  detail?: string | null;
  /** Which pass of the refine loop this step belongs to. */
  attempt: number;
}

export interface Answer {
  question: string;
  summary: string;
  claims: Claim[];
  conclusion: string;
  evidence: Record<string, AnswerEvidence>;
  evidence_sufficient: boolean;
  activity: ActivityStep[];
}