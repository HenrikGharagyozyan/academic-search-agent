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

export interface Answer {
  question: string;
  summary: string;
  claims: Claim[];
  conclusion: string;
  evidence: Record<string, AnswerEvidence>;
  evidence_sufficient: boolean;
}