import type { Answer } from "../types/answer";

export function buildCitationNumbers(answer: Answer): Map<string, number> {
  const numbers = new Map<string, number>();
  let counter = 1;

  for (const claim of answer.claims) {
    for (const id of claim.evidence_ids) {
      if (!numbers.has(id) && answer.evidence[id]) {
        numbers.set(id, counter);
        counter++;
      }
    }
  }

  return numbers;
}