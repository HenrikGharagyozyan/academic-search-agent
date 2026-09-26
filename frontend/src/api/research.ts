import type { Answer } from "../types/answer";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

export interface ProgressEvent {
  stage: string;
  label: string;
}

export type StreamEvent =
  | { type: "progress"; data: ProgressEvent }
  | { type: "result"; data: Answer }
  | { type: "error"; data: { detail: string } };

export async function* streamQuestion(question: string): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE_URL}/answer/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      const lines = rawEvent.split("\n");
      const eventLine = lines.find((l) => l.startsWith("event: "));
      const dataLine = lines.find((l) => l.startsWith("data: "));
      if (!dataLine) continue;

      const type = (eventLine?.slice("event: ".length) ?? "message") as StreamEvent["type"];
      const data = JSON.parse(dataLine.slice("data: ".length));

      yield { type, data } as StreamEvent;
    }
  }
}