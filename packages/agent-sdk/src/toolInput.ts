import type { ToolUseBlock } from "./index.js";

// Internal per-response parse state; never serialized into provider messages.
export const toolInputFailures = new WeakMap<ToolUseBlock, { kind: "json" | "object"; position?: number; detail?: string }>();

// Distill a JSON.parse failure into a stable description. Never echoes argument text:
// V8's "Unexpected token" style embeds a source snippet, so only the token itself is kept.
export function describeJsonParseError(error: unknown): { position?: number; detail?: string } {
  if (!(error instanceof Error)) return {};
  const message = error.message;
  const atPosition = /^(.+?) (in|after) JSON at position (\d+)(?: \(line \d+ column \d+\))?$/.exec(message);
  if (atPosition) return { detail: `${atPosition[1]} ${atPosition[2]} JSON`, position: Number(atPosition[3]) };
  if (message === "Unexpected end of JSON input") return { detail: message };
  const token = /^(Unexpected token (?:'[^']*'|.))[, ]/.exec(message);
  if (token) {
    const position = /position (\d+)/.exec(message);
    return { detail: token[1], ...(position ? { position: Number(position[1]) } : {}) };
  }
  if (message.startsWith("JSON Parse error:")) return { detail: message.slice("JSON Parse error:".length).trim() };
  return {};
}
