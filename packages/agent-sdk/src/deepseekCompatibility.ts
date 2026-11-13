import type { ModelMessage } from "./index.js";

export function isDeepSeekBaseURL(baseURL: string): boolean {
  try { return new URL(baseURL).hostname === "api.deepseek.com"; }
  catch { return false; }
}

/**
 * DeepSeek cannot enable thinking halfway through a non-thinking tool turn.
 * A tool call answered in thinking mode is a thinking turn even when the
 * provider returned an empty thinking block: DeepSeek accepts that block back
 * with thinking enabled, and keeping thinking on keeps its prefix cache valid.
 */
export function needsNonThinkingToolContinuation(messages: ModelMessage[]): boolean {
  const last = messages.at(-1);
  if (last?.role !== "user" || !Array.isArray(last.content) || last.content.length === 0 ||
      !last.content.every(block => block.type === "tool_result")) return false;
  let assistant: ModelMessage | undefined;
  for (let index = messages.length - 2; index >= 0; index--) {
    if (messages[index]?.role === "assistant") { assistant = messages[index]; break; }
  }
  if (!assistant || !Array.isArray(assistant.content) ||
      !assistant.content.some(block => block.type === "tool_use")) return false;
  return !assistant.content.some(block =>
    block.type === "thinking" ||
    (block.type === "openai_reasoning" && block.item.content?.some(part => part.text.length > 0)),
  );
}
