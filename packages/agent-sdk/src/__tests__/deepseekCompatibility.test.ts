import { describe, expect, test } from "bun:test";
import { isDeepSeekBaseURL, needsNonThinkingToolContinuation } from "../deepseekCompatibility.js";
import type { ContentBlock, ModelMessage } from "../index.js";

const call: ContentBlock = { type: "tool_use", id: "call_1", name: "submit_output", input: { answer: 42 } };
const result: ModelMessage = { role: "user", content: [{ type: "tool_result", tool_use_id: "call_1", content: "Structured output recorded" }] };
const thinking: ContentBlock = { type: "thinking", thinking: "fixture", signature: "signature" };
const responsesThinking: ContentBlock = { type: "openai_reasoning", item: { type: "reasoning", id: "rs_1", summary: [],
  content: [{ type: "reasoning_text", text: "fixture" }] } };
const conversation = (blocks: ContentBlock[]): ModelMessage[] => [{ role: "user", content: "Task" }, { role: "assistant", content: blocks }, result];

describe("DeepSeek same-turn thinking compatibility", () => {
  test("a non-thinking tool turn must consume its result without enabling thinking", () => {
    expect(needsNonThinkingToolContinuation(conversation([call]))).toBe(true);
  });
  test.each([thinking, responsesThinking])("a genuine reasoning-bearing tool turn may continue thinking (%s)", block => {
    expect(needsNonThinkingToolContinuation(conversation([block, call]))).toBe(false);
  });
  test("reasoning on an earlier assistant does not authorize thinking on the latest non-thinking call", () => {
    const messages = conversation([call]);
    messages.splice(1, 0, { role: "assistant", content: [thinking] }, { role: "user", content: "Continue" });
    expect(needsNonThinkingToolContinuation(messages)).toBe(true);
  });
  test.each(["Next task", [{ type: "text", text: "Next task" }] as ContentBlock[]])("a new user prompt restores normal thinking selection", content => {
    expect(needsNonThinkingToolContinuation([...conversation([call]), { role: "user", content }])).toBe(false);
  });
  test("empty histories and assistant text endings do not need a tool continuation", () => {
    expect(needsNonThinkingToolContinuation([])).toBe(false);
    expect(needsNonThinkingToolContinuation(conversation([{ type: "text", text: "done" }]))).toBe(false);
  });
  test("DeepSeek matching uses the exact hostname", () => {
    expect(isDeepSeekBaseURL("https://api.deepseek.com/v1")).toBe(true);
    expect(isDeepSeekBaseURL("https://api.deepseek.com/anthropic")).toBe(true);
    expect(isDeepSeekBaseURL("https://api.deepseek.com.proxy.example")).toBe(false);
    expect(isDeepSeekBaseURL("https://api.openai.com")).toBe(false);
    expect(isDeepSeekBaseURL("invalid")).toBe(false);
  });
});
