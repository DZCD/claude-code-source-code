import { describe, expect, test } from "bun:test";
import {
  agentTool,
  createAgent,
  type AgentLikeEvent,
  type ModelClient,
} from "../index.js";

function textAssistant(text: string) {
  return {
    role: "assistant" as const,
    content: [{ type: "text" as const, text }],
  };
}

function toolUseAssistant(id: string, name: string, input: Record<string, unknown>) {
  return {
    role: "assistant" as const,
    content: [{ type: "tool_use" as const, id, name, input }],
  };
}

async function collect(iterable: AsyncIterable<AgentLikeEvent>): Promise<AgentLikeEvent[]> {
  const messages: AgentLikeEvent[] = [];
  for await (const message of iterable) {
    messages.push(message);
  }
  return messages;
}

describe("agentTool", () => {
  test("ask mode calls the target AgentLike and returns its final result as tool_result", async () => {
    const child = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          expect(String(messages.at(-1)?.content)).toContain("Implement the API");
          return textAssistant("child delivered the API");
        },
      },
    });
    let parentCalls = 0;
    const parentClient: ModelClient = {
      async createMessage({ messages, tools }) {
        parentCalls++;
        if (parentCalls === 1) {
          expect(tools.map(tool => tool.name)).toContain("engineering");
          return toolUseAssistant("toolu_1", "engineering", {
            mode: "ask",
            task: "Implement the API",
            expectedOutput: "A concise delivery report",
            acceptanceCriteria: ["Return the main endpoint list"],
          });
        }

        expect(messages.at(-1)).toMatchObject({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_1",
              content: "child delivered the API",
            },
          ],
        });
        return textAssistant("parent final");
      },
    };
    const parent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: parentClient,
      tools: [
        agentTool("engineering", child, {
          description: "Ask engineering for delivery work.",
        }),
      ],
    });

    const events = await collect(parent.query("Ask engineering to implement the API."));

    expect(events.some(event =>
      event.type === "user" &&
      event.tool_use_result === "child delivered the API"
    )).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: "result", result: "parent final" });
  });

  test("handoff mode without a runtime returns a transparent tool error", async () => {
    const child = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          return textAssistant("child should not run");
        },
      },
    });
    let parentCalls = 0;
    const parent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          parentCalls++;
          if (parentCalls === 1) {
            return toolUseAssistant("toolu_1", "engineering", {
              mode: "handoff",
              task: "Implement a long-running feature",
            });
          }
          return textAssistant("parent saw the error");
        },
      },
      tools: [
        agentTool("engineering", child, {
          description: "Hand work to engineering.",
        }),
      ],
    });

    const events = await collect(parent.query("Hand off this task."));
    const toolResult = events.find(event => event.type === "user");

    expect(toolResult).toMatchObject({
      type: "user",
      error: expect.any(Error),
    });
    expect(toolResult?.type === "user" ? toolResult.tool_use_result : "").toContain("mode=handoff requires an AgentRuntime");
    expect(events.at(-1)).toMatchObject({ type: "result", result: "parent saw the error" });
  });

  test("unsupported observe mode reports the unsupported action instead of silently downgrading", async () => {
    const child = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          return textAssistant("child should not run");
        },
      },
    });
    let parentCalls = 0;
    const parent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          parentCalls++;
          if (parentCalls === 1) {
            return toolUseAssistant("toolu_1", "engineering", {
              mode: "observe",
              task: "Observe a long-running implementation",
            });
          }
          return textAssistant("parent saw unsupported mode");
        },
      },
      tools: [
        agentTool("engineering", child, {
          description: "Work with engineering.",
        }),
      ],
    });

    const events = await collect(parent.query("Observe engineering work."));
    const toolResult = events.find(event => event.type === "user");

    expect(toolResult?.type === "user" ? toolResult.tool_use_result : "").toContain("mode=observe is not supported");
    expect(toolResult?.type === "user" ? toolResult.tool_use_result : "").toContain("Available modes: ask, handoff");
    expect(events.at(-1)).toMatchObject({ type: "result", result: "parent saw unsupported mode" });
  });

  test("isConcurrencySafe opts delegated calls into parallel execution under the default safe mode", async () => {
    let active = 0;
    let maxActive = 0;
    const makeChild = (label: string) => createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          active++;
          maxActive = Math.max(maxActive, active);
          await new Promise(resolve => setTimeout(resolve, 30));
          active--;
          return textAssistant(`${label} done`);
        },
      },
    });
    let parentCalls = 0;
    const parent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          parentCalls++;
          if (parentCalls === 1) {
            return {
              role: "assistant" as const,
              content: [
                { type: "tool_use" as const, id: "toolu_1", name: "reviewer_a", input: { mode: "ask", task: "Review A" } },
                { type: "tool_use" as const, id: "toolu_2", name: "reviewer_b", input: { mode: "ask", task: "Review B" } },
              ],
            };
          }
          return textAssistant("parent final");
        },
      },
      tools: [
        agentTool("reviewer_a", makeChild("A"), {
          description: "Delegate review A.",
          isConcurrencySafe: () => true,
        }),
        agentTool("reviewer_b", makeChild("B"), {
          description: "Delegate review B.",
          isConcurrencySafe: () => true,
        }),
      ],
    });

    const events = await collect(parent.query("Run both reviews."));
    expect(events.at(-1)).toMatchObject({ type: "result", result: "parent final" });
    expect(maxActive).toBe(2);
  });

  test("delegated calls without isConcurrencySafe stay sequential under the default safe mode", async () => {
    let active = 0;
    let maxActive = 0;
    const makeChild = (label: string) => createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          active++;
          maxActive = Math.max(maxActive, active);
          await new Promise(resolve => setTimeout(resolve, 30));
          active--;
          return textAssistant(`${label} done`);
        },
      },
    });
    let parentCalls = 0;
    const parent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          parentCalls++;
          if (parentCalls === 1) {
            return {
              role: "assistant" as const,
              content: [
                { type: "tool_use" as const, id: "toolu_1", name: "reviewer_a", input: { mode: "ask", task: "Review A" } },
                { type: "tool_use" as const, id: "toolu_2", name: "reviewer_b", input: { mode: "ask", task: "Review B" } },
              ],
            };
          }
          return textAssistant("parent final");
        },
      },
      tools: [
        agentTool("reviewer_a", makeChild("A"), { description: "Delegate review A." }),
        agentTool("reviewer_b", makeChild("B"), { description: "Delegate review B." }),
      ],
    });

    const events = await collect(parent.query("Run both reviews."));
    expect(events.at(-1)).toMatchObject({ type: "result", result: "parent final" });
    expect(maxActive).toBe(1);
  });
});
