import { describe, expect, test } from "bun:test";
import {
  agentTool,
  createAgent,
  defineAgent,
  isAgentSpec,
  type AgentLikeEvent,
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

describe("defineAgent", () => {
  test("requires a model at definition time", () => {
    expect(() => defineAgent({ model: "" })).toThrow("AgentOptions.model is required");
  });

  test("spawn() returns independent sessions that do not share history", async () => {
    const callsPerClient: number[][] = [];
    const spec = defineAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          // With a shared client we cannot attribute calls to a session, so
          // this spec is spawned with per-session client overrides below.
          throw new Error("shared client should not be used");
        },
      },
    });

    for (let i = 0; i < 2; i++) {
      const calls: number[] = [];
      callsPerClient.push(calls);
      const session = spec.spawn({
        modelClient: {
          async createMessage({ messages }) {
            calls.push(messages.length);
            return textAssistant(`reply ${i}`);
          },
        },
      });
      await session.prompt(`task ${i}`);
    }

    // A fresh session's first model request contains exactly one message: the
    // new user prompt. If history leaked between spawns, the second session
    // would have started with the first session's turns included.
    expect(callsPerClient[0]).toEqual([1]);
    expect(callsPerClient[1]).toEqual([1]);
  });

  test("a spawned session retains its own history across prompts", async () => {
    const historyLengths: number[] = [];
    const spec = defineAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          historyLengths.push(messages.length);
          return textAssistant("ok");
        },
      },
    });

    const session = spec.spawn();
    await session.prompt("first");
    await session.prompt("second");

    // second prompt = previous user + assistant turns plus the new user message
    expect(historyLengths).toEqual([1, 3]);
  });

  test("isAgentSpec distinguishes specs from live sessions", () => {
    const spec = defineAgent({ apiKey: "test-key", model: "claude-test", modelClient: { async createMessage() { return textAssistant("ok"); } } });
    const session = createAgent({ apiKey: "test-key", model: "claude-test", modelClient: { async createMessage() { return textAssistant("ok"); } } });

    expect(isAgentSpec(spec)).toBe(true);
    expect(isAgentSpec(session)).toBe(false);
  });
});

describe("agentTool with an AgentSpec target", () => {
  test("each tool call spawns a fresh session with no memory of previous calls", async () => {
    const childHistoryLengths: number[] = [];
    const childTasks: string[] = [];
    const spec = defineAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          childHistoryLengths.push(messages.length);
          childTasks.push(String(messages.at(-1)?.content));
          return textAssistant("child done");
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
            return toolUseAssistant("toolu_1", "review", { mode: "ask", task: "Review commit one" });
          }
          if (parentCalls === 2) {
            return toolUseAssistant("toolu_2", "review", { mode: "ask", task: "Review commit two" });
          }
          return textAssistant("parent final");
        },
      },
      tools: [agentTool("review", spec, { description: "Ask the reviewer." })],
    });

    const events = await collect(parent.query("Review two commits, one at a time."));

    expect(childTasks).toEqual([
      expect.stringContaining("Review commit one"),
      expect.stringContaining("Review commit two"),
    ]);
    // Both child sessions started empty: the spec spawned a new Agent per call.
    expect(childHistoryLengths).toEqual([1, 1]);
    expect(events.at(-1)).toMatchObject({ type: "result", result: "parent final" });
  });

  test("a live AgentLike target keeps its history across tool calls", async () => {
    const childHistoryLengths: number[] = [];
    const child = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          childHistoryLengths.push(messages.length);
          return textAssistant("child done");
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
            return toolUseAssistant("toolu_1", "review", { mode: "ask", task: "Review commit one" });
          }
          if (parentCalls === 2) {
            return toolUseAssistant("toolu_2", "review", { mode: "ask", task: "Review commit two" });
          }
          return textAssistant("parent final");
        },
      },
      tools: [agentTool("review", child, { description: "Ask the reviewer." })],
    });

    await collect(parent.query("Review two commits, one at a time."));

    // The second call sees the first task's user+assistant turns: continuity
    // is the explicit consequence of registering a session instead of a spec.
    expect(childHistoryLengths).toEqual([1, 3]);
  });

  test("the tool description declares the target's statefulness", () => {
    const spec = defineAgent({ apiKey: "test-key", model: "claude-test", modelClient: { async createMessage() { return textAssistant("ok"); } } });
    const session = createAgent({ apiKey: "test-key", model: "claude-test", modelClient: { async createMessage() { return textAssistant("ok"); } } });

    const specTool = agentTool("review", spec, { description: "Ask the reviewer." });
    const sessionTool = agentTool("review", session, { description: "Ask the reviewer." });

    expect(specTool.description).toContain("fresh session with no memory of previous calls");
    expect(sessionTool.description).toContain("long-lived session that retains conversation history");
  });
});
