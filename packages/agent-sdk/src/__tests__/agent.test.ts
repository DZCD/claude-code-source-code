import { describe, expect, test } from "bun:test";
import { z } from "zod/v4";
import {
  AbortError,
  MaxTurnsError,
  ToolExecutionError,
  createAgent,
  tool,
  type ModelClient,
  type SDKMessage,
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

function clientFromResponses(responses: Array<ReturnType<typeof textAssistant> | ReturnType<typeof toolUseAssistant>>): ModelClient {
  let index = 0;
  return {
    async createMessage() {
      const response = responses[index++];
      if (!response) throw new Error("No mock response available");
      return response;
    },
  };
}

function streamingClientFromResponses(responses: Array<ReturnType<typeof textAssistant> | ReturnType<typeof toolUseAssistant>>): ModelClient {
  let index = 0;
  return {
    async createMessage({ onStreamEvent }) {
      const response = responses[index++];
      if (!response) throw new Error("No mock response available");
      onStreamEvent?.({ type: "message_start" });
      for (const block of response.content) {
        if (block.type === "text") {
          onStreamEvent?.({ type: "content_block_delta", delta: { type: "text_delta", text: block.text } });
        } else {
          onStreamEvent?.({ type: "content_block_start", content_block: block });
        }
      }
      onStreamEvent?.({ type: "message_stop" });
      return response;
    },
  };
}

async function collect(iterable: AsyncIterable<SDKMessage>): Promise<SDKMessage[]> {
  const messages: SDKMessage[] = [];
  for await (const message of iterable) {
    messages.push(message);
  }
  return messages;
}

describe("agent-sdk", () => {
  test("streams init, assistant, and success result for a plain answer", async () => {
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: clientFromResponses([textAssistant("hello")]),
    });

    const messages = await collect(agent.query("Say hello"));

    expect(messages.map(message => message.type)).toEqual([
      "system",
      "assistant",
      "result",
    ]);
    expect(messages[0]).toMatchObject({
      type: "system",
      subtype: "init",
      model: "claude-test",
      tools: [],
    });
    expect(messages[1]).toMatchObject({
      type: "assistant",
      message: { content: [{ type: "text", text: "hello" }] },
    });
    expect(messages[2]).toMatchObject({
      type: "result",
      subtype: "success",
      result: "hello",
      is_error: false,
    });
  });

  test("executes a custom tool and feeds the result back to the model", async () => {
    const seenMessages: unknown[] = [];
    const modelClient: ModelClient = {
      async createMessage({ messages }) {
        seenMessages.push(messages);
        if (seenMessages.length === 1) {
          return toolUseAssistant("toolu_1", "calculator", { expr: "2+2" });
        }
        return textAssistant("The answer is 4");
      },
    };
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient,
      tools: [
        tool(
          "calculator",
          "Evaluate a simple expression",
          z.object({ expr: z.string() }),
          async input => ({ content: input.expr === "2+2" ? "4" : "unknown" }),
        ),
      ],
    });

    const messages = await collect(agent.query("What is 2+2?"));

    expect(messages.map(message => message.type)).toEqual([
      "system",
      "assistant",
      "user",
      "assistant",
      "result",
    ]);
    expect(messages[2]).toMatchObject({
      type: "user",
      message: {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "toolu_1",
            content: "4",
          },
        ],
      },
      tool_use_result: "4",
    });
    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "success",
      result: "The answer is 4",
    });
  });

  test("passes systemPrompt to the model client", async () => {
    const seenSystemPrompts: Array<string | undefined> = [];
    const modelClient: ModelClient = {
      async createMessage({ systemPrompt }) {
        seenSystemPrompts.push(systemPrompt);
        return textAssistant("handled");
      },
    };
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      systemPrompt: "You are the backend executor. Only handle API and database work.",
      modelClient,
    });

    await collect(agent.query("Build the API"));

    expect(seenSystemPrompts).toEqual([
      "You are the backend executor. Only handle API and database work.",
    ]);
  });

  test("emits stream events for streaming text responses", async () => {
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: streamingClientFromResponses([textAssistant("hello")]),
    });

    const messages = await collect(agent.query("Say hello", { stream: true }));

    expect(messages.map(message => message.type)).toEqual([
      "system",
      "stream_event",
      "stream_event",
      "stream_event",
      "assistant",
      "result",
    ]);
    expect(messages[2]).toMatchObject({
      type: "stream_event",
      event: { type: "content_block_delta", delta: { text: "hello" } },
    });
  });

  test("emits stream events for streaming tool calls", async () => {
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: streamingClientFromResponses([
        toolUseAssistant("toolu_1", "calculator", { expr: "2+2" }),
        textAssistant("4"),
      ]),
      tools: [
        tool("calculator", "Calculate", z.object({ expr: z.string() }), async () => ({
          content: "4",
        })),
      ],
    });

    const messages = await collect(agent.query("What is 2+2?", { stream: true }));

    expect(messages.some(message => message.type === "stream_event" && message.event.type === "content_block_start")).toBe(true);
    expect(messages.some(message => message.type === "user")).toBe(true);
    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "success",
      result: "4",
    });
  });

  test("can disable model streaming while still returning tool results", async () => {
    const calls: boolean[] = [];
    const modelClient: ModelClient = {
      async createMessage({ stream }) {
        calls.push(stream);
        return calls.length === 1
          ? toolUseAssistant("toolu_1", "calculator", { expr: "2+2" })
          : textAssistant("4");
      },
    };
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient,
      tools: [
        tool("calculator", "Calculate", z.object({ expr: z.string() }), async () => ({
          content: "4",
        })),
      ],
    });

    const messages = await collect(agent.query("What is 2+2?", { stream: false }));

    expect(calls).toEqual([false, false]);
    expect(messages.some(message => message.type === "stream_event")).toBe(false);
    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "success",
      result: "4",
    });
  });

  test("turns permission denial into a tool result instead of throwing", async () => {
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: clientFromResponses([
        toolUseAssistant("toolu_1", "danger", { value: "rm -rf" }),
        textAssistant("I cannot run that."),
      ]),
      tools: [
        tool("danger", "Dangerous action", z.object({ value: z.string() }), async () => ({
          content: "should not run",
        })),
      ],
      permission: async () => ({ behavior: "deny", message: "blocked by policy" }),
    });

    const messages = await collect(agent.query("Run danger"));

    expect(messages[2]).toMatchObject({
      type: "user",
      message: {
        content: [
          {
            type: "tool_result",
            tool_use_id: "toolu_1",
            content: "blocked by policy",
            is_error: true,
          },
        ],
      },
    });
    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "success",
      result: "I cannot run that.",
    });
  });

  test("turns tool handler failures into error tool results", async () => {
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: clientFromResponses([
        toolUseAssistant("toolu_1", "explode", {}),
        textAssistant("The tool failed."),
      ]),
      tools: [
        tool("explode", "Throws", z.object({}), async () => {
          throw new Error("boom");
        }),
      ],
    });

    const messages = await collect(agent.query("Use explode"));

    expect(messages[2]).toMatchObject({
      type: "user",
      message: {
        content: [
          {
            type: "tool_result",
            tool_use_id: "toolu_1",
            content: "Tool explode failed: boom",
            is_error: true,
          },
        ],
      },
    });
    expect(messages[2]).toHaveProperty("error");
    expect((messages[2] as { error: unknown }).error).toBeInstanceOf(ToolExecutionError);
  });

  test("stops with an error result when maxTurns is reached", async () => {
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      maxTurns: 1,
      modelClient: clientFromResponses([
        toolUseAssistant("toolu_1", "calculator", { expr: "2+2" }),
      ]),
      tools: [
        tool("calculator", "Calculate", z.object({ expr: z.string() }), async () => ({
          content: "4",
        })),
      ],
    });

    const messages = await collect(agent.query("What is 2+2?"));

    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "error_max_turns",
      is_error: true,
    });
    expect((messages.at(-1) as { error: unknown }).error).toBeInstanceOf(MaxTurnsError);
  });

  test("returns an abort result when the signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: clientFromResponses([textAssistant("unreachable")]),
    });

    const messages = await collect(agent.query("Hi", { signal: controller.signal }));

    expect(messages).toHaveLength(2);
    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "error_abort",
      is_error: true,
    });
    expect((messages.at(-1) as { error: unknown }).error).toBeInstanceOf(AbortError);
  });

  test("prompt returns the final success result", async () => {
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: clientFromResponses([textAssistant("done")]),
    });

    const result = await agent.prompt("Finish");

    expect(result).toMatchObject({
      type: "result",
      subtype: "success",
      result: "done",
    });
  });
});
