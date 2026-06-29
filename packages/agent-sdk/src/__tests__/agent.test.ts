import { describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod/v4";
import {
  AbortError,
  MaxTurnsError,
  ToolExecutionError,
  createAgent,
  createJsonlContextTracer,
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
      tools: ["Read", "Write", "Edit", "LS", "Glob", "Grep", "Bash"],
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

  test("writes agent context trace entries to jsonl", async () => {
    const dir = await mkdtemp(join(tmpdir(), "agent-sdk-trace-"));
    const tracePath = join(dir, "trace.jsonl");
    try {
      const agent = createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: clientFromResponses([
          toolUseAssistant("toolu_1", "calculator", { expr: "2+2" }),
          textAssistant("The answer is 4"),
        ]),
        tools: [
          tool("calculator", "Calculate", z.object({ expr: z.string() }), async () => ({
            content: "4",
          })),
        ],
        tracer: createJsonlContextTracer({ path: tracePath }),
      });

      await collect(agent.query("What is 2+2?", { stream: false }));

      const entries = (await readFile(tracePath, "utf8"))
        .trim()
        .split("\n")
        .map(line => JSON.parse(line));
      expect(entries.map(entry => entry.type)).toContain("run_start");
      expect(entries.map(entry => entry.type)).toContain("user_message");
      expect(entries.map(entry => entry.type)).toContain("model_request");
      expect(entries.map(entry => entry.type)).toContain("assistant_message");
      expect(entries.map(entry => entry.type)).toContain("tool_use");
      expect(entries.map(entry => entry.type)).toContain("tool_result");
      expect(entries.at(-1)).toMatchObject({
        version: 1,
        type: "result",
        data: {
          subtype: "success",
          result: "The answer is 4",
        },
        source: {
          kind: "agent",
        },
      });
      expect(entries.every(entry => entry.session_id === entries[0].session_id)).toBe(true);
      expect(entries.map(entry => entry.seq)).toEqual(entries.map((_, index) => index + 1));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
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

    expect(seenSystemPrompts[0]).toContain("You are the backend executor. Only handle API and database work.");
    expect(seenSystemPrompts[0]).toContain("Your private workspace is:");
    expect(seenSystemPrompts[0]).toContain(join(homedir(), ".agent", "workspaces", "agent-"));
  });

  test("workspace option gives the agent private workspace tools and instructions", async () => {
    const dir = await mkdtemp(join(tmpdir(), "agent-sdk-workspace-"));
    const seenRequests: Array<{
      systemPrompt?: string;
      toolNames: string[];
    }> = [];
    let calls = 0;
    const modelClient: ModelClient = {
      async createMessage({ systemPrompt, tools }) {
        calls++;
        seenRequests.push({
          systemPrompt,
          toolNames: tools.map(tool => tool.name),
        });
        if (calls === 1) {
          return toolUseAssistant("toolu_1", "Write", {
            file_path: "notes/delivery.txt",
            content: "workspace delivery",
          });
        }
        return textAssistant("Wrote notes/delivery.txt and verified it in my workspace.");
      },
    };

    try {
      const agent = createAgent({
        apiKey: "test-key",
        model: "claude-test",
        systemPrompt: "You are the backend executor.",
        workspace: dir,
        modelClient,
      });

      const messages = await collect(agent.query("Create a delivery artifact."));
      const artifact = await readFile(join(dir, "notes", "delivery.txt"), "utf8");

      expect(messages.at(-1)).toMatchObject({
        type: "result",
        subtype: "success",
        result: "Wrote notes/delivery.txt and verified it in my workspace.",
      });
      expect(artifact).toBe("workspace delivery");
      expect(seenRequests[0]?.toolNames).toEqual([
        "Read",
        "Write",
        "Edit",
        "LS",
        "Glob",
        "Grep",
        "Bash",
      ]);
      expect(seenRequests[0]?.systemPrompt).toContain("You are the backend executor.");
      expect(seenRequests[0]?.systemPrompt).toContain("Your private workspace is:");
      expect(seenRequests[0]?.systemPrompt).toContain(dir);
      expect(seenRequests[0]?.systemPrompt).toContain("Reply in natural language with the important workspace paths");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("creates the configured workspace directory when the agent is created", async () => {
    const dir = await mkdtemp(join(tmpdir(), "agent-sdk-workspace-root-"));
    const workspace = join(dir, "missing", "agent-workspace");
    try {
      createAgent({
        apiKey: "test-key",
        model: "claude-test",
        workspace,
        modelClient: clientFromResponses([textAssistant("handled")]),
      });

      expect((await stat(workspace)).isDirectory()).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("defaults the agent workspace to ~/.agent/workspaces/name when omitted", async () => {
    const seenRequests: Array<{
      systemPrompt?: string;
      toolNames: string[];
    }> = [];
    const modelClient: ModelClient = {
      async createMessage({ systemPrompt, tools }) {
        seenRequests.push({
          systemPrompt,
          toolNames: tools.map(tool => tool.name),
        });
        return textAssistant("handled");
      },
    };
    const agent = createAgent({
      apiKey: "test-key",
      name: "Backend Agent",
      model: "claude-test",
      modelClient,
    });

    await collect(agent.query("Use the default workspace."));

    const expectedWorkspace = join(homedir(), ".agent", "workspaces", "Backend_Agent");
    expect(seenRequests[0]?.toolNames).toEqual([
      "Read",
      "Write",
      "Edit",
      "LS",
      "Glob",
      "Grep",
      "Bash",
    ]);
    expect(seenRequests[0]?.systemPrompt).toContain("Your private workspace is:");
    expect(seenRequests[0]?.systemPrompt).toContain(expectedWorkspace);
  });

  test("uses 16384 as the default maxTokens for model requests and traces", async () => {
    const seenMaxTokens: number[] = [];
    const traceEvents: Array<{ type: string; data?: Record<string, unknown> }> = [];
    const modelClient: ModelClient = {
      async createMessage({ maxTokens }) {
        seenMaxTokens.push(maxTokens);
        return textAssistant("handled");
      },
    };
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient,
      tracer: {
        onEvent(event) {
          traceEvents.push(event);
        },
      },
    });

    await collect(agent.query("Use the default output budget", { stream: false }));

    expect(seenMaxTokens).toEqual([16384]);
    expect(traceEvents.find(event => event.type === "model_request")).toMatchObject({
      data: {
        max_tokens: 16384,
      },
    });
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

  test("defaults maxTurns to 50 model requests", async () => {
    let calls = 0;
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          calls++;
          return toolUseAssistant(`toolu_${calls}`, "calculator", { expr: "2+2" });
        },
      },
      tools: [
        tool("calculator", "Calculate", z.object({ expr: z.string() }), async () => ({
          content: "4",
        })),
      ],
    });

    const messages = await collect(agent.query("Keep calculating."));

    expect(calls).toBe(50);
    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "error_max_turns",
      is_error: true,
      num_turns: 50,
    });
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

  test("Anthropic client serializes image and document user blocks unchanged", async () => {
    let requestBody: any;
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        requestBody = await request.json();
        return Response.json({
          id: "msg_test",
          type: "message",
          role: "assistant",
          model: "claude-test",
          content: [{ type: "text", text: "received" }],
          stop_reason: "end_turn",
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
        });
      },
    });

    try {
      const agent = createAgent({
        apiKey: "test-key",
        baseURL: `http://127.0.0.1:${server.port}`,
        model: "claude-test",
      });

      await collect(agent.query([
        { type: "text", text: "Compare these files." },
        {
          type: "image",
          source: {
            type: "base64",
            media_type: "image/png",
            data: "iVBORw0KGgo=",
          },
        },
        {
          type: "document",
          source: {
            type: "base64",
            media_type: "application/pdf",
            data: "JVBERi0xLjQK",
          },
        },
      ], { stream: false }));

      expect(requestBody.messages[0].content).toEqual([
        { type: "text", text: "Compare these files." },
        {
          type: "image",
          source: {
            type: "base64",
            media_type: "image/png",
            data: "iVBORw0KGgo=",
          },
        },
        {
          type: "document",
          source: {
            type: "base64",
            media_type: "application/pdf",
            data: "JVBERi0xLjQK",
          },
        },
      ]);
    } finally {
      server.stop(true);
    }
  });
});
