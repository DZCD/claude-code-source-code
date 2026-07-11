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
  createBareAgent,
  createBuiltinTools,
  createCompositeContextTracer,
  createJsonlContextTracer,
  createLangSmithContextTracer,
  tool,
  type ContextTraceEvent,
  type ModelClient,
  type ModelRequest,
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

type FakeRunConfig = {
  name: string;
  run_type?: string;
  id?: string;
  project_name?: string;
  parent_run?: FakeRunTree;
  start_time?: number | string;
  metadata?: Record<string, unknown>;
  tags?: string[];
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  error?: string;
  client?: unknown;
  replicas?: Array<Record<string, unknown>>;
};

type FakeLangSmithClient = {
  flush(): Promise<void>;
  awaitPendingTraceBatches(): Promise<void>;
};

class FakeRunTree {
  static runs: FakeRunTree[] = [];
  static operations: Array<{ type: string; run: string; options?: unknown }> = [];

  id?: string;
  name: string;
  run_type: string;
  project_name?: string;
  parent_run?: FakeRunTree;
  child_runs: FakeRunTree[] = [];
  start_time?: number | string;
  metadata: Record<string, unknown>;
  tags?: string[];
  inputs: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  error?: string;
  events: unknown[] = [];
  client?: FakeLangSmithClient;
  replicas?: Array<Record<string, unknown>>;

  constructor(config: FakeRunConfig) {
    this.id = config.id;
    this.name = config.name;
    this.run_type = config.run_type ?? "chain";
    this.project_name = config.project_name;
    this.parent_run = config.parent_run;
    this.start_time = config.start_time;
    this.metadata = config.metadata ?? {};
    this.tags = config.tags;
    this.inputs = config.inputs ?? {};
    this.outputs = config.outputs;
    this.error = config.error;
    this.client = config.client as FakeLangSmithClient | undefined;
    this.replicas = config.replicas;
    FakeRunTree.runs.push(this);
  }

  createChild(config: FakeRunConfig): FakeRunTree {
    const child = new FakeRunTree({ ...config, parent_run: this });
    this.child_runs.push(child);
    return child;
  }

  async postRun(excludeChildRuns?: boolean): Promise<void> {
    FakeRunTree.operations.push({
      type: "post",
      run: this.name,
      options: { excludeChildRuns },
    });
  }

  async end(
    outputs?: Record<string, unknown>,
    error?: string,
    _endTime?: number,
    metadata?: Record<string, unknown>,
  ): Promise<void> {
    this.outputs = outputs;
    this.error = error;
    if (metadata) {
      this.metadata = { ...this.metadata, ...metadata };
    }
    FakeRunTree.operations.push({ type: "end", run: this.name });
  }

  async patchRun(options?: { excludeInputs?: boolean }): Promise<void> {
    FakeRunTree.operations.push({
      type: "patch",
      run: this.name,
      options,
    });
  }

  addEvent(event: unknown): void {
    this.events.push(event);
  }

  static reset(): void {
    FakeRunTree.runs = [];
    FakeRunTree.operations = [];
  }
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

  test("passes structured output format to the model request", async () => {
    const requests: ModelRequest[] = [];
    const modelClient: ModelClient = {
      async createMessage(request) {
        requests.push(request);
        return textAssistant('{"answer":4}');
      },
    };
    const outputFormat = {
      type: "json_schema" as const,
      schema: {
        type: "object",
        properties: {
          answer: { type: "number" },
        },
        required: ["answer"],
        additionalProperties: false,
      },
    };

    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient,
    });

    const result = await agent.prompt("What is 2+2?", { outputFormat });

    expect(result.result).toBe('{"answer":4}');
    expect(requests[0]?.outputFormat).toEqual(outputFormat);
  });

  test("passes agent thinking config to the model request and allows query overrides", async () => {
    const requests: ModelRequest[] = [];
    const modelClient: ModelClient = {
      async createMessage(request) {
        requests.push(request);
        return textAssistant("done");
      },
    };
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      thinkingConfig: { type: "enabled", budgetTokens: 8_000 },
      modelClient,
    });

    await agent.prompt("Use the default.");
    await agent.prompt("Override it.", { thinkingConfig: { type: "disabled" } });

    expect(requests[0]?.thinkingConfig).toEqual({ type: "enabled", budgetTokens: 8_000 });
    expect(requests[1]?.thinkingConfig).toEqual({ type: "disabled" });
  });

  test("does not reject non-json text when outputFormat is ignored by the provider", async () => {
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: clientFromResponses([
        textAssistant('答案如下：\n\n```json\n{"answer":4}\n```'),
      ]),
    });

    const result = await agent.prompt("What is 2+2?", { outputFormat: "json" });

    expect(result).toMatchObject({
      type: "result",
      subtype: "success",
      is_error: false,
      result: '答案如下：\n\n```json\n{"answer":4}\n```',
    });
  });

  test("createBareAgent does not add workspace prompt, tools, or directory", async () => {
    const name = `bare-agent-${Date.now()}`;
    const defaultWorkspace = join(homedir(), ".agent", "workspaces", name);
    await rm(defaultWorkspace, { recursive: true, force: true });
    const requests: ModelRequest[] = [];
    const modelClient: ModelClient = {
      async createMessage(request) {
        requests.push(request);
        return textAssistant("bare");
      },
    };

    const agent = createBareAgent({
      apiKey: "test-key",
      name,
      model: "claude-test",
      modelClient,
    });

    const messages = await collect(agent.query("Say hello", { stream: false }));

    expect(messages[0]).toMatchObject({
      type: "system",
      subtype: "init",
      model: "claude-test",
      tools: [],
    });
    expect(requests[0]?.systemPrompt).toBeUndefined();
    expect(requests[0]?.tools).toEqual([]);
    await expect(stat(defaultWorkspace)).rejects.toThrow();
  });

  test("createBareAgent lets callers opt into system prompts and builtin tools", async () => {
    const cwd = await mkdtemp(join(tmpdir(), "agent-sdk-bare-"));
    const requests: ModelRequest[] = [];
    const modelClient: ModelClient = {
      async createMessage(request) {
        requests.push(request);
        return textAssistant("customized");
      },
    };

    try {
      const agent = createBareAgent({
        apiKey: "test-key",
        model: "claude-test",
        systemPrompt: "You are a deliberately configured agent.",
        tools: createBuiltinTools({ cwd }),
        modelClient,
      });

      const messages = await collect(agent.query("Say hello", { stream: false }));

      expect(messages[0]).toMatchObject({
        type: "system",
        tools: ["Read", "Write", "Edit", "LS", "Glob", "Grep", "Bash"],
      });
      expect(requests[0]?.systemPrompt).toBe("You are a deliberately configured agent.");
      expect(requests[0]?.tools.map(tool => tool.name)).toEqual(["Read", "Write", "Edit", "LS", "Glob", "Grep", "Bash"]);
    } finally {
      await rm(cwd, { recursive: true, force: true });
    }
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

  test("passes host business context to custom tools without AgentRuntime shape", async () => {
    type QcContext = {
      patientRecordId: string;
      scoringStandardId: string;
    };
    const seenMessages: unknown[] = [];
    const modelClient: ModelClient = {
      async createMessage({ messages }) {
        seenMessages.push(messages);
        if (seenMessages.length === 1) {
          return toolUseAssistant("toolu_1", "read_patient_record", {});
        }
        return textAssistant("I found the current patient record.");
      },
    };
    const readPatientRecordInputSchema = z.object({});
    const qcTool = tool<QcContext>();
    const agent = createBareAgent<QcContext>({
      apiKey: "test-key",
      model: "claude-test",
      modelClient,
      tools: [
        qcTool(
          "read_patient_record",
          "Read the current patient record",
          readPatientRecordInputSchema,
          async (_input, { context }) => ({
            content: JSON.stringify({
              patientRecordId: context?.patientRecordId,
              scoringStandardId: context?.scoringStandardId,
            }),
          }),
        ),
      ],
    });

    const messages = await collect(agent.query("Review the current patient record", {
      context: {
        patientRecordId: "ocr_123",
        scoringStandardId: "tumor-v1",
      },
      stream: false,
    }));

    expect(messages[2]).toMatchObject({
      type: "user",
      tool_use_result: JSON.stringify({
        patientRecordId: "ocr_123",
        scoringStandardId: "tumor-v1",
      }),
    });
    expect(messages.at(-1)).toMatchObject({
      type: "result",
      result: "I found the current patient record.",
    });
  });

  test("returns structured permission_denied tool results for workspace tools", async () => {
    const cwd = await mkdtemp(join(tmpdir(), "agent-sdk-private-"));
    const outsideDir = await mkdtemp(join(tmpdir(), "agent-sdk-outside-"));
    const outsideFile = join(outsideDir, "server.js");
    let calls = 0;

    try {
      const modelClient: ModelClient = {
        async createMessage({ messages }) {
          calls++;
          if (calls === 1) {
            return toolUseAssistant("toolu_1", "Write", {
              file_path: outsideFile,
              content: "escape",
            });
          }
          const content = messages.at(-1)?.content;
          const toolResult = Array.isArray(content)
            ? content.find(block => block.type === "tool_result")
            : undefined;
          const result = String(toolResult?.content ?? "");
          expect(result).toContain('"status": "permission_denied"');
          expect(result).toContain('"tool": "Write"');
          expect(result).toContain('"requestedPath"');
          expect(result).toContain('"allowedWriteRoots"');
          expect(result).toContain('"suggestedNextStep"');
          return textAssistant("I will ask for a grant or use an allowed root.");
        },
      };
      const agent = createAgent({
        apiKey: "test-key",
        model: "claude-test",
        workspace: cwd,
        modelClient,
      });

      const messages = await collect(agent.query("Write outside the workspace.", { stream: false }));

      expect(messages.at(-1)).toMatchObject({
        type: "result",
        subtype: "success",
        result: "I will ask for a grant or use an allowed root.",
      });
    } finally {
      await rm(cwd, { recursive: true, force: true });
      await rm(outsideDir, { recursive: true, force: true });
    }
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

  test("fans out context trace events to composite tracers", async () => {
    const seen: string[] = [];
    const event: ContextTraceEvent = {
      version: 1,
      timestamp: "2026-06-29T00:00:00.000Z",
      session_id: "session_1",
      run_id: "11111111-1111-4111-8111-111111111111",
      seq: 1,
      source: { kind: "agent", name: "agent" },
      type: "run_start",
      data: { model: "claude-test", tools: [] },
    };
    const tracer = createCompositeContextTracer([
      {
        onEvent(input) {
          seen.push(`first:${input.type}`);
        },
        async flush() {
          seen.push("first:flush");
        },
        async close() {
          seen.push("first:close");
        },
      },
      {
        onEvent(input) {
          seen.push(`second:${input.type}`);
        },
        async flush() {
          seen.push("second:flush");
        },
        async close() {
          seen.push("second:close");
        },
      },
    ]);

    await tracer.onEvent(event);
    await tracer.flush?.();
    await tracer.close?.();

    expect(seen).toEqual([
      "first:run_start",
      "second:run_start",
      "first:flush",
      "second:flush",
      "first:close",
      "second:close",
    ]);
  });

  test("maps agent context trace events to LangSmith chain and llm runs", async () => {
    FakeRunTree.reset();
    const tracer = createLangSmithContextTracer({
      RunTree: FakeRunTree,
      projectName: "agent-sdk-tests",
      tags: ["test-suite"],
      metadata: { environment: "test" },
    });
    const agent = createAgent({
      apiKey: "test-key",
      name: "researcher",
      model: "claude-test",
      systemPrompt: "You are the research lead. Keep answers concise.",
      modelClient: clientFromResponses([textAssistant("hello")]),
    });

    await collect(agent.query("Say hello", { stream: false, tracer }));
    await tracer.flush?.();

    const root = FakeRunTree.runs.find(run => run.run_type === "chain");
    const llm = FakeRunTree.runs.find(run => run.run_type === "llm");
    expect(root).toMatchObject({
      name: "researcher",
      run_type: "chain",
      project_name: "agent-sdk-tests",
      inputs: {
        message: {
          role: "user",
          content: "Say hello",
        },
      },
      outputs: {
        subtype: "success",
        result: "hello",
      },
      metadata: {
        environment: "test",
        sdk_session_id: expect.any(String),
        sdk_run_id: expect.any(String),
        sdk_source_kind: "agent",
        sdk_source_name: "researcher",
        model: "claude-test",
      },
    });
    expect(root?.tags).toEqual(expect.arrayContaining([
      "agent-lattice",
      "test-suite",
      "source:agent",
    ]));
    expect(llm).toMatchObject({
      run_type: "llm",
      parent_run: root,
      inputs: {
        model: "claude-test",
        systemPrompt: expect.stringContaining("You are the research lead. Keep answers concise."),
        messages: [
          {
            role: "system",
            content: expect.stringContaining("You are the research lead. Keep answers concise."),
          },
          {
            role: "user",
            content: "Say hello",
          },
        ],
        stream: false,
      },
      outputs: {
        message: {
          role: "assistant",
          content: [{ type: "text", text: "hello" }],
        },
      },
      metadata: {
        sdk_event_type: "model_request",
        sdk_parent_run_id: undefined,
      },
    });
    expect(FakeRunTree.operations).toEqual(expect.arrayContaining([
      { type: "post", run: "researcher", options: { excludeChildRuns: true } },
      { type: "patch", run: "researcher", options: { excludeInputs: false } },
    ]));
  });

  test("closes LangSmith tracer by draining RunTree clients", async () => {
    FakeRunTree.reset();
    const clientOperations: string[] = [];
    const client: FakeLangSmithClient = {
      async flush() {
        clientOperations.push("client:flush");
      },
      async awaitPendingTraceBatches() {
        clientOperations.push("client:awaitPendingTraceBatches");
      },
    };
    const replicaClient: FakeLangSmithClient = {
      async flush() {
        clientOperations.push("replica:flush");
      },
      async awaitPendingTraceBatches() {
        clientOperations.push("replica:awaitPendingTraceBatches");
      },
    };
    const tracer = createLangSmithContextTracer({
      RunTree: FakeRunTree,
      projectName: "agent-sdk-tests",
      client: client as never,
      apiKey: "test-langsmith-key",
      apiUrl: "https://api.smith.langchain.com",
      workspaceId: "workspace-123",
      runTree: config => new FakeRunTree({
        ...config,
        client,
        replicas: (config.replicas ?? []).map(replica => Array.isArray(replica)
          ? replica
          : { ...replica, client: replicaClient }),
      }) as never,
    });
    const agent = createAgent({
      apiKey: "test-key",
      name: "close-drain-check",
      model: "claude-test",
      modelClient: clientFromResponses([textAssistant("done")]),
    });

    await collect(agent.query("Trace then close", { stream: false, tracer }));
    clientOperations.length = 0;
    await tracer.close?.();

    expect(clientOperations).toEqual([
      "client:flush",
      "client:awaitPendingTraceBatches",
      "replica:flush",
      "replica:awaitPendingTraceBatches",
    ]);
  });

  test("maps SDK tool use events to LangSmith tool runs without agent loop changes", async () => {
    FakeRunTree.reset();
    const tracer = createLangSmithContextTracer({
      RunTree: FakeRunTree,
      projectName: "agent-sdk-tests",
    });
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
    });

    await collect(agent.query("What is 2+2?", { stream: false, tracer }));
    await tracer.flush?.();

    const root = FakeRunTree.runs.find(run => run.run_type === "chain");
    const toolRun = FakeRunTree.runs.find(run => run.run_type === "tool");
    const llmRuns = FakeRunTree.runs.filter(run => run.run_type === "llm");
    expect(llmRuns).toHaveLength(2);
    expect(toolRun).toMatchObject({
      name: "calculator",
      run_type: "tool",
      parent_run: root,
      inputs: {
        input: { expr: "2+2" },
      },
      outputs: {
        content: "4",
        is_error: false,
      },
      metadata: {
        tool_use_id: "toolu_1",
        tool_name: "calculator",
      },
    });
    expect(root?.outputs).toMatchObject({
      subtype: "success",
      result: "The answer is 4",
    });
  });

  test("passes LangSmith workspace connection options to RunTree replicas", async () => {
    FakeRunTree.reset();
    const tracer = createLangSmithContextTracer({
      RunTree: FakeRunTree,
      projectName: "agent-sdk-tests",
      apiUrl: "https://api.smith.langchain.com",
      apiKey: "test-langsmith-key",
      workspaceId: "workspace-123",
    });
    const agent = createAgent({
      apiKey: "test-key",
      name: "workspace-check",
      model: "claude-test",
      modelClient: clientFromResponses([textAssistant("ok")]),
    });

    await collect(agent.query("Trace with workspace", { stream: false, tracer }));

    const root = FakeRunTree.runs.find(run => run.run_type === "chain");
    expect(root?.replicas).toEqual([
      {
        projectName: "agent-sdk-tests",
        apiUrl: "https://api.smith.langchain.com",
        apiKey: "test-langsmith-key",
        workspaceId: "workspace-123",
      },
    ]);
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

  test("Anthropic client serializes json output format as output_config", async () => {
    let requestBody: any;
    let betaHeader: string | null = null;
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        requestBody = await request.json();
        betaHeader = request.headers.get("anthropic-beta");
        return Response.json({
          id: "msg_test",
          type: "message",
          role: "assistant",
          model: "claude-test",
          content: [{ type: "text", text: "{\"ok\":true}" }],
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

      await agent.prompt("Return JSON.", { outputFormat: "json", stream: false });

      expect(requestBody.output_config).toEqual({
        format: {
          type: "json_schema",
          schema: {},
        },
      });
      expect(betaHeader).toContain("structured-outputs-2025-12-15");
    } finally {
      server.stop(true);
    }
  });

  test("Anthropic client serializes thinking config and preserves thinking blocks", async () => {
    let requestBody: any;
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        requestBody = await request.json();
        return Response.json({
          id: "msg_thinking",
          type: "message",
          role: "assistant",
          model: "claude-test",
          content: [
            { type: "thinking", thinking: "I should answer briefly.", signature: "sig_test" },
            { type: "text", text: "Done." },
          ],
          stop_reason: "end_turn",
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 2 },
        });
      },
    });

    try {
      const agent = createAgent({
        apiKey: "test-key",
        baseURL: `http://127.0.0.1:${server.port}`,
        model: "claude-test",
        maxTokens: 4_000,
        thinkingConfig: { type: "enabled", budgetTokens: 8_000 },
      });

      const messages = await collect(agent.query("Think.", { stream: false }));

      expect(requestBody.thinking).toEqual({ type: "enabled", budget_tokens: 3_999 });
      expect(messages[1]).toMatchObject({
        type: "assistant",
        message: {
          content: [
            { type: "thinking", thinking: "I should answer briefly.", signature: "sig_test" },
            { type: "text", text: "Done." },
          ],
        },
      });
    } finally {
      server.stop(true);
    }
  });
});
