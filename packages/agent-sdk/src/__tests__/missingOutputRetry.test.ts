import { describe, expect, test } from "bun:test";
import { z } from "zod/v4";
import { createBareAgent, defineContextTracer, defineHistoryStore, tool,
  type AssistantModelMessage, type ModelRequest, type ContextTraceEvent, type SDKMessage } from "../index.js";

const schema = z.object({ result: z.string() });
const text = (stopReason = "end_turn"): AssistantModelMessage => ({
  role: "assistant", stopReason,
  content: [{ type: "thinking", thinking: "analysis", signature: "saved-signature" }, { type: "text", text: "answer without submission" }],
});
const submit = (input: unknown = { result: "done" }): AssistantModelMessage => ({
  role: "assistant", stopReason: "tool_use",
  content: [{ type: "tool_use", id: "submit-1", name: "submit_output", input: input as Record<string, unknown> }],
});
function createScript(responses: AssistantModelMessage[]) {
  const requests: ModelRequest[] = [];
  return { requests, client: { async createMessage(request: ModelRequest) {
    requests.push(request);
    const response = responses[requests.length - 1];
    if (!response) throw new Error("Unexpected extra model request");
    return response;
  } } };
}
async function collect(agent: ReturnType<typeof createBareAgent>, signal?: AbortSignal) {
  const events: SDKMessage[] = [];
  for await (const event of agent.query("work", { stream: false, signal })) events.push(event);
  return { events, result: events.findLast(event => event.type === "result")! };
}

describe("retryMissingOutput", () => {
  test("default - missing submission - fails without another request", async () => {
    const script = createScript([text()]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, modelClient: script.client });
    expect((await agent.prompt("work")).subtype).toBe("error_missing_output");
    expect(script.requests).toHaveLength(1);
  });

  test("recovery - thinking history and completed tool - submits without replay and persists the reminder", async () => {
    let reads = 0;
    const trace: ContextTraceEvent[] = [];
    const stored: unknown[] = [];
    const script = createScript([
      { role: "assistant", stopReason: "tool_use", content: [{ type: "tool_use", id: "read-1", name: "read", input: {} }] },
      text(), submit(),
    ]);
    const agent = createBareAgent({
      model: "test", outputSchema: schema, retryMissingOutput: true, maxTurns: 3,
      thinkingConfig: { type: "enabled", budgetTokens: 1024 }, reasoningEffort: "high",
      tools: [tool("read", "read", z.object({}), () => { reads++; return { content: "full evidence" }; })],
      historyStore: defineHistoryStore({ load: () => [], append: m => { stored.push(structuredClone(m)); }, replace: () => {} }),
      tracer: defineContextTracer({ onEvent: e => { trace.push(e); } }), modelClient: script.client,
    });
    const { result, events } = await collect(agent);
    expect(result).toMatchObject({ subtype: "success", num_turns: 3, structuredResult: { result: "done" } });
    expect(reads).toBe(1);
    expect(script.requests[1]).toMatchObject({ thinkingConfig: { type: "enabled" }, reasoningEffort: "high" });
    const recovery = script.requests[2]!;
    expect(recovery.thinkingConfig).toEqual({ type: "disabled" });
    expect(recovery.reasoningEffort).toBeUndefined();
    expect(recovery.toolChoice).toEqual({ type: "tool", name: "submit_output" });
    expect(recovery.tools.map(t => t.name)).toEqual(["submit_output"]);
    expect(recovery.messages).toContainEqual(text());
    expect(JSON.stringify(recovery.messages)).toContain("full evidence");
    expect(JSON.stringify(stored)).toContain("No structured output has been submitted");
    expect(events.filter(e => e.type === "result")).toHaveLength(1);
    expect(events.filter(e => e.type === "user")).toHaveLength(2);
    expect(trace.filter(e => e.type === "model_request").at(-1)?.data).toMatchObject({ missingOutputRetry: true, toolChoice: recovery.toolChoice });
    expect(trace.filter(e => e.type === "result")).toHaveLength(1);
  });

  test.each(["end_turn", "max_tokens", "refusal", "model_context_window_exceeded"])("recovery - second response %s - stops after one additional request", async reason => {
    const script = createScript([text(), text(reason)]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: true, modelClient: script.client });
    expect((await agent.prompt("work")).subtype).toBe("error_missing_output");
    expect(script.requests).toHaveLength(2);
  });

  test.each(["max_tokens", "refusal", "model_context_window_exceeded", "stop_sequence"])("initial response - %s - does not trigger recovery", async reason => {
    const script = createScript([text(reason)]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: true, modelClient: script.client });
    expect((await agent.prompt("work")).subtype).toBe("error_missing_output");
    expect(script.requests).toHaveLength(1);
  });

  test("recovery - invalid submission - records tool error without a third request", async () => {
    const script = createScript([text(), submit({ result: 42 })]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: true, modelClient: script.client });
    const { result, events } = await collect(agent);
    expect(result.subtype).toBe("error_missing_output");
    expect(script.requests).toHaveLength(2);
    expect(events.some(e => e.type === "user" && e.error !== undefined)).toBe(true);
  });

  test.each(["other", "mixed", "duplicate", "truncated"])("recovery - %s tool batch - executes no tools and preserves paired history", async scenario => {
    let writes = 0;
    const other = { type: "tool_use" as const, id: "write-1", name: "write", input: {} };
    const response = submit();
    if (scenario === "other") response.content = [other];
    if (scenario === "mixed") response.content.push(other);
    if (scenario === "duplicate") response.content.push({ type: "tool_use", id: "submit-2", name: "submit_output", input: { result: "duplicate" } });
    if (scenario === "truncated") response.stopReason = "max_tokens";
    const script = createScript([text(), response]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: true, modelClient: script.client,
      tools: [tool("write", "side effect", z.object({}), () => { writes++; return { content: "written" }; })],
    });
    expect((await agent.prompt("work")).subtype).toBe("error_missing_output");
    expect(writes).toBe(0);
    expect(script.requests).toHaveLength(2);
    const last = (await agent.getHistory()).at(-1)!;
    expect(last.role).toBe("user");
    expect(Array.isArray(last.content) && last.content.every(b => b.type === "tool_result" && b.is_error)).toBe(true);
  });

  test("maxTurns - no remaining turn - never exceeds the original budget", async () => {
    const script = createScript([text()]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: true, maxTurns: 1, modelClient: script.client });
    expect((await agent.prompt("work")).subtype).toBe("error_max_turns");
    expect(script.requests).toHaveLength(1);
  });

  test("existing submission - summary - does not trigger recovery", async () => {
    const script = createScript([submit(), text()]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, submitOutputEndTurn: false,
      retryMissingOutput: true, modelClient: script.client });
    expect((await agent.prompt("work")).structuredResult).toEqual({ result: "done" });
    expect(script.requests.every(r => r.toolChoice === undefined)).toBe(true);
  });

  test("query overrides - enabled thinking and effort - recovery still disables both", async () => {
    const script = createScript([text(), submit()]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: true, modelClient: script.client });
    expect((await agent.prompt("work", { thinkingConfig: { type: "enabled", budgetTokens: 1024 }, reasoningEffort: "max" })).subtype).toBe("success");
    expect(script.requests[1]?.thinkingConfig).toEqual({ type: "disabled" });
    expect(script.requests[1]?.reasoningEffort).toBeUndefined();
  });

  test("recovery - large context - skips compaction and retains the one-request limit", async () => {
    const response = { ...text(), usage: { input_tokens: 150000, output_tokens: 10 } };
    const script = createScript([response, submit()]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: true,
      autoCompact: { thresholdTokens: 1, keepRecentMessages: 1 }, modelClient: script.client,
      historyStore: defineHistoryStore({ load: () => [
        { role: "user", content: "previous task" }, { role: "assistant", content: "previous answer" },
      ], append: () => {}, replace: () => {} }),
    });
    expect((await agent.prompt("work")).subtype).toBe("success");
    expect(script.requests).toHaveLength(2);
    expect(script.requests[1]?.messages).toContainEqual(response);
  });

  test("custom tool - disabled option - preserves ordinary text success", async () => {
    const script = createScript([text()]);
    const agent = createBareAgent({ model: "test", retryMissingOutput: false, modelClient: script.client,
      tools: [tool("submit_output", "submit", schema, () => ({ content: "ok" }))] });
    expect((await agent.prompt("work")).subtype).toBe("success");
    expect(script.requests).toHaveLength(1);
  });

  test("custom tool - ends without delivery - enabled option rejects handler success", async () => {
    const script = createScript([submit()]);
    const agent = createBareAgent({ model: "test", retryMissingOutput: true, modelClient: script.client,
      tools: [tool("submit_output", "submit", schema, () => ({ content: "ok", endTurn: true }))] });
    expect((await agent.prompt("work")).subtype).toBe("error_missing_output");
    expect(script.requests).toHaveLength(1);
  });

  test("recovery - interrupted after assistant event - never executes the submission", async () => {
    let submissions = 0;
    const script = createScript([text(), submit()]);
    const agent = createBareAgent({ model: "test", retryMissingOutput: true, modelClient: script.client,
      tools: [tool("submit_output", "submit", schema, () => { submissions++; return { content: "ok", structuredResult: "done", endTurn: true }; })] });
    const events: SDKMessage[] = [];
    for await (const event of agent.query("work")) {
      events.push(event);
      if (event.type === "assistant" && event.message.content.some(b => b.type === "tool_use")) agent.interrupt();
    }
    expect(submissions).toBe(0);
    expect(events.findLast(e => e.type === "result")?.subtype).toBe("interrupted");
  });

  test.each([true, false])("custom submit_output - structured delivery=%p - validates delivery, not handler success", async deliver => {
    const script = createScript([text(), submit()]);
    const agent = createBareAgent({ model: "test", retryMissingOutput: true, modelClient: script.client,
      tools: [tool("submit_output", "submit", schema, input => ({ content: "ok", ...(deliver ? { structuredResult: { business: input.result } } : {}), endTurn: true }))],
    });
    const result = await agent.prompt("work");
    expect(result.subtype).toBe(deliver ? "success" : "error_missing_output");
    if (deliver) expect(result.structuredResult).toEqual({ business: "done" });
  });

  test("summary after recovery - submitOutputEndTurn false - restores normal thinking", async () => {
    const script = createScript([text(), submit(), text()]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, submitOutputEndTurn: false,
      retryMissingOutput: true, thinkingConfig: { type: "enabled", budgetTokens: 1024 }, modelClient: script.client,
    });
    expect((await agent.prompt("work")).structuredResult).toEqual({ result: "done" });
    expect(script.requests[2]?.toolChoice).toBeUndefined();
    expect(script.requests[2]?.thinkingConfig?.type).toBe("enabled");
  });

  test("new query - previous attempt used recovery - each query has its own allowance", async () => {
    const script = createScript([text(), submit(), text(), submit()]);
    const agent = createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: true, modelClient: script.client });
    expect((await agent.prompt("first")).subtype).toBe("success");
    expect((await agent.prompt("second")).subtype).toBe("success");
    expect(script.requests.filter(r => r.toolChoice)).toHaveLength(2);
  });

  test("configuration - missing tool or invalid flag - rejects at construction", () => {
    expect(() => createBareAgent({ model: "test", retryMissingOutput: true })).toThrow("requires outputSchema");
    expect(() => createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: 1 as any })).toThrow("must be a boolean");
  });

  test.each(["abort", "interrupt", "timeout"])("recovery - %s during request - preserves cancellation subtype", async mode => {
    const controller = new AbortController();
    let calls = 0;
    let agent: ReturnType<typeof createBareAgent>;
    agent = createBareAgent({ model: "test", outputSchema: schema, retryMissingOutput: true, requestTimeoutMs: 30,
      modelClient: { async createMessage() {
        if (++calls === 1) return text();
        if (mode === "abort") controller.abort();
        if (mode === "interrupt") agent.interrupt();
        return new Promise<AssistantModelMessage>(() => {});
      } },
    });
    const { result, events } = await collect(agent, controller.signal);
    expect(result.subtype).toBe(mode === "abort" ? "error_abort" : mode === "interrupt" ? "interrupted" : "error_timeout");
    expect(calls).toBe(2);
    expect(events.filter(e => e.type === "result")).toHaveLength(1);
  });
});

describe("missing-output recovery wire protocol", () => {
  test.each([false, true])("Anthropic transport - stream=%p - serializes thinking disabled and forced tool choice", async stream => {
    const bodies: any[] = [];
    const server = Bun.serve({ port: 0, async fetch(request) {
      const body = await request.json() as any;
      bodies.push(body);
      const recovery = bodies.length === 2;
      const content = recovery ? submit().content : text().content;
      const reason = recovery ? "tool_use" : "end_turn";
      const message = { id: "fake-response", type: "message", role: "assistant", model: "test", content, stop_reason: reason, stop_sequence: null, usage: { input_tokens: 100, output_tokens: 10 } };
      if (!stream) return Response.json(message);
      const events: any[] = [{ type: "message_start", message: { ...message, content: [], stop_reason: null } }];
      content.forEach((block, index) => {
        events.push({ type: "content_block_start", index, content_block: block }, { type: "content_block_stop", index });
      });
      events.push({ type: "message_delta", delta: { stop_reason: reason }, usage: { output_tokens: 10 } }, { type: "message_stop" });
      return new Response(events.map(e => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join(""), { headers: { "content-type": "text/event-stream" } });
    } });
    try {
      const agent = createBareAgent({ model: "test", apiKey: "fake", baseURL: server.url.origin,
        outputSchema: schema, retryMissingOutput: true, thinkingConfig: { type: "enabled", budgetTokens: 1024 }, reasoningEffort: "high",
      });
      expect((await agent.prompt("work", { stream })).structuredResult).toEqual({ result: "done" });
      expect(bodies).toHaveLength(2);
      expect(bodies[0].thinking.type).toBe("enabled");
      expect(bodies[1].thinking).toEqual({ type: "disabled" });
      expect(bodies[1].tool_choice).toEqual({ type: "tool", name: "submit_output" });
      expect(bodies[1].reasoning_effort).toBeUndefined();
      expect(JSON.stringify(bodies[1].messages)).toContain("saved-signature");
    } finally { server.stop(true); }
  });
});
