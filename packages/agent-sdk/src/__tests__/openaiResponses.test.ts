import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod/v4";
import {
  APIError, createBareAgent, createJsonlHistoryStore, createOpenAIResponsesClient, tool,
  type ModelRequest, type SDKMessage,
} from "../index.js";

const text = (value = "done", phase?: "commentary" | "final_answer") => ({
  type: "message", id: "msg_1", role: "assistant", status: "completed", phase,
  content: [{ type: "output_text", text: value, annotations: [] }],
});
const reasoning = {
  type: "reasoning", id: "rs_1", summary: [{ type: "summary_text", text: "summary" }],
  encrypted_content: "encrypted-reasoning", status: "completed",
};
const call = (args = '{"city":"Paris"}', name = "weather", id = "call_1") => ({
  type: "function_call", id: "fc_1", call_id: id, name, arguments: args, status: "completed",
});
function response(output: unknown[] = [text()], extra = {}) {
  return {
    id: "resp_1", object: "response", status: "completed", model: "test-model",
    output, error: null, incomplete_details: null,
    usage: { input_tokens: 100, output_tokens: 20, input_tokens_details: { cached_tokens: 30 } },
    ...extra,
  };
}
function request(extra: Partial<ModelRequest> = {}): ModelRequest {
  return { model: "test-model", maxTokens: 200, messages: [{ role: "user", content: "hi" }], tools: [], stream: false, ...extra };
}
const servers: ReturnType<typeof Bun.serve>[] = [];
afterEach(() => { for (const server of servers.splice(0)) server.stop(true); });
function endpoint(handler: (req: Request, body: any) => Response | Promise<Response>) {
  const bodies: any[] = [];
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(req) {
    const body = await req.json(); bodies.push(body);
    return handler(req, body);
  } });
  servers.push(server);
  const client = createOpenAIResponsesClient({ apiKey: "test-key", baseURL: `${server.url}v1` });
  return { client, bodies, baseURL: `${server.url}v1` };
}
function sse(events: unknown[]) {
  const bytes = new TextEncoder().encode(events.map(event => `data: ${JSON.stringify(event)}\r\n\r\n`).join(""));
  // Split even inside JSON/UTF-8 boundaries to exercise the official SSE parser.
  return new Response(new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7));
    controller.close();
  } }), { headers: { "Content-Type": "text/event-stream" } });
}

describe("createOpenAIResponsesClient", () => {
  test("sends Responses HTTP protocol and maps usage without counting cached input twice", async () => {
    const { client, bodies } = endpoint(req => {
      expect(new URL(req.url).pathname).toBe("/v1/responses");
      expect(req.headers.get("authorization")).toBe("Bearer test-key");
      return Response.json(response());
    });
    const result = await client.createMessage(request({ systemPrompt: "system", tools: [{
      name: "weather", description: "weather", input_schema: { type: "object", properties: { city: { type: "string" } } },
    }], toolChoice: { type: "tool", name: "weather" } }));
    expect(bodies[0]).toMatchObject({
      model: "test-model", instructions: "system", max_output_tokens: 200, store: false,
      include: ["reasoning.encrypted_content"], input: [{ role: "user", content: "hi" }],
      tools: [{ type: "function", name: "weather", strict: false }],
      tool_choice: { type: "function", name: "weather" }, stream: false,
    });
    expect(bodies[0].previous_response_id).toBeUndefined();
    expect(result).toMatchObject({ content: [{ type: "text", text: "done" }], stopReason: "end_turn",
      providerResponseId: "resp_1", model: "test-model", usage: { input_tokens: 70, output_tokens: 20, cache_read_input_tokens: 30 } });
  });

  test("replays tool calls, encrypted reasoning and assistant phase after JSONL history restoration", async () => {
    const dir = await mkdtemp(join(tmpdir(), "responses-history-"));
    try {
      const { client, bodies, baseURL } = endpoint(() => Response.json(response(
        bodies.length === 1 ? [reasoning, text("checking", "commentary"), call()]
          : [text("sunny", "final_answer")],
      )));
      let executions = 0;
      const options = { model: "test-model", modelClient: client, retryMissingOutput: false,
        historyStore: createJsonlHistoryStore({ path: join(dir, "history.jsonl") }),
        tools: [tool("weather", "weather", z.object({ city: z.string() }), async ({ city }) => {
          executions++; return { content: `sunny in ${city}` };
        })],
      };
      const first = createBareAgent(options);
      expect((await first.prompt("weather?", { stream: false })).result).toBe("sunny");
      expect(executions).toBe(1);
      expect(bodies[1].input).toContainEqual(reasoning);
      expect(bodies[1].input).toContainEqual({ role: "assistant", content: "checking", phase: "commentary" });
      expect(bodies[1].input).toContainEqual({ type: "function_call", call_id: "call_1", name: "weather", arguments: '{"city":"Paris"}' });
      expect(bodies[1].input).toContainEqual({ type: "function_call_output", call_id: "call_1", output: "sunny in Paris" });
      // A new client and Agent own no response-id cache; role/content alone suffice.
      const restored = createBareAgent({ ...options, modelClient: createOpenAIResponsesClient({ apiKey: "test-key", baseURL }) });
      await restored.prompt("again", { stream: false });
      expect(bodies[2].input).toContainEqual(reasoning);
      expect(bodies[2].input).toContainEqual({ role: "assistant", content: "sunny", phase: "final_answer" });
    } finally { await rm(dir, { recursive: true, force: true }); }
  });

  test("streams native events and takes complete tool arguments and reasoning from terminal output", async () => {
    const final = response([reasoning, call(), text("你好")]);
    const events = [
      { type: "response.created", response: response([], { status: "in_progress" }) },
      { type: "response.function_call_arguments.delta", delta: '{"city":' },
      { type: "response.output_text.delta", delta: "你好" },
      { type: "response.completed", response: final },
    ];
    const { client, bodies } = endpoint(() => sse(events));
    const seen: unknown[] = [];
    const result = await client.createMessage(request({ stream: true, onStreamEvent: event => seen.push(event) }));
    expect(bodies[0].stream).toBe(true);
    expect(seen).toEqual(events);
    expect(result.content).toMatchObject([{ type: "openai_reasoning", item: reasoning },
      { type: "tool_use", id: "call_1", input: { city: "Paris" } }, { type: "text", text: "你好" }]);
  });

  test("Agent yields native stream events before the model response completes", async () => {
    let finish!: () => void;
    const gate = new Promise<void>(resolve => { finish = resolve; });
    const { client } = endpoint(() => new Response(new ReadableStream({ async start(controller) {
      const emit = (event: unknown) => controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`));
      emit({ type: "response.output_text.delta", delta: "hello" });
      await gate;
      emit({ type: "response.completed", response: response([text("hello")]) });
      controller.close();
    } }), { headers: { "Content-Type": "text/event-stream" } }));
    const agent = createBareAgent({ model: "test-model", modelClient: client, retryMissingOutput: false });
    const collected: SDKMessage[] = [];
    for await (const event of agent.query("hi")) {
      collected.push(event);
      if (event.type === "stream_event") finish();
    }
    expect(collected.find(event => event.type === "stream_event")).toMatchObject({ event: { type: "response.output_text.delta" } });
    expect(collected.at(-1)).toMatchObject({ subtype: "success", result: "hello" });
  });

  test.each([false, true])("incomplete response maps output exhaustion and refusal (stream=%s)", async stream => {
    let next = response([text("partial")], { status: "incomplete", incomplete_details: { reason: "max_output_tokens" } });
    const { client } = endpoint(() => stream ? sse([{ type: next.status === "incomplete" ? "response.incomplete" : "response.completed", response: next }]) : Response.json(next));
    expect((await client.createMessage(request({ stream }))).stopReason).toBe("max_tokens");
    next = response([{ ...text(), content: [{ type: "refusal", refusal: "cannot help" }] }]);
    expect(await client.createMessage(request({ stream }))).toMatchObject({ stopReason: "refusal", content: [{ text: "cannot help" }] });
  });

  test.each(["[]", "null", '{"city":'])("malformed arguments %s trigger tool feedback rather than execution", async args => {
    const { client, bodies } = endpoint(() => Response.json(response(bodies.length === 1 ? [call(args)] : [text()])));
    let executions = 0;
    const agent = createBareAgent({ model: "test-model", modelClient: client, retryMissingOutput: false,
      tools: [tool("weather", "weather", z.object({ city: z.string().optional() }), async () => { executions++; return { content: "ok" }; })] });
    expect((await agent.prompt("go", { stream: false })).subtype).toBe("success");
    expect(executions).toBe(0);
    expect(bodies[1].input.find((item: any) => item.type === "function_call_output").output).toContain("weather");
  });

  test("outputSchema retries with forced submit_output, without requiring reasoning on ordinary models", async () => {
    const { client, bodies } = endpoint(() => Response.json(response(bodies.length === 1 ? [text("forgot")]
      : [call('{"answer":42}', "submit_output")])));
    const agent = createBareAgent({ model: "test-model", modelClient: client, outputSchema: z.object({ answer: z.number() }) });
    expect(await agent.prompt("answer", { stream: false })).toMatchObject({ subtype: "success", structuredResult: { answer: 42 } });
    expect(bodies[1].tool_choice).toEqual({ type: "function", name: "submit_output" });
    expect(bodies[1].reasoning).toBeUndefined();
  });

  test.each([
    ["https://api.deepseek.com", true],
    ["https://api.deepseek.com/v1", true],
    ["https://api.deepseek.com.proxy.example/v1", false],
    ["https://api.openai.com/v1", false],
  ])("submission recovery disables DeepSeek thinking only for the official host (%s)", async (baseURL, deepSeek) => {
    const bodies: any[] = [];
    const client = createOpenAIResponsesClient({ apiKey: "test-key", baseURL,
      reasoningEffort: "low", fetch: async (_url, init) => {
        const body = JSON.parse(String(init?.body));
        bodies.push(body);
        if (body.tool_choice && deepSeek && body.reasoning?.effort !== "none") {
          return Response.json({ error: { message: "Thinking mode does not support this tool_choice" } }, { status: 400 });
        }
        return Response.json(response(bodies.length === 1 ? [text("forgot")]
          : [call('{"answer":42}', "submit_output")]));
      },
    });
    const agent = createBareAgent({ model: "test-model", modelClient: client, outputSchema: z.object({ answer: z.number() }) });
    expect(await agent.prompt("answer", { stream: false })).toMatchObject({ subtype: "success", structuredResult: { answer: 42 } });
    expect(bodies[0].reasoning).toEqual({ effort: "low" });
    expect(bodies[1].tool_choice).toEqual({ type: "function", name: "submit_output" });
    expect(bodies[1].reasoning).toEqual(deepSeek ? { effort: "none" } : undefined);
  });

  test("maps JSON formats, images and reasoning configuration without mutating schemas", async () => {
    const { baseURL, bodies } = endpoint(() => Response.json(response()));
    const client = createOpenAIResponsesClient({ apiKey: "test-key", baseURL, reasoningEffort: "medium" });
    await client.createMessage(request({ outputFormat: "json", messages: [{ role: "user", content: [
      { type: "text", text: "describe" }, { type: "image", source: { type: "url", url: "https://example.com/image.png" } },
      { type: "image", source: { type: "base64", media_type: "image/png", data: "aW1hZ2U=" } },
    ] }] }));
    expect(bodies[0].reasoning).toEqual({ effort: "medium" });
    expect(bodies[0].text.format).toEqual({ type: "json_object" });
    expect(bodies[0].input[1].content[0].image_url).toBe("https://example.com/image.png");
    expect(bodies[0].input[2].content[0].image_url).toBe("data:image/png;base64,aW1hZ2U=");
    const schema = { type: "object", properties: { answer: { type: "number" } }, required: ["answer"], additionalProperties: false };
    await client.createMessage(request({ reasoningEffort: "max", outputFormat: { type: "json_schema", schema } }));
    expect(bodies[1]).toMatchObject({ reasoning: { effort: "xhigh" }, text: { format: { type: "json_schema", schema, strict: true } } });
    await client.createMessage(request({ thinkingConfig: { type: "disabled" } }));
    expect(bodies[2].reasoning).toBeUndefined();
    await expect(client.createMessage(request({ thinkingConfig: { type: "adaptive" } }))).rejects.toThrow("reasoningEffort");
    await expect(client.createMessage(request({ messages: [{ role: "user", content: [{ type: "document", source: { type: "text", media_type: "text/plain", data: "doc" } }] }] }))).rejects.toThrow("document");
    expect(bodies).toHaveLength(3);
    const noReasoning = createOpenAIResponsesClient({ apiKey: "test-key", baseURL, reasoningEffort: "none" });
    await noReasoning.createMessage(request({ thinkingConfig: { type: "disabled" } }));
    expect(bodies[3].reasoning).toEqual({ effort: "none" });
  });

  test("HTTP errors are SDK APIErrors and failed/truncated streams never become successful answers", async () => {
    const { client } = endpoint(() => Response.json({ error: { message: "unauthorized" } }, { status: 401 }));
    await expect(client.createMessage(request())).rejects.toBeInstanceOf(APIError);
    for (const events of [
      [{ type: "response.output_text.delta", delta: "partial" }],
      [{ type: "error", message: "broken" }],
      [{ type: "response.failed", response: response([], { status: "failed", error: { message: "failed" } }) }],
    ]) {
      const { client } = endpoint(() => sse(events));
      await expect(client.createMessage(request({ stream: true }))).rejects.toBeInstanceOf(APIError);
    }
  });

  test("rejects unsupported hosted-tool output rather than silently losing it", async () => {
    const { client } = endpoint(() => Response.json(response([{ type: "web_search_call", id: "ws_1" }])));
    await expect(client.createMessage(request())).rejects.toThrow("web_search_call");
  });

  test("multiple tool calls keep call IDs and multimodal results through the loop", async () => {
    const { client, bodies } = endpoint(() => Response.json(response(bodies.length === 1
      ? [call('{"city":"Paris"}', "weather", "call_a"), call('{"city":"Rome"}', "weather", "call_b")]
      : [text()])));
    const cities: string[] = [];
    const agent = createBareAgent({ model: "test-model", modelClient: client, retryMissingOutput: false,
      tools: [tool("weather", "weather", z.object({ city: z.string() }), async ({ city }) => {
        cities.push(city);
        return { content: [{ type: "text" as const, text: city },
          { type: "image" as const, source: { type: "url" as const, url: "https://example.com/weather.png" } }] };
      })] });
    expect((await agent.prompt("both", { stream: false })).subtype).toBe("success");
    expect(cities).toEqual(["Paris", "Rome"]);
    const outputs = bodies[1].input.filter((item: any) => item.type === "function_call_output");
    expect(outputs.map((item: any) => item.call_id)).toEqual(["call_a", "call_b"]);
    expect(outputs[0].output).toEqual([{ type: "input_text", text: "Paris" },
      { type: "input_image", image_url: "https://example.com/weather.png", detail: "auto" }]);
  });

  test("OpenAI context_length_exceeded participates in existing auto-compaction recovery", async () => {
    const { client, bodies } = endpoint(() => {
      if (bodies.length === 1) return Response.json({ error: {
        message: "Maximum context length exceeded", type: "invalid_request_error", code: "context_length_exceeded",
      } }, { status: 400 });
      return Response.json(response([text(bodies.length === 2 ? "Summary of earlier conversation" : "recovered")]));
    });
    const agent = createBareAgent({ model: "test-model", modelClient: client, retryMissingOutput: false,
      autoCompact: { thresholdTokens: 100000, keepRecentMessages: 2 } });
    await agent.replaceHistory([
      { role: "user", content: "old task" }, { role: "assistant", content: "old answer" },
      { role: "user", content: "recent task" }, { role: "assistant", content: "recent answer" },
    ]);
    expect(await agent.prompt("continue", { stream: false })).toMatchObject({ subtype: "success", result: "recovered" });
    expect(bodies).toHaveLength(3);
  });

  test("Agent interrupt drops partial Responses output and lets the HTTP stream close", async () => {
    const { client } = endpoint(() => new Response(new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode('data: {"type":"response.output_text.delta","delta":"partial"}\n\n'));
    } }), { headers: { "Content-Type": "text/event-stream" } }));
    const agent = createBareAgent({ model: "test-model", modelClient: client, retryMissingOutput: false });
    const events: SDKMessage[] = [];
    for await (const event of agent.query("hi")) {
      events.push(event);
      if (event.type === "stream_event") expect(agent.interrupt()).toBe(true);
    }
    expect(events.at(-1)).toMatchObject({ subtype: "interrupted" });
    expect((await agent.getHistory()).some(message => message.role === "assistant")).toBe(false);
  });

  test("cancellation and timeout close pending HTTP requests", async () => {
    const { client } = endpoint(async req => {
      await new Promise<void>(resolve => req.signal.addEventListener("abort", () => resolve(), { once: true }));
      return Response.json(response());
    });
    const controller = new AbortController();
    const pending = client.createMessage(request({ signal: controller.signal }));
    controller.abort(new Error("cancelled by host"));
    await expect(pending).rejects.toThrow("cancelled by host");
    await expect(client.createMessage(request({ timeoutMs: 30 }))).rejects.toThrow();
  });
});
