import { afterEach, describe, expect, test } from "bun:test";
import { z } from "zod/v4";
import {
  createBareAgent, createOpenAIResponsesClient, defineContextTracer, tool,
  type ContentBlock, type SDKMessage,
} from "../index.js";

type Step = { content: ContentBlock[]; stop?: "end_turn" | "tool_use" | "max_tokens"; rawArguments?: string };
const text = (value: string): Step => ({ content: [{ type: "text", text: value }] });
const submit = (answer: unknown, id = "submit_1"): Step => ({
  content: [{ type: "tool_use", id, name: "submit_output", input: { answer } }], stop: "tool_use",
});
const echo: ContentBlock = { type: "tool_use", id: "echo_1", name: "echo", input: { value: "evidence" } };
const reasoning: ContentBlock = { type: "thinking", thinking: "reasoning fixture", signature: "fixture-signature" };
const servers: ReturnType<typeof Bun.serve>[] = [];
afterEach(() => { for (const server of servers.splice(0)) server.stop(true); });
const sse = (events: unknown[]) => new Response(events.map(event => `event: ${(event as { type: string }).type}\ndata: ${JSON.stringify(event)}\n\n`).join(""), {
  headers: { "Content-Type": "text/event-stream" },
});

function transport(protocol: "anthropic" | "responses", steps: Step[]) {
  const requests: any[] = [];
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
    const body = await request.json() as any;
    requests.push(body);
    const step = steps[requests.length - 1];
    if (!step) return Response.json({ error: { message: "Unexpected extra model call" } }, { status: 500 });
    const stop = step.stop ?? (step.content.some(block => block.type === "tool_use") ? "tool_use" : "end_turn");
    if (protocol === "anthropic") {
      const message = { id: `msg_${requests.length}`, type: "message", role: "assistant", model: "test",
        content: step.content, stop_reason: stop, usage: { input_tokens: 10, output_tokens: 20 } };
      if (!body.stream) return Response.json(message);
      const events: unknown[] = [{ type: "message_start", message: { ...message, content: [], usage: { input_tokens: 10, output_tokens: 0 } } }];
      step.content.forEach((block, index) => {
        if (block.type === "tool_use") {
          events.push({ type: "content_block_start", index, content_block: { ...block, input: {} } });
          const args = step.rawArguments ?? JSON.stringify(block.input);
          events.push({ type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: args } });
        } else events.push({ type: "content_block_start", index, content_block: block });
        events.push({ type: "content_block_stop", index });
      });
      events.push({ type: "message_delta", delta: { stop_reason: stop }, usage: { output_tokens: 20 } }, { type: "message_stop" });
      return sse(events);
    }
    const output = step.content.map((block, index) => {
      if (block.type === "tool_use") return { type: "function_call", id: `fc_${index}`, call_id: block.id,
        name: block.name, arguments: step.rawArguments ?? JSON.stringify(block.input), status: "completed" };
      if (block.type === "thinking") return { type: "reasoning", id: `rs_${index}`, summary: [],
        content: [{ type: "reasoning_text", text: block.thinking }], status: "completed" };
      if (block.type === "text") return { type: "message", role: "assistant", id: `message_${index}`, status: "completed",
        content: [{ type: "output_text", text: block.text, annotations: [] }] };
      throw new Error(`Unsupported fixture ${block.type}`);
    });
    const response = { id: `resp_${requests.length}`, object: "response", model: "test", output,
      status: stop === "max_tokens" ? "incomplete" : "completed",
      incomplete_details: stop === "max_tokens" ? { reason: "max_output_tokens" } : null,
      usage: { input_tokens: 10, output_tokens: 20, input_tokens_details: { cached_tokens: 0 } } };
    if (!body.stream) return Response.json(response);
    const events: unknown[] = [{ type: "response.created", response: { ...response, output: [], status: "in_progress" } }];
    output.forEach((item, output_index) => {
      events.push({ type: "response.output_item.added", output_index, item });
      if (item.type === "function_call") events.push({ type: "response.function_call_arguments.delta", output_index, delta: item.arguments });
      events.push({ type: "response.output_item.done", output_index, item });
    });
    events.push({ type: stop === "max_tokens" ? "response.incomplete" : "response.completed", response });
    return sse(events);
  } });
  servers.push(server);
  return { requests, baseURL: String(server.url) };
}

type Scenario = { name: string; steps: Step[]; submitOutputEndTurn?: boolean; retryMissingOutput?: boolean;
  subtype?: string; answer?: number; echoes?: number; errorContains?: string; streamingOnly?: boolean };
const cases: Scenario[] = [
  { name: "default submission ends immediately", steps: [submit(42)], answer: 42 },
  { name: "invalid schema returns an error and can be corrected", steps: [submit("invalid"), submit(42, "submit_2")], answer: 42, errorContains: "submit_output" },
  { name: "submission can be followed by a summary", steps: [submit(42), text("done")], answer: 42, submitOutputEndTurn: false },
  { name: "latest valid submission wins", steps: [submit(1), submit(42, "submit_2"), text("done")], answer: 42, submitOutputEndTurn: false },
  { name: "mixed submit and work batch executes neither tool", steps: [{ content: [echo, ...submit(1).content] }, submit(42, "submit_2")], answer: 42, errorContains: "submit_output_exclusive_batch" },
  { name: "duplicate submissions in one batch are rejected", steps: [{ content: [...submit(1).content, ...submit(2, "submit_2").content] }, submit(42, "submit_3")], answer: 42, errorContains: "submit_output_exclusive_batch" },
  { name: "truncated valid-looking calls are not executed", steps: [{ ...submit(1), stop: "max_tokens" }, submit(42, "submit_2")], answer: 42, errorContains: "truncated" },
  { name: "tool-looking prose is never executed as a submission", steps: [text('<DSML><invoke name="submit_output">{"answer":1}</invoke></DSML>'), submit(42)], answer: 42 },
  { name: "missing submission is retried exactly once", steps: [text("forgot"), submit(42)], answer: 42 },
  { name: "disabled recovery fails on a text ending", steps: [text("forgot")], retryMissingOutput: false, subtype: "error_missing_output" },
  { name: "failed recovery is bounded", steps: [text("forgot"), text("forgot again")], subtype: "error_missing_output" },
  { name: "invalid recovery payload fails without a third attempt", steps: [text("forgot"), submit("invalid")], subtype: "error_missing_output" },
  { name: "completed work is not replayed during recovery", steps: [{ content: [reasoning, echo] }, text("forgot"), submit(42)], answer: 42, echoes: 1 },
  { name: "recovery can be followed by reasoning and a summary", steps: [text("forgot"), submit(42), { content: [reasoning, { type: "text", text: "done" }] }], answer: 42, submitOutputEndTurn: false },
  { name: "malformed streamed JSON returns feedback before correction", steps: [{ ...submit(1), rawArguments: '{"answer":' }, submit(42, "submit_2")], answer: 42, errorContains: "JSON", streamingOnly: true },
];

async function run(protocol: "anthropic" | "responses", scenario: Scenario, stream: boolean) {
  const { requests, baseURL } = transport(protocol, scenario.steps);
  let echoes = 0;
  const traces: string[] = [];
  const agent = createBareAgent({ model: "test", maxTurns: 6, outputSchema: z.object({ answer: z.number() }),
    ...(scenario.submitOutputEndTurn !== undefined ? { submitOutputEndTurn: scenario.submitOutputEndTurn } : {}),
    ...(scenario.retryMissingOutput !== undefined ? { retryMissingOutput: scenario.retryMissingOutput } : {}),
    tools: [tool("echo", "Echo", z.object({ value: z.string() }), async ({ value }) => { echoes++; return { content: value }; })],
    tracer: defineContextTracer({ onEvent(event) { traces.push(event.type); } }),
    ...(protocol === "anthropic" ? { apiKey: "test-key", baseURL }
      : { modelClient: createOpenAIResponsesClient({ apiKey: "test-key", baseURL }) }),
  });
  const events: SDKMessage[] = [];
  for await (const event of agent.query("answer 42", { stream })) events.push(event);
  const final = events.at(-1);
  expect(final).toMatchObject({ type: "result", subtype: scenario.subtype ?? "success" });
  if (!final || final.type !== "result") throw new Error("Missing final result");
  expect(final.structuredResult).toEqual(scenario.answer === undefined ? undefined : { answer: scenario.answer });
  expect(echoes).toBe(scenario.echoes ?? 0);
  expect(requests).toHaveLength(scenario.steps.length);
  const feedback = events.filter(event => event.type === "user").map(event => event.message.content);
  if (scenario.errorContains) expect(JSON.stringify(feedback)).toContain(scenario.errorContains);
  if (scenario.steps[0]?.content[0]?.type === "text" && scenario.retryMissingOutput !== false && requests.length > 1) {
    expect(requests[1].tools.map((item: any) => item.name)).toEqual(["submit_output"]);
    expect(requests[1].tool_choice).toEqual(protocol === "anthropic"
      ? { type: "tool", name: "submit_output" } : { type: "function", name: "submit_output" });
    if (protocol === "anthropic") expect(requests[1].thinking).toEqual({ type: "disabled" });
  }
  return { subtype: final.subtype, is_error: final.is_error, result: final.result,
    structuredResult: final.structuredResult, num_turns: final.num_turns,
    usage: final.usage, echoes, feedback, traceTypes: traces };
}

describe("Responses / Anthropic submit_output transport parity", () => {
  for (const stream of [false, true]) for (const scenario of cases) {
    if (scenario.streamingOnly && !stream) continue;
    test(`${scenario.name} (stream=${stream})`, async () => {
      const anthropic = await run("anthropic", scenario, stream);
      const responses = await run("responses", scenario, stream);
      expect(responses).toEqual(anthropic);
    });
  }
});
