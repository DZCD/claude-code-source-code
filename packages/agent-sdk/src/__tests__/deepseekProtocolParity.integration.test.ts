import { describe, expect, test } from "bun:test";
import { crc32, deflateSync } from "node:zlib";
import { z } from "zod/v4";
import {
  agentTool, createBareAgent, createOpenAIResponsesClient, defineContextTracer, tool,
  type BareAgentOptions, type ContentBlock, type SDKMessage,
} from "../index.js";

const enabled = Boolean(process.env.OPENAI_API_KEY && process.env.OPENAI_MODEL &&
  process.env.OPENAI_BASE_URL && new URL(process.env.OPENAI_BASE_URL).hostname === "api.deepseek.com");
const liveTest = test.skipIf(!enabled);
type Protocol = "anthropic" | "responses";
function options(protocol: Protocol, thinking: boolean): BareAgentOptions {
  return { model: process.env.OPENAI_MODEL!, maxTokens: 4096, maxTurns: 6,
    ...(protocol === "anthropic"
      ? { apiKey: process.env.OPENAI_API_KEY, baseURL: "https://api.deepseek.com/anthropic",
        thinkingConfig: thinking ? { type: "enabled" as const, budgetTokens: 1024 } : { type: "disabled" as const } }
      : { modelClient: createOpenAIResponsesClient({ apiKey: process.env.OPENAI_API_KEY,
        baseURL: process.env.OPENAI_BASE_URL, reasoningEffort: thinking ? "low" : "none" }) }),
  };
}
function reasoningText(content: ContentBlock[]): string {
  return content.map(block => block.type === "thinking" ? block.thinking
    : block.type === "openai_reasoning" ? (block.item.content ?? []).map(part => part.text).join("") : "").join("");
}
async function collect(agent: ReturnType<typeof createBareAgent>, prompt: string | ContentBlock[], stream: boolean) {
  const events: SDKMessage[] = [];
  for await (const event of agent.query(prompt, { stream, requestTimeoutMs: 60_000 })) events.push(event);
  const final = events.at(-1);
  expect(final).toMatchObject({ type: "result", subtype: "success", is_error: false });
  if (!final || final.type !== "result") throw new Error("Missing final result");
  return { events, final };
}
function colorImage(color: "red" | "blue") {
  const chunk = (name: string, data: Buffer) => {
    const payload = Buffer.concat([Buffer.from(name), data]);
    const length = Buffer.alloc(4); length.writeUInt32BE(data.length);
    const sum = Buffer.alloc(4); sum.writeUInt32BE(crc32(payload));
    return Buffer.concat([length, payload, sum]);
  };
  const header = Buffer.alloc(13); header.writeUInt32BE(64, 0); header.writeUInt32BE(64, 4); header[8] = 8; header[9] = 2;
  const pixels = Buffer.alloc(64 * 193);
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) pixels[y * 193 + 1 + x * 3 + (color === "red" ? 0 : 2)] = 255;
  return { type: "image" as const, source: { type: "base64" as const, media_type: "image/png" as const,
    data: Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", header),
      chunk("IDAT", deflateSync(pixels)), chunk("IEND", Buffer.alloc(0))]).toString("base64") } };
}

describe("DeepSeek Responses / Anthropic live behavior parity", () => {
  for (const protocol of ["anthropic", "responses"] as const) for (const stream of [false, true]) {
    for (const thinking of [false, true]) {
      const label = `${protocol}, stream=${stream}, thinking=${thinking}`;
      liveTest(`submit_output ends immediately and preserves reasoning/tracing (${label})`, async () => {
        const traced: ContentBlock[][] = [];
        const agent = createBareAgent({ ...options(protocol, thinking), outputSchema: z.object({ answer: z.number() }),
          tracer: defineContextTracer({ onEvent(event) {
            if (event.type === "assistant_message") traced.push((event.data.message as { content: ContentBlock[] }).content);
          } }),
        });
        const { events, final } = await collect(agent, "Compute 19 + 23. Submit the numeric answer with submit_output on its own.", stream);
        expect(final.structuredResult).toEqual({ answer: 42 });
        expect(final.num_turns).toBe(1);
        const assistants = events.filter(event => event.type === "assistant");
        const chain = assistants.map(event => reasoningText(event.message.content)).join("");
        expect(chain.length > 0).toBe(thinking);
        expect(traced.map(reasoningText).join("") === chain).toBe(true);
        const deltas = events.filter(event => event.type === "stream_event").map(event => {
          if (event.event.type === "response.reasoning_text.delta") return event.event.delta as string;
          const delta = event.event.delta as { type?: string; thinking?: string } | undefined;
          return delta?.type === "thinking_delta" ? delta.thinking ?? "" : "";
        }).join("");
        if (stream) expect(deltas === chain).toBe(true);
      }, 180_000);

      liveTest(`submit_output can continue to a natural-language close (${label})`, async () => {
        const agent = createBareAgent({ ...options(protocol, thinking), outputSchema: z.object({ answer: z.number() }),
          submitOutputEndTurn: false, systemPrompt: "Call submit_output exactly once with answer 42. After its tool result confirms recording, reply exactly DONE. Do not submit again." });
        const { events, final } = await collect(agent, "Submit answer 42, then close as instructed.", stream);
        expect(final.structuredResult).toEqual({ answer: 42 });
        expect(final.result.trim()).toBe("DONE");
        expect(final.num_turns).toBe(2);
        expect(events.some(event => event.type === "user" && JSON.stringify(event.message.content).includes("Structured output recorded"))).toBe(true);
      }, 180_000);

      liveTest(`image tool results feed a structured submission (${label})`, async () => {
        let loads = 0;
        const agent = createBareAgent({ ...options(protocol, thinking), outputSchema: z.object({ color: z.enum(["red", "blue"]) }),
          tools: [tool("load_image", "Load the evidence image. Inspect it before submitting a color.", z.object({}), async () => {
            loads++; return { content: [{ type: "text" as const, text: "Evidence image:" }, colorImage("blue")] };
          })],
        });
        const { final } = await collect(agent, "First call load_image to retrieve the image. Inspect its dominant color, then submit that color using submit_output. Do not guess before loading.", stream);
        expect(loads).toBe(1);
        expect(final.structuredResult).toEqual({ color: "blue" });
        expect(final.num_turns).toBe(2);
      }, 180_000);
    }

    liveTest(`multiple images preserve order in a structured submission (${protocol}, stream=${stream})`, async () => {
      const agent = createBareAgent({ ...options(protocol, true), outputSchema: z.object({
        first: z.enum(["red", "blue"]), second: z.enum(["red", "blue"]),
      }) });
      const { final } = await collect(agent, [
        { type: "text", text: "Inspect these two images in order. Submit their dominant colors as first and second using submit_output." },
        colorImage("red"), colorImage("blue"),
      ], stream);
      expect(final.structuredResult).toEqual({ first: "red", second: "blue" });
    }, 180_000);

    liveTest(`invalid refined submission can be corrected (${protocol}, stream=${stream})`, async () => {
      let rejected = 0;
      const outputSchema = z.object({ answer: z.number() }).superRefine((value, context) => {
        if (value.answer === 13) { rejected++; context.addIssue({ code: "custom", message: "Answer 13 is rejected. Submit answer 42 instead." }); }
      });
      const agent = createBareAgent({ ...options(protocol, true), outputSchema,
        systemPrompt: "First call submit_output with answer 13. If the tool rejects it, follow the tool's correction instruction and submit again in a separate turn." });
      const { events, final } = await collect(agent, "Perform the two-step submission validation exercise.", stream);
      expect(rejected).toBe(1);
      expect(final.structuredResult).toEqual({ answer: 42 });
      expect(events.some(event => event.type === "user" && JSON.stringify(event.message.content).includes("Answer 13 is rejected"))).toBe(true);
    }, 180_000);

    liveTest(`last of two submissions wins before a summary (${protocol}, stream=${stream})`, async () => {
      const agent = createBareAgent({ ...options(protocol, true), outputSchema: z.object({ answer: z.number() }), submitOutputEndTurn: false,
        systemPrompt: "Call submit_output with answer 1. Once recorded, revise by calling submit_output with answer 42 in a separate turn. After the second recording, reply exactly DONE. Submit exactly twice." });
      const { events, final } = await collect(agent, "Perform both submissions and finish.", stream);
      expect(final.structuredResult).toEqual({ answer: 42 });
      expect(final.result.trim()).toBe("DONE");
      const values = events.filter(event => event.type === "assistant").flatMap(event => event.message.content
        .flatMap(block => block.type === "tool_use" && block.name === "submit_output" ? [block.input.answer] : []));
      expect(values).toEqual([1, 42]);
    }, 180_000);

    liveTest(`URL image is fetched and understood (${protocol}, stream=${stream})`, async () => {
      const agent = createBareAgent({ ...options(protocol, false), outputSchema: z.object({ language: z.string() }) });
      const { final } = await collect(agent, [
        { type: "text", text: "What programming language does this logo represent? Submit its name as language using submit_output." },
        { type: "image", source: { type: "url", url: "https://www.python.org/static/community_logos/python-logo.png" } },
      ], stream);
      expect((final.structuredResult as { language: string }).language.toLowerCase()).toBe("python");
    }, 180_000);
  }

  for (const effort of ["low", "high", "max"] as const) for (const stream of [false, true]) {
    liveTest(`query reasoning override preserves ${effort} (responses, stream=${stream})`, async () => {
      let sentEffort: string | undefined;
      const client = createOpenAIResponsesClient({ apiKey: process.env.OPENAI_API_KEY,
        baseURL: process.env.OPENAI_BASE_URL, reasoningEffort: "none",
        fetch: async (input, init) => {
          sentEffort = JSON.parse(String(init?.body)).reasoning?.effort;
          return await fetch(input, init);
        },
      });
      const result = await client.createMessage({ model: process.env.OPENAI_MODEL!, maxTokens: 4096, stream,
        reasoningEffort: effort, tools: [], timeoutMs: 60_000,
        messages: [{ role: "user", content: "Calculate 13 times 17. Return only the final number." }] });
      expect(sentEffort).toBe(effort);
      expect(reasoningText(result.content).length).toBeGreaterThan(0);
      expect(result.content.filter(block => block.type === "text").map(block => block.text).join("").trim()).toBe("221");
      expect(result.stopReason).toBe("end_turn");
    }, 90_000);
  }

  for (const protocol of ["anthropic", "responses"] as const) {
    liveTest(`interrupt during reasoning does not persist a partial assistant (${protocol})`, async () => {
      const agent = createBareAgent(options(protocol, true));
      let reasoningDeltas = 0;
      let final: SDKMessage | undefined;
      for await (const event of agent.query("Compute 987 times 654 carefully, then explain the result.", { requestTimeoutMs: 60_000 })) {
        if (event.type === "stream_event") {
          const delta = event.event.delta as { type?: string } | undefined;
          if (event.event.type === "response.reasoning_text.delta" || delta?.type === "thinking_delta") {
            reasoningDeltas++; agent.interrupt();
          }
        }
        if (event.type === "result") final = event;
      }
      expect(reasoningDeltas).toBeGreaterThan(0);
      expect(final).toMatchObject({ subtype: "interrupted", is_error: false });
      expect((await agent.getHistory()).some(message => message.role === "assistant")).toBe(false);
    }, 90_000);
  }

  for (const protocol of ["anthropic", "responses"] as const) for (const stream of [false, true]) {
    liveTest(`agentTool inherits a child's structured submission despite its closing prose (${protocol}, stream=${stream})`, async () => {
      const child = createBareAgent({ ...options(protocol, true), outputSchema: z.object({ answer: z.number() }),
        submitOutputEndTurn: false,
        systemPrompt: "Call submit_output exactly once with answer 42. After the tool confirms recording, reply exactly CHILD_DONE and do not submit again." });
      const parent = createBareAgent({ ...options(protocol, true), outputSchema: z.object({ answer: z.number() }),
        tools: [agentTool("solver", child, { description: "Solve the task and return its structured answer.",
          inputSchema: z.object({ question: z.string() }), mapInput: input => input.question })],
        systemPrompt: "First call solver to produce answer 42. After receiving its structured answer, call submit_output with that answer on its own. Do not skip solver." });
      const { final } = await collect(parent, "Use solver and submit the returned answer.", stream);
      expect(final.structuredResult).toEqual({ answer: 42 });
      const toolResults = (await parent.getHistory()).flatMap(message => Array.isArray(message.content)
        ? message.content.filter(block => block.type === "tool_result") : []);
      expect(toolResults.some(block => typeof block.content === "string" && block.content.includes('"answer":42'))).toBe(true);
      expect(toolResults.some(block => typeof block.content === "string" && block.content.includes("CHILD_DONE"))).toBe(false);
      const childText = (await child.getHistory()).filter(message => message.role === "assistant").flatMap(message =>
        Array.isArray(message.content) ? message.content.filter(block => block.type === "text").map(block => block.text) : []);
      expect(childText.at(-1)?.trim()).toBe("CHILD_DONE");
    }, 180_000);
  }

  for (const stream of [false, true]) {
    liveTest(`recovery finishes its non-thinking tool turn, then restores thinking on new input (responses, stream=${stream})`, async () => {
      const efforts: string[] = [];
      const client = createOpenAIResponsesClient({ apiKey: process.env.OPENAI_API_KEY,
        baseURL: process.env.OPENAI_BASE_URL, reasoningEffort: "low",
        fetch: async (input, init) => {
          efforts.push(JSON.parse(String(init?.body)).reasoning?.effort);
          return await fetch(input, init);
        },
      });
      let calls = 0;
      const agent = createBareAgent({ model: process.env.OPENAI_MODEL!, maxTokens: 4096, maxTurns: 4,
        outputSchema: z.object({ answer: z.number() }), submitOutputEndTurn: false,
        systemPrompt: "When no tools are available reply READY. When tools are available submit answer 42 exactly once. After a tool confirms recording, reply exactly DONE and do not submit again.",
        modelClient: { async createMessage(request) {
          calls++; return await client.createMessage(calls === 1 ? { ...request, tools: [] } : request);
        } },
      });
      const { events, final } = await collect(agent, "Submit answer 42 and close as instructed.", stream);
      expect(final.structuredResult).toEqual({ answer: 42 });
      expect(final.result.trim()).toBe("DONE");
      expect(efforts).toEqual(["low", "none", "none"]);
      const hasReasoning = events.filter(event => event.type === "assistant").map(event => reasoningText(event.message.content).length > 0);
      expect(hasReasoning).toEqual([true, false, false]);
      const beforeNext = efforts.length;
      const nextEvents: SDKMessage[] = [];
      for await (const event of agent.query("New user task: compute 6 times 7, submit the numeric answer using submit_output, then reply DONE.",
        { stream, requestTimeoutMs: 60_000 })) nextEvents.push(event);
      expect(efforts[beforeNext]).toBe("low");
      expect(nextEvents.at(-1)).toMatchObject({ subtype: "success", is_error: false, structuredResult: { answer: 42 } });
      expect(nextEvents.some(event => event.type === "assistant" && reasoningText(event.message.content).length > 0)).toBe(true);
    }, 180_000);
  }

  for (const thinkingConfig of [{ type: "enabled" as const, budgetTokens: 1024 }, { type: "adaptive" as const }]) {
    for (const stream of [false, true]) {
      liveTest(`existing thinkingConfig ${thinkingConfig.type} overrides a factory off switch (responses, stream=${stream})`, async () => {
        const agent = createBareAgent({ model: process.env.OPENAI_MODEL!, maxTokens: 4096,
          modelClient: createOpenAIResponsesClient({ apiKey: process.env.OPENAI_API_KEY,
            baseURL: process.env.OPENAI_BASE_URL, reasoningEffort: "none" }),
          outputSchema: z.object({ answer: z.number() }),
        });
        const events: SDKMessage[] = [];
        for await (const event of agent.query("Calculate 13 times 17 and submit the numeric answer using submit_output.", {
          thinkingConfig, stream, requestTimeoutMs: 60_000,
        })) events.push(event);
        expect(events.at(-1)).toMatchObject({ subtype: "success", is_error: false, structuredResult: { answer: 221 } });
        expect(events.some(event => event.type === "assistant" && reasoningText(event.message.content).length > 0)).toBe(true);
      }, 90_000);
    }
  }

  for (const stream of [false, true]) {
    liveTest(`Anthropic recovery also keeps the tool continuation non-thinking (stream=${stream})`, async () => {
      const agent = createBareAgent({ ...options("anthropic", true), outputSchema: z.object({ answer: z.number() }),
        submitOutputEndTurn: false,
        systemPrompt: "Diagnostic sequence: On the FIRST response reply only READY with NO tool call. If a later reminder requests submit_output, submit answer 42. Once a tool result says Structured output recorded, reply exactly DONE without tools." });
      const { events, final } = await collect(agent, "Perform the diagnostic sequence.", stream);
      expect(final.structuredResult).toEqual({ answer: 42 });
      expect(final.result.trim()).toBe("DONE");
      expect(final.num_turns).toBe(3);
      expect(events.filter(event => event.type === "assistant").map(event => reasoningText(event.message.content).length > 0))
        .toEqual([true, false, false]);
      const previousHistoryLength = (await agent.getHistory()).length;
      const next = await agent.prompt("New user task: first calculate 6 times 7, then call submit_output with that answer, then reply DONE.",
        { stream, requestTimeoutMs: 60_000 });
      expect(next).toMatchObject({ subtype: "success", is_error: false, structuredResult: { answer: 42 } });
      const history = await agent.getHistory();
      expect(history.slice(previousHistoryLength).some(message => message.role === "assistant" &&
        Array.isArray(message.content) && reasoningText(message.content).length > 0)).toBe(true);
    }, 180_000);
  }
});
