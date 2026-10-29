import { describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { crc32, deflateSync } from "node:zlib";
import {
  createBareAgent, createJsonlHistoryStore, createOpenAIResponsesClient,
  type AssistantModelMessage, type ModelRequest, type SDKMessage,
} from "../index.js";

const enabled = Boolean(process.env.OPENAI_API_KEY && process.env.OPENAI_MODEL &&
  process.env.OPENAI_BASE_URL && new URL(process.env.OPENAI_BASE_URL).hostname === "api.deepseek.com");
const liveTest = test.skipIf(!enabled);
function client(reasoningEffort: "low" | "none" = "none") {
  return createOpenAIResponsesClient({ apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL, reasoningEffort });
}
function request(extra: Partial<ModelRequest> = {}): ModelRequest {
  return { model: process.env.OPENAI_MODEL!, maxTokens: 2048, stream: false,
    timeoutMs: 45_000, tools: [], messages: [{ role: "user", content: "Reply exactly READY" }], ...extra };
}
function text(message: AssistantModelMessage) {
  return message.content.filter(block => block.type === "text").map(block => block.text).join("").trim();
}
function redImage(): string {
  const chunk = (name: string, data: Buffer) => {
    const payload = Buffer.concat([Buffer.from(name), data]);
    const length = Buffer.alloc(4); length.writeUInt32BE(data.length);
    const checksum = Buffer.alloc(4); checksum.writeUInt32BE(crc32(payload));
    return Buffer.concat([length, payload, checksum]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(64, 0); header.writeUInt32BE(64, 4); header[8] = 8; header[9] = 2;
  const pixels = Buffer.alloc(64 * (64 * 3 + 1));
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) pixels[y * 193 + 1 + x * 3] = 255;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", header), chunk("IDAT", deflateSync(pixels)), chunk("IEND", Buffer.alloc(0))]).toString("base64");
}

describe("DeepSeek Responses live protocol smoke", () => {
  liveTest("text response includes usage and a normal stop reason", async () => {
    const result = await client().createMessage(request());
    expect(text(result)).toBe("READY");
    expect(result.stopReason).toBe("end_turn");
    expect(result.usage?.output_tokens).toBeGreaterThan(0);
    expect(result.usage?.input_tokens).toBeGreaterThanOrEqual(0);
  }, 60_000);

  liveTest("plaintext reasoning survives a function-call/result round trip", async () => {
    const modelClient = client("low");
    const question = { role: "user" as const, content: "Call echo with value JADE_6829. After receiving its result return exactly that value." };
    const tools = [{ name: "echo", description: "Echo a value.", input_schema: {
      type: "object", properties: { value: { type: "string" } }, required: ["value"], additionalProperties: false,
    } }];
    // DeepSeek thinking mode permits auto tool choice, but rejects forced tool choice.
    const first = await modelClient.createMessage(request({ messages: [question], tools }));
    const reasoning = first.content.find(block => block.type === "openai_reasoning");
    const call = first.content.find(block => block.type === "tool_use");
    expect(reasoning?.item.content?.some(part => part.type === "reasoning_text" && part.text.length > 0)).toBe(true);
    expect(call).toBeDefined();
    if (!call) throw new Error("Expected an echo tool call");
    expect(call.input.value).toBe("JADE_6829");
    const second = await modelClient.createMessage(request({ tools, messages: [question,
      { role: "assistant", content: JSON.parse(JSON.stringify(first.content)) },
      { role: "user", content: [{ type: "tool_result", tool_use_id: call.id, content: "JADE_6829" }] },
    ] }));
    expect(text(second)).toBe("JADE_6829");
    expect(second.stopReason).toBe("end_turn");
  }, 120_000);

  for (const outputFormat of ["json" as const, { type: "json_schema" as const, schema: {
    type: "object", properties: { answer: { type: "integer" } }, required: ["answer"], additionalProperties: false,
  } }]) {
    liveTest(`JSON output (${typeof outputFormat === "string" ? outputFormat : outputFormat.type})`, async () => {
      const result = await client().createMessage(request({ outputFormat,
        messages: [{ role: "user", content: 'Return a JSON object with answer equal to 42. Output JSON only.' }] }));
      expect(JSON.parse(text(result))).toEqual({ answer: 42 });
      expect(result.stopReason).toBe("end_turn");
    }, 60_000);
  }

  liveTest("JSONL restores reasoning and memory into a new client and Agent", async () => {
    const directory = await mkdtemp(join(tmpdir(), "deepseek-responses-history-"));
    try {
      const options = { model: process.env.OPENAI_MODEL!, maxTokens: 2048,
        historyStore: createJsonlHistoryStore({ path: join(directory, "history.jsonl") }) };
      const first = createBareAgent({ ...options, modelClient: client("low") });
      expect((await first.prompt("Remember code MINT_4172. Reply only OK.", { stream: false, requestTimeoutMs: 45_000 })).subtype).toBe("success");
      expect((await first.getHistory()).some(message => Array.isArray(message.content) &&
        message.content.some(block => block.type === "openai_reasoning"))).toBe(true);
      const restored = createBareAgent({ ...options, modelClient: client("low") });
      const result = await restored.prompt("What was the code? Reply only with it.", { stream: false, requestTimeoutMs: 45_000 });
      expect(result.subtype).toBe("success");
      expect(result.result.trim()).toBe("MINT_4172");
    } finally { await rm(directory, { recursive: true, force: true }); }
  }, 120_000);

  liveTest("base64 image is interpreted as an image", async () => {
    const result = await client().createMessage(request({ messages: [{ role: "user", content: [
      { type: "text", text: "Name the dominant color in the image. Reply with one English color word only." },
      { type: "image", source: { type: "base64", media_type: "image/png", data: redImage() } },
    ] }] }));
    expect(text(result).toLowerCase()).toBe("red");
  }, 60_000);

  liveTest("interrupt closes a real stream and drops the unfinished assistant turn", async () => {
    const agent = createBareAgent({ model: process.env.OPENAI_MODEL!, modelClient: client(), maxTokens: 2048 });
    let deltas = 0;
    let final: SDKMessage | undefined;
    for await (const event of agent.query("Write a long numbered list of 100 animals.", { requestTimeoutMs: 45_000 })) {
      if (event.type === "stream_event" && event.event.type === "response.output_text.delta") {
        deltas++; agent.interrupt();
      }
      if (event.type === "result") final = event;
    }
    expect(deltas).toBeGreaterThan(0);
    expect(final).toMatchObject({ subtype: "interrupted" });
    expect((await agent.getHistory()).some(message => message.role === "assistant")).toBe(false);
  }, 60_000);

  liveTest("exhausting max_output_tokens reports max_tokens", async () => {
    const result = await client().createMessage(request({ maxTokens: 16,
      messages: [{ role: "user", content: "List integers from 1 to 1000 separated by commas, without other text." }] }));
    expect(result.stopReason).toBe("max_tokens");
  }, 60_000);
});
