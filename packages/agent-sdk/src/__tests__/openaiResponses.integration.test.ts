import { describe, expect, test } from "bun:test";
import { z } from "zod/v4";
import { createBareAgent, createOpenAIResponsesClient, tool, type SDKMessage } from "../index.js";

// Explicitly opt in: no model fallback and no credentials embedded in source.
describe("OpenAI Responses live integration", () => {
  for (const stream of [false, true]) {
    test.skipIf(!process.env.OPENAI_API_KEY || !process.env.OPENAI_MODEL)(
      `function tool and structured submission (stream=${stream})`,
      async () => {
        const secret = String(Math.floor(Math.random() * 1_000_000));
        let executed = false;
        const agent = createBareAgent({
          model: process.env.OPENAI_MODEL!,
          modelClient: createOpenAIResponsesClient({
            apiKey: process.env.OPENAI_API_KEY,
            baseURL: process.env.OPENAI_BASE_URL,
          }),
          maxTurns: 5,
          tools: [tool("read_secret", "Read the secret value. Do not guess it.", z.object({}), async () => {
            executed = true;
            return { content: secret };
          })],
          outputSchema: z.object({ secret: z.string() }),
        });
        const messages: SDKMessage[] = [];
        for await (const message of agent.query("Call read_secret, then submit the secret using submit_output.", { stream, requestTimeoutMs: 60_000 })) {
          messages.push(message);
        }
        expect(executed).toBe(true);
        expect(messages.at(-1)).toMatchObject({ subtype: "success", structuredResult: { secret } });
        if (stream) expect(messages.some(message => message.type === "stream_event")).toBe(true);
      },
      180_000,
    );
  }

  for (const stream of [false, true]) {
    test.skipIf(!process.env.OPENAI_API_KEY || !process.env.OPENAI_MODEL ||
      !process.env.OPENAI_BASE_URL || new URL(process.env.OPENAI_BASE_URL).hostname !== "api.deepseek.com")(
      `DeepSeek recovers missing submission with thinking disabled (stream=${stream})`,
      async () => {
        const client = createOpenAIResponsesClient({
          apiKey: process.env.OPENAI_API_KEY,
          baseURL: process.env.OPENAI_BASE_URL,
          reasoningEffort: "low",
        });
        let calls = 0;
        const agent = createBareAgent({
          model: process.env.OPENAI_MODEL!, maxTurns: 3, maxTokens: 2048,
          outputSchema: z.object({ answer: z.number() }),
          systemPrompt: 'Return JSON {"answer":42} if no tools are available. Otherwise submit answer 42 using submit_output.',
          modelClient: {
            async createMessage(request) {
              // Exercise a real text-only first response, then the SDK's forced recovery.
              calls++;
              return await client.createMessage(calls === 1 ? { ...request, tools: [] } : request);
            },
          },
        });
        const result = await agent.prompt("Answer 42.", { stream, requestTimeoutMs: 60_000 });
        expect(result).toMatchObject({ subtype: "success", structuredResult: { answer: 42 } });
        expect(calls).toBe(2);
      },
      180_000,
    );
  }
});
