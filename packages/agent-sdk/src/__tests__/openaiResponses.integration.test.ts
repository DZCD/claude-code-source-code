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
});
