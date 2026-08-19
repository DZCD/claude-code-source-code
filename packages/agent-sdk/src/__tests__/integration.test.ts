import { describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod/v4";
import { createAgent, createAgentWorkspaceTools, agentTool, tool, type SDKMessage } from "../index.js";

async function collect(iterable: AsyncIterable<SDKMessage>): Promise<SDKMessage[]> {
  const messages: SDKMessage[] = [];
  for await (const message of iterable) {
    messages.push(message);
  }
  return messages;
}

function deepseekAgentOptions() {
  return {
    apiKey: process.env.DEEPSEEK_API_KEY,
    baseURL: process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com/anthropic",
    model: process.env.DEEPSEEK_MODEL ?? "deepseek-v4-flash",
    maxTurns: 5,
  };
}

function magicNumberTool() {
  return tool(
    "get_magic_number",
    "Return the magic number. Must be used when the user asks for the magic number.",
    z.object({}),
    async () => ({ content: "42" }),
  );
}

describe("agent-sdk integration", () => {
  test.skipIf(!process.env.ANTHROPIC_API_KEY)(
    "can call the Anthropic API when ANTHROPIC_API_KEY is set",
    async () => {
      const agent = createAgent({
        apiKey: process.env.ANTHROPIC_API_KEY,
        model: process.env.ANTHROPIC_MODEL ?? "claude-sonnet-4-6",
        maxTurns: 1,
      });

      const result = await agent.prompt("Reply with exactly: pong");

      expect(result.type).toBe("result");
      expect(result.subtype).toBe("success");
      expect(result.result.toLowerCase()).toContain("pong");
    },
  );

  test.skipIf(!process.env.DEEPSEEK_API_KEY)(
    "DeepSeek streaming text response works",
    async () => {
      const agent = createAgent(deepseekAgentOptions());

      const messages = await collect(agent.query("请只回复两个字母：OK", { stream: true }));
      const streamEvents = messages.filter(message => message.type === "stream_event");
      const result = messages.at(-1);

      expect(streamEvents.length).toBeGreaterThan(0);
      expect(result).toMatchObject({
        type: "result",
        subtype: "success",
        is_error: false,
      });
      expect((result && result.type === "result" ? result.result : "").toLowerCase()).toContain("ok");
    },
  );

  test.skipIf(!process.env.DEEPSEEK_API_KEY)(
    "DeepSeek streaming tool call works",
    async () => {
      const agent = createAgent({
        ...deepseekAgentOptions(),
        tools: [magicNumberTool()],
      });

      const messages = await collect(
        agent.query("必须调用 get_magic_number 工具获取数字，然后只回答这个数字。", {
          stream: true,
        }),
      );
      const streamEvents = messages.filter(message => message.type === "stream_event");
      const toolResult = messages.find(
        message => message.type === "user" && JSON.stringify(message.message).includes("tool_result"),
      );
      const result = messages.at(-1);

      expect(streamEvents.length).toBeGreaterThan(0);
      expect(toolResult).toBeDefined();
      expect(result).toMatchObject({
        type: "result",
        subtype: "success",
        is_error: false,
      });
      expect(result && result.type === "result" ? result.result : "").toContain("42");
    },
  );

  test.skipIf(!process.env.DEEPSEEK_API_KEY)(
    "DeepSeek non-streaming tool call works",
    async () => {
      const agent = createAgent({
        ...deepseekAgentOptions(),
        tools: [magicNumberTool()],
      });

      const messages = await collect(
        agent.query("必须调用 get_magic_number 工具获取数字，然后只回答这个数字。", {
          stream: false,
        }),
      );
      const streamEvents = messages.filter(message => message.type === "stream_event");
      const toolResult = messages.find(
        message => message.type === "user" && JSON.stringify(message.message).includes("tool_result"),
      );
      const result = messages.at(-1);

      expect(streamEvents).toHaveLength(0);
      expect(toolResult).toBeDefined();
      expect(result).toMatchObject({
        type: "result",
        subtype: "success",
        is_error: false,
      });
      expect(result && result.type === "result" ? result.result : "").toContain("42");
    },
  );

  test.skipIf(!process.env.DEEPSEEK_API_KEY)(
    "DeepSeek can use the built-in Read tool",
    async () => {
      const cwd = await mkdtemp(join(tmpdir(), "agent-sdk-deepseek-read-"));
      try {
        await writeFile(join(cwd, "answer.txt"), "codex builtin read works", "utf8");
        const agent = createAgent({
          ...deepseekAgentOptions(),
          tools: createAgentWorkspaceTools({ cwd, allowedDirectories: [cwd] }).filter(tool => tool.name === "Read"),
        });

        const messages = await collect(
          agent.query("必须调用 Read 工具读取 answer.txt，然后只回答文件内容。", {
            stream: false,
          }),
        );
        const toolResult = messages.find(
          message => message.type === "user" && JSON.stringify(message.message).includes("tool_result"),
        );
        const result = messages.at(-1);

        expect(toolResult).toBeDefined();
        expect(result).toMatchObject({
          type: "result",
          subtype: "success",
          is_error: false,
        });
        expect(result && result.type === "result" ? result.result : "").toContain("codex builtin read works");
      } finally {
        await rm(cwd, { recursive: true, force: true });
      }
    },
  );

  test.skipIf(!process.env.DEEPSEEK_API_KEY)(
    "DeepSeek submits structured output via submit_output",
    async () => {
      const agent = createAgent({
        ...deepseekAgentOptions(),
        outputSchema: z.object({
          sum: z.number(),
          note: z.string(),
        }),
      });

      const result = await agent.prompt("计算 19+23，完成后必须调用 submit_output 工具提交结果。", {
        stream: false,
      });

      expect(result).toMatchObject({
        type: "result",
        subtype: "success",
        is_error: false,
      });
      expect(result.structuredResult).toMatchObject({ sum: 42 });
    },
  );

  test.skipIf(!process.env.DEEPSEEK_API_KEY)(
    "DeepSeek parent agentTool receives the child's validated structured output",
    async () => {
      const outputSchema = z.object({
        sum: z.number(),
        note: z.string(),
      });
      const child = createAgent({
        ...deepseekAgentOptions(),
        outputSchema,
      });
      const parent = createAgent({
        ...deepseekAgentOptions(),
        tools: [
          agentTool("calculator", child, {
            description: "Compute an arithmetic task and return the structured result.",
            outputSchema,
          }),
        ],
      });

      const result = await parent.prompt(
        "必须调用 calculator 工具（mode 为 ask）计算 19+23，然后只回答得到的 sum。",
        { stream: false },
      );

      expect(result).toMatchObject({
        type: "result",
        subtype: "success",
        is_error: false,
      });
      expect(result.result).toContain("42");
    },
  );
});
