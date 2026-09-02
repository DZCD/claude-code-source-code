import { describe, expect, test } from "bun:test";
import { z } from "zod/v4";
import {
  agentTool,
  createAgent,
  createBareAgent,
  defineAgent,
  delegateTool,
  tool,
  type SDKMessage,
} from "../index.js";

const validAgentOptions = {
  model: "claude-test",
  modelClient: { async createMessage(): Promise<never> { throw new Error("not called"); } },
};

describe("strict option validation", () => {
  test("createAgent rejects unknown options", () => {
    expect(() =>
      createAgent({ ...validAgentOptions, bogusOption: true } as any),
    ).toThrow(/unknown option "bogusOption"/);
  });

  test("createBareAgent rejects unknown options", () => {
    expect(() =>
      createBareAgent({ ...validAgentOptions, outputschema: undefined } as any),
    ).toThrow(/unknown option "outputschema"/);
  });

  test("defineAgent rejects unknown options", () => {
    expect(() =>
      defineAgent({ ...validAgentOptions, bogusOption: true } as any),
    ).toThrow(/unknown option "bogusOption"/);
  });

  test("agentTool rejects unknown options", () => {
    const child = createBareAgent(validAgentOptions);
    expect(() =>
      agentTool("child", child, { description: "d", bogusOption: 1 } as any),
    ).toThrow(/unknown option "bogusOption"/);
  });

  test("delegateTool rejects unknown options", () => {
    const child = createBareAgent(validAgentOptions);
    expect(() =>
      delegateTool("child", "d", child, { bogusOption: 1 } as any),
    ).toThrow(/unknown option "bogusOption"/);
  });

  test("tool() rejects unknown ToolOptions keys", () => {
    expect(() =>
      tool("t", "d", z.object({}), async () => ({ content: "x" }), { bogus: 1 } as any),
    ).toThrow(/unknown option "bogus"/);
  });

  test("known options are accepted", () => {
    const child = createBareAgent(validAgentOptions);
    expect(() =>
      agentTool("child", child, { description: "d", metadata: { version: 1 } }),
    ).not.toThrow();
    expect(() => createBareAgent({ ...validAgentOptions, maxTurns: 3 })).not.toThrow();
  });
});

describe("tool metadata", () => {
  test("tool() carries metadata onto the definition", () => {
    const definition = tool(
      "t",
      "d",
      z.object({}),
      async () => ({ content: "x" }),
      { metadata: { version: "1.2.3", tags: ["a"] } },
    );
    expect(definition.metadata).toEqual({ version: "1.2.3", tags: ["a"] });
  });

  test("agentTool carries metadata onto the definition", () => {
    const child = createBareAgent(validAgentOptions);
    const definition = agentTool("child", child, {
      description: "d",
      metadata: { contractVersion: 2 },
    });
    expect(definition.metadata).toEqual({ contractVersion: 2 });
  });

  test("agentTool carries isConcurrencySafe onto the definition", () => {
    const child = createBareAgent(validAgentOptions);
    const definition = agentTool("child", child, {
      description: "d",
      isConcurrencySafe: input => input.mode === "ask",
    });
    expect(definition.isConcurrencySafe?.({ mode: "ask", task: "t" })).toBe(true);
    expect(definition.isConcurrencySafe?.({ mode: "handoff", task: "t" })).toBe(false);
  });

  test("agentTool without isConcurrencySafe has no isConcurrencySafe key", () => {
    const child = createBareAgent(validAgentOptions);
    const definition = agentTool("child", child, { description: "d" });
    expect("isConcurrencySafe" in definition).toBe(false);
  });

  test("definitions without metadata have no metadata key", () => {
    const definition = tool("t", "d", z.object({}), async () => ({ content: "x" }));
    expect("metadata" in definition).toBe(false);
  });
});

describe("workspace: false", () => {
  test("createAgent with workspace: false installs no workspace tools", async () => {
    const agent = createAgent({
      ...validAgentOptions,
      workspace: false,
      modelClient: {
        async createMessage() {
          return { role: "assistant" as const, content: [{ type: "text" as const, text: "hi" }] };
        },
      },
    });
    const messages: SDKMessage[] = [];
    for await (const message of agent.query("hi", { stream: false })) {
      messages.push(message);
    }
    expect(messages[0]).toMatchObject({ type: "system", subtype: "init", tools: [] });
  });

  test("defineAgent with workspace: false spawns bare sessions", async () => {
    const spec = defineAgent({
      ...validAgentOptions,
      workspace: false,
      modelClient: {
        async createMessage() {
          return { role: "assistant" as const, content: [{ type: "text" as const, text: "hi" }] };
        },
      },
    });
    const messages: SDKMessage[] = [];
    for await (const message of spec.spawn().query("hi", { stream: false })) {
      messages.push(message);
    }
    expect(messages[0]).toMatchObject({ type: "system", subtype: "init", tools: [] });
  });
});
