import { describe, expect, test } from "bun:test";
import { z } from "zod/v4";
import {
  agentTool,
  createAgent,
  createBareAgent,
  defineAgent,
  delegateTool,
  tool,
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

  test("definitions without metadata have no metadata key", () => {
    const definition = tool("t", "d", z.object({}), async () => ({ content: "x" }));
    expect("metadata" in definition).toBe(false);
  });
});
