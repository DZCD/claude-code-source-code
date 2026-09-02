import { describe, expect, test } from "bun:test";
import { z } from "zod/v4";
import {
  MissingOutputError,
  agentTool,
  createBareAgent,
  defineAgent,
  tool,
  type ModelClient,
  type ModelRequest,
  type SDKMessage,
} from "../index.js";

function textAssistant(text: string) {
  return {
    role: "assistant" as const,
    content: [{ type: "text" as const, text }],
  };
}

function toolUseAssistant(id: string, name: string, input: Record<string, unknown>) {
  return {
    role: "assistant" as const,
    content: [{ type: "tool_use" as const, id, name, input }],
  };
}

function clientFromResponses(
  responses: Array<ReturnType<typeof textAssistant> | ReturnType<typeof toolUseAssistant>>,
  capturedRequests?: ModelRequest[],
): ModelClient {
  let index = 0;
  return {
    async createMessage(request) {
      capturedRequests?.push(request);
      const response = responses[index++];
      if (!response) throw new Error("No mock response available");
      return response;
    },
  };
}

async function collect(iterable: AsyncIterable<SDKMessage>): Promise<SDKMessage[]> {
  const messages: SDKMessage[] = [];
  for await (const message of iterable) {
    messages.push(message);
  }
  return messages;
}

const outputSchema = z.object({
  answer: z.number(),
  summary: z.string(),
});

function submittedPayload() {
  return { answer: 42, summary: "done" };
}

describe("AgentOptions.outputSchema", () => {
  test("injects submit_output and ends the run with the validated structuredResult", async () => {
    const requests: ModelRequest[] = [];
    const agent = createBareAgent({
      model: "claude-test",
      outputSchema,
      modelClient: clientFromResponses(
        [toolUseAssistant("submit_1", "submit_output", submittedPayload())],
        requests,
      ),
    });

    const messages = await collect(agent.query("Do the task", { stream: false }));

    expect(messages[0]).toMatchObject({
      type: "system",
      subtype: "init",
      tools: ["submit_output"],
    });
    // The model is shown the zod schema converted to JSON schema.
    const toolSchema = requests[0]?.tools.find(definition => definition.name === "submit_output");
    expect(toolSchema?.input_schema).toMatchObject({ type: "object" });

    const result = messages.at(-1);
    expect(result).toMatchObject({
      type: "result",
      subtype: "success",
      is_error: false,
      structuredResult: submittedPayload(),
    });
  });

  test("schema violations are returned to the loop so the model can retry", async () => {
    const agent = createBareAgent({
      model: "claude-test",
      outputSchema,
      modelClient: clientFromResponses([
        toolUseAssistant("submit_bad", "submit_output", { answer: "not-a-number" }),
        toolUseAssistant("submit_good", "submit_output", submittedPayload()),
      ]),
    });

    const messages = await collect(agent.query("Do the task", { stream: false }));

    const toolResultMessage = messages.find(
      message => message.type === "user" && typeof message.tool_use_result === "string",
    );
    expect(toolResultMessage).toMatchObject({ type: "user" });
    expect(String((toolResultMessage as { tool_use_result: string }).tool_use_result)).toContain(
      "Tool submit_output failed",
    );

    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "success",
      structuredResult: submittedPayload(),
    });
  });

  test("ending the turn without submitting fails with error_missing_output", async () => {
    const agent = createBareAgent({
      model: "claude-test",
      outputSchema,
      modelClient: clientFromResponses([textAssistant("Here is the answer in prose.")]),
    });

    const result = await agent.prompt("Do the task", { stream: false });

    expect(result.subtype).toBe("error_missing_output");
    expect(result.is_error).toBe(true);
    expect(result.error).toBeInstanceOf(MissingOutputError);
    expect(result.structuredResult).toBeUndefined();
  });

  test("submit_output must own its batch: mixed batches are rejected and the loop continues", async () => {
    const echo = tool("echo", "Echo input.", z.object({ value: z.string() }), async input => ({
      content: input.value,
    }));
    const agent = createBareAgent({
      model: "claude-test",
      tools: [echo],
      outputSchema,
      modelClient: clientFromResponses([
        {
          role: "assistant" as const,
          content: [
            { type: "tool_use" as const, id: "echo_1", name: "echo", input: { value: "hi" } },
            { type: "tool_use" as const, id: "submit_1", name: "submit_output", input: submittedPayload() },
          ],
        },
        toolUseAssistant("submit_2", "submit_output", submittedPayload()),
      ]),
    });

    const messages = await collect(agent.query("Do the task", { stream: false }));

    const rejected = messages.find(
      message =>
        message.type === "user" &&
        JSON.stringify(message.tool_use_result).includes("submit_output_exclusive_batch"),
    );
    expect(rejected).toBeDefined();

    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "success",
      structuredResult: submittedPayload(),
    });
  });

  test("two submit_output calls in one batch are rejected", async () => {
    const agent = createBareAgent({
      model: "claude-test",
      outputSchema,
      modelClient: clientFromResponses([
        {
          role: "assistant" as const,
          content: [
            { type: "tool_use" as const, id: "submit_1", name: "submit_output", input: submittedPayload() },
            { type: "tool_use" as const, id: "submit_2", name: "submit_output", input: submittedPayload() },
          ],
        },
        toolUseAssistant("submit_3", "submit_output", submittedPayload()),
      ]),
    });

    const messages = await collect(agent.query("Do the task", { stream: false }));

    expect(
      messages.some(
        message =>
          message.type === "user" &&
          JSON.stringify(message.tool_use_result).includes("submit_output_exclusive_batch"),
      ),
    ).toBe(true);
    expect(messages.at(-1)).toMatchObject({ type: "result", subtype: "success" });
  });

  test("a user tool named submit_output collides at creation time", () => {
    const clash = tool("submit_output", "User tool.", z.object({}), async () => ({ content: "x" }));
    expect(() =>
      createBareAgent({
        model: "claude-test",
        tools: [clash],
        outputSchema,
        modelClient: clientFromResponses([]),
      }),
    ).toThrow(/reserved/);
  });

  test("addTools rejects a later collision with submit_output", () => {
    const agent = createBareAgent({
      model: "claude-test",
      outputSchema,
      modelClient: clientFromResponses([]),
    });
    const clash = tool("submit_output", "User tool.", z.object({}), async () => ({ content: "x" }));
    expect(() => agent.addTools([clash])).toThrow(/reserved/);
  });

  test("without outputSchema there is no submit_output tool and plain-text endings succeed", async () => {
    const agent = createBareAgent({
      model: "claude-test",
      modelClient: clientFromResponses([textAssistant("plain answer")]),
    });

    const messages = await collect(agent.query("Say hi", { stream: false }));

    expect(messages[0]).toMatchObject({ type: "system", subtype: "init", tools: [] });
    expect(messages.at(-1)).toMatchObject({ type: "result", subtype: "success", result: "plain answer" });
  });
});

describe("AgentOptions.submitOutputEndTurn", () => {
  test("endTurn: false records the submission and lets the model close with a summary", async () => {
    const requests: ModelRequest[] = [];
    const agent = createBareAgent({
      model: "claude-test",
      outputSchema,
      submitOutputEndTurn: false,
      modelClient: clientFromResponses(
        [
          toolUseAssistant("submit_1", "submit_output", submittedPayload()),
          textAssistant("In short: the answer is 42."),
        ],
        requests,
      ),
    });

    const result = await agent.prompt("Do the task", { stream: false });

    expect(result).toMatchObject({
      subtype: "success",
      is_error: false,
      result: "In short: the answer is 42.",
      structuredResult: submittedPayload(),
    });
    // The submission's tool result went back to the model instead of ending the run.
    const followUp = requests[1]?.messages.at(-1);
    expect(followUp).toMatchObject({
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "submit_1" }],
    });
    expect(JSON.stringify(followUp)).toContain("Structured output recorded");
  });

  test("re-submitting revises the recorded payload: last submission wins", async () => {
    const agent = createBareAgent({
      model: "claude-test",
      outputSchema,
      submitOutputEndTurn: false,
      modelClient: clientFromResponses([
        toolUseAssistant("submit_1", "submit_output", { answer: 1, summary: "first draft" }),
        toolUseAssistant("submit_2", "submit_output", submittedPayload()),
        textAssistant("done"),
      ]),
    });

    const result = await agent.prompt("Do the task", { stream: false });

    expect(result).toMatchObject({ subtype: "success", structuredResult: submittedPayload() });
  });

  test("ending without any submission still fails with error_missing_output", async () => {
    const agent = createBareAgent({
      model: "claude-test",
      outputSchema,
      submitOutputEndTurn: false,
      modelClient: clientFromResponses([textAssistant("prose without submitting")]),
    });

    const result = await agent.prompt("Do the task", { stream: false });

    expect(result.subtype).toBe("error_missing_output");
    expect(result.error).toBeInstanceOf(MissingOutputError);
    expect(result.structuredResult).toBeUndefined();
  });

  test("submit_output must still own its batch when endTurn is false", async () => {
    const echo = tool("echo", "Echo input.", z.object({ value: z.string() }), async input => ({
      content: input.value,
    }));
    const agent = createBareAgent({
      model: "claude-test",
      tools: [echo],
      outputSchema,
      submitOutputEndTurn: false,
      modelClient: clientFromResponses([
        {
          role: "assistant" as const,
          content: [
            { type: "tool_use" as const, id: "echo_1", name: "echo", input: { value: "hi" } },
            { type: "tool_use" as const, id: "submit_1", name: "submit_output", input: submittedPayload() },
          ],
        },
        toolUseAssistant("submit_2", "submit_output", submittedPayload()),
        textAssistant("closing"),
      ]),
    });

    const messages = await collect(agent.query("Do the task", { stream: false }));

    expect(
      messages.some(
        message =>
          message.type === "user" &&
          JSON.stringify(message.tool_use_result).includes("submit_output_exclusive_batch"),
      ),
    ).toBe(true);
    expect(messages.at(-1)).toMatchObject({
      type: "result",
      subtype: "success",
      structuredResult: submittedPayload(),
    });
  });

  test("agentTool ask still consumes structuredResult from a spec that closes with a summary", async () => {
    const requests: ModelRequest[] = [];
    const childSpec = defineAgent({
      model: "claude-test-child",
      outputSchema,
      submitOutputEndTurn: false,
      modelClient: clientFromResponses([
        toolUseAssistant("submit_1", "submit_output", submittedPayload()),
        textAssistant("child closing summary"),
      ]),
    });
    const parent = createBareAgent({
      model: "claude-test-parent",
      tools: [
        agentTool("solver", childSpec, {
          description: "Solve a task.",
          outputSchema,
        }),
      ],
      modelClient: clientFromResponses(
        [
          toolUseAssistant("call_1", "solver", { mode: "ask", task: "solve it" }),
          textAssistant("parent done"),
        ],
        requests,
      ),
    });

    const result = await parent.prompt("Use the solver", { stream: false });

    expect(result.subtype).toBe("success");
    const toolResults = JSON.stringify(requests[1]?.messages);
    expect(toolResults).toContain(JSON.stringify(JSON.stringify(submittedPayload())));
  });
});

describe("agentTool outputSchema", () => {
  function createChild(responses: Array<ReturnType<typeof textAssistant> | ReturnType<typeof toolUseAssistant>>) {
    return createBareAgent({
      model: "claude-test-child",
      outputSchema,
      modelClient: clientFromResponses(responses),
    });
  }

  function createParent(child: ReturnType<typeof createChild>, capturedRequests?: ModelRequest[]) {
    return createBareAgent({
      model: "claude-test-parent",
      tools: [
        agentTool("solver", child, {
          description: "Solve a task.",
          outputSchema,
        }),
      ],
      modelClient: clientFromResponses(
        [
          toolUseAssistant("call_1", "solver", { mode: "ask", task: "solve it" }),
          textAssistant("parent done"),
        ],
        capturedRequests,
      ),
    });
  }

  test("ask returns the child's validated structured output as JSON", async () => {
    const requests: ModelRequest[] = [];
    const parent = createParent(
      createChild([toolUseAssistant("submit_1", "submit_output", submittedPayload())]),
      requests,
    );

    const result = await parent.prompt("Use the solver", { stream: false });

    expect(result.subtype).toBe("success");
    const toolResults = JSON.stringify(requests[1]?.messages);
    expect(toolResults).toContain(JSON.stringify(JSON.stringify(submittedPayload())));
  });

  test("a child that ends without submitting produces a child_output_invalid tool error", async () => {
    const requests: ModelRequest[] = [];
    const parent = createParent(createChild([textAssistant("no submission")]), requests);

    const result = await parent.prompt("Use the solver", { stream: false });

    expect(result.subtype).toBe("success");
    const toolResults = JSON.stringify(requests[1]?.messages);
    expect(toolResults).toContain("child_output_invalid");
  });
});

describe("agentTool typed delegation (inputSchema + mapInput)", () => {
  const judgeInputSchema = z.object({
    caseSummary: z.string(),
    score: z.number(),
  });

  function createJudge(childRequests: ModelRequest[]) {
    return defineAgent({
      model: "claude-test-child",
      outputSchema,
      modelClient: clientFromResponses(
        [toolUseAssistant("submit_1", "submit_output", submittedPayload())],
        childRequests,
      ),
    });
  }

  function createTypedParent(child: ReturnType<typeof createJudge>, parentRequests?: ModelRequest[]) {
    return createBareAgent({
      model: "claude-test-parent",
      tools: [
        agentTool("judge", child, {
          description: "Judge a case.",
          inputSchema: judgeInputSchema,
          mapInput: input => `Judge this case: ${input.caseSummary} (score ${input.score})`,
          // outputSchema omitted on purpose: inherited from the target spec.
        }),
      ],
      modelClient: clientFromResponses(
        [
          toolUseAssistant("call_1", "judge", { caseSummary: "case A", score: 3 }),
          textAssistant("parent done"),
        ],
        parentRequests,
      ),
    });
  }

  test("typed input is validated, mapped to the child prompt, and outputSchema is inherited", async () => {
    const childRequests: ModelRequest[] = [];
    const parentRequests: ModelRequest[] = [];
    const parent = createTypedParent(createJudge(childRequests), parentRequests);

    const result = await parent.prompt("Judge case A", { stream: false });

    expect(result.subtype).toBe("success");
    // mapInput projected the validated input into the child's prompt.
    expect(JSON.stringify(childRequests[0]?.messages)).toContain("Judge this case: case A (score 3)");
    // The parent omitted outputSchema; the child spec's declaration was
    // inherited, so the tool result is the validated output as JSON.
    expect(JSON.stringify(parentRequests[1]?.messages)).toContain(
      JSON.stringify(JSON.stringify(submittedPayload())),
    );
  });

  test("invalid typed input is rejected before the child is invoked", async () => {
    const childRequests: ModelRequest[] = [];
    const parentRequests: ModelRequest[] = [];
    const parent = createBareAgent({
      model: "claude-test-parent",
      tools: [
        agentTool("judge", createJudge(childRequests), {
          description: "Judge a case.",
          inputSchema: judgeInputSchema,
          mapInput: input => `Judge this case: ${input.caseSummary}`,
        }),
      ],
      modelClient: clientFromResponses(
        [
          toolUseAssistant("call_1", "judge", { caseSummary: "case A", score: "not-a-number" }),
          textAssistant("parent done"),
        ],
        parentRequests,
      ),
    });

    const result = await parent.prompt("Judge case A", { stream: false });

    expect(result.subtype).toBe("success");
    // The parse failure went back to the parent loop as an error tool_result.
    expect(JSON.stringify(parentRequests[1]?.messages)).toContain("Tool judge failed");
    // The child was never invoked.
    expect(childRequests).toHaveLength(0);
  });

  test("inputSchema without mapInput (and vice versa) throws at assembly time", () => {
    const child = createJudge([]);
    expect(() =>
      agentTool("judge", child, {
        description: "Judge a case.",
        inputSchema: judgeInputSchema,
      }),
    ).toThrow(/mapInput/);
    expect(() =>
      agentTool("judge", child, {
        description: "Judge a case.",
        mapInput: () => "task",
      }),
    ).toThrow(/inputSchema/);
  });
});

describe("agentTool outputSchema inheritance and assembly-time checks", () => {
  test("an explicit outputSchema equal in structure to the target's does not throw", () => {
    const child = createBareAgent({
      model: "claude-test-child",
      outputSchema,
      modelClient: clientFromResponses([]),
    });
    // A separately constructed but structurally identical schema is accepted.
    const sameShape = z.object({ answer: z.number(), summary: z.string() });
    expect(() =>
      agentTool("child", child, { description: "Run the child.", outputSchema: sameShape }),
    ).not.toThrow();
  });

  test("an explicit outputSchema that drifts from the target's throws at assembly time", () => {
    const child = createBareAgent({
      model: "claude-test-child",
      outputSchema,
      modelClient: clientFromResponses([]),
    });
    const drifted = z.object({ answer: z.number() });
    expect(() =>
      agentTool("child", child, { description: "Run the child.", outputSchema: drifted }),
    ).toThrow(/does not match the target's declared outputSchema/);
  });

  test("a live Agent target also exposes its declaration for inheritance", async () => {
    const child = createBareAgent({
      model: "claude-test-child",
      outputSchema,
      modelClient: clientFromResponses([toolUseAssistant("s1", "submit_output", submittedPayload())]),
    });
    expect(child.outputSchema).toBe(outputSchema);

    const parentRequests: ModelRequest[] = [];
    const parent = createBareAgent({
      model: "claude-test-parent",
      tools: [agentTool("child", child, { description: "Run the child." })],
      modelClient: clientFromResponses(
        [
          toolUseAssistant("call_1", "child", { mode: "ask", task: "go" }),
          textAssistant("done"),
        ],
        parentRequests,
      ),
    });

    const result = await parent.prompt("Use the child", { stream: false });
    expect(result.subtype).toBe("success");
    expect(JSON.stringify(parentRequests[1]?.messages)).toContain(
      JSON.stringify(JSON.stringify(submittedPayload())),
    );
  });
});

describe("agentTool structured channel without a child outputSchema declaration", () => {
  // A child that delivers via its own domain-validating submit tool: the tool
  // attaches the payload as structuredResult with endTurn, no outputSchema.
  function createDomainChild(payload: Record<string, unknown>) {
    return createBareAgent({
      model: "claude-test-child",
      tools: [
        tool(
          "deliver_judgment",
          "Deliver the validated judgment.",
          z.object({ verdict: z.string(), citations: z.array(z.string()) }),
          async input => ({
            content: "Judgment delivered.",
            endTurn: true,
            structuredResult: input,
          }),
        ),
      ],
      modelClient: clientFromResponses([
        toolUseAssistant("deliver_1", "deliver_judgment", payload),
      ]),
    });
  }

  const payload = { verdict: "supported", citations: ["doc-1"] };

  function parentRequests(): ModelRequest[] {
    return [];
  }

  async function runParent(childToolsOptions: { outputSchema?: z.ZodTypeAny }) {
    const requests = parentRequests();
    const parent = createBareAgent({
      model: "claude-test-parent",
      tools: [
        agentTool("judge", createDomainChild(payload), {
          description: "Judge a case.",
          ...childToolsOptions,
        }),
      ],
      modelClient: clientFromResponses(
        [
          toolUseAssistant("call_1", "judge", { mode: "ask", task: "judge it" }),
          textAssistant("parent done"),
        ],
        requests,
      ),
    });
    const result = await parent.prompt("Use the judge", { stream: false });
    return { result, requests };
  }

  test("explicit outputSchema with an undeclared target validates the structuredResult", async () => {
    const { result, requests } = await runParent({
      outputSchema: z.object({ verdict: z.string(), citations: z.array(z.string()) }),
    });
    expect(result.subtype).toBe("success");
    expect(JSON.stringify(requests[1]?.messages)).toContain(JSON.stringify(JSON.stringify(payload)));
  });

  test("explicit outputSchema rejects a payload that fails it, child_output_invalid", async () => {
    const { result, requests } = await runParent({
      outputSchema: z.object({ verdict: z.literal("rejected"), citations: z.array(z.string()) }),
    });
    expect(result.subtype).toBe("success");
    expect(JSON.stringify(requests[1]?.messages)).toContain("child_output_invalid");
  });

  test("no schema anywhere: the structuredResult is passed through as JSON", async () => {
    const { result, requests } = await runParent({});
    expect(result.subtype).toBe("success");
    expect(JSON.stringify(requests[1]?.messages)).toContain(JSON.stringify(JSON.stringify(payload)));
  });
});
