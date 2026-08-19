import { describe, expect, test } from "bun:test";
import { z } from "zod/v4";
import {
  MissingOutputError,
  agentTool,
  createBareAgent,
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
