import { describe, expect, test } from "bun:test";
import {
  createAgent,
  createMultiAgent,
  createSupervisor,
  createSubAgent,
  type ModelClient,
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

async function collect(iterable: AsyncIterable<SDKMessage>): Promise<SDKMessage[]> {
  const messages: SDKMessage[] = [];
  for await (const message of iterable) {
    messages.push(message);
  }
  return messages;
}

describe("supervisor delegation", () => {
  test("keeps createMultiAgent as a compatibility alias", () => {
    expect(createMultiAgent).toBe(createSupervisor);
  });

  test("creates delegate tools for sub-agents", async () => {
    const subAgent = createSubAgent({
      name: "researcher",
      description: "Research a topic",
      agent: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: {
          async createMessage() {
            return textAssistant("sub-agent answer");
          },
        },
      }),
    });

    const supervisor = createSupervisor({
      supervisor: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: {
          async createMessage() {
            return textAssistant("done");
          },
        },
      }),
      subAgents: [subAgent],
    });

    expect(supervisor.tools.map(tool => tool.name)).toEqual(["delegate_researcher"]);
  });

  test("supervisor can delegate work to a sub-agent", async () => {
    const subAgent = createSubAgent({
      name: "researcher",
      description: "Research a topic",
      agent: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: {
          async createMessage() {
            return textAssistant("researched answer");
          },
        },
      }),
    });
    const supervisorCalls: unknown[] = [];
    const supervisorClient: ModelClient = {
      async createMessage({ messages }) {
        supervisorCalls.push(messages);
        if (supervisorCalls.length === 1) {
          return toolUseAssistant("toolu_1", "delegate_researcher", {
            task: "Research SDK architecture",
          });
        }
        return textAssistant("final answer");
      },
    };
    const supervisor = createSupervisor({
      supervisor: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: supervisorClient,
      }),
      subAgents: [subAgent],
    });

    const messages = await collect(supervisor.query("Use the researcher."));

    expect(messages.map(message => message.type)).toEqual([
      "system",
      "assistant",
      "user",
      "assistant",
      "result",
    ]);
    expect(messages[2]).toMatchObject({
      type: "user",
      tool_use_result: "researched answer",
    });
    expect(messages.at(-1)).toMatchObject({
      type: "result",
      result: "final answer",
    });
  });
});
