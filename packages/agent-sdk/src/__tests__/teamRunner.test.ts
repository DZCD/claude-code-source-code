import { describe, expect, test } from "bun:test";
import {
  createAgent,
  createMemoryMailbox,
  createTeamRunner,
  delegateTool,
  type ModelClient,
  type TeamRunnerMessage,
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

async function collect(iterable: AsyncIterable<TeamRunnerMessage>): Promise<TeamRunnerMessage[]> {
  const messages: TeamRunnerMessage[] = [];
  for await (const message of iterable) {
    messages.push(message);
  }
  return messages;
}

describe("team runner", () => {
  test("delegateTool runs another AgentLike through the runner mailbox", async () => {
    const engineeringAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          return textAssistant("Engineering result");
        },
      },
    });
    let rootCalls = 0;
    const rootClient: ModelClient = {
      async createMessage() {
        rootCalls++;
        if (rootCalls === 1) {
          return toolUseAssistant("toolu_1", "engineering", {
            task: "Design the API",
          });
        }
        return textAssistant("CEO final");
      },
    };
    const rootAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: rootClient,
      tools: [
        delegateTool("engineering", "Delegate engineering work", engineeringAgent),
      ],
    });
    const mailbox = createMemoryMailbox();
    const runner = createTeamRunner({ root: rootAgent, mailbox });

    const messages = await collect(runner.query("Use engineering."));
    const inbox = await mailbox.inbox("manager", { status: "all" });

    expect(messages.some(message => message.type === "team_message" && message.subtype === "sent")).toBe(true);
    expect(messages.some(message => message.type === "team_message" && message.subtype === "replied")).toBe(true);
    expect(messages.some(message =>
      message.type === "agent_message" &&
      message.source.member === "engineering" &&
      message.message.type === "result" &&
      message.message.result === "Engineering result"
    )).toBe(true);
    expect(messages.some(message =>
      message.type === "user" &&
      message.tool_use_result === "Engineering result"
    )).toBe(true);
    expect(messages.find(message => message.type === "result")).toMatchObject({
      type: "result",
      result: "CEO final",
    });
    expect(inbox[0]).toMatchObject({
      from: "engineering",
      to: "manager",
      content: "Engineering result",
      workItemRole: "upstream_report",
    });
  });

  test("delegated AgentLike can delegate to another AgentLike", async () => {
    const backendAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          return textAssistant("Backend result");
        },
      },
    });
    let engineeringCalls = 0;
    const engineeringAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          engineeringCalls++;
          if (engineeringCalls === 1) {
            return toolUseAssistant("toolu_2", "backend", {
              task: "Implement storage",
            });
          }
          return textAssistant("Engineering final with backend");
        },
      },
      tools: [
        delegateTool("backend", "Delegate backend work", backendAgent),
      ],
    });
    let rootCalls = 0;
    const rootAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          rootCalls++;
          if (rootCalls === 1) {
            return toolUseAssistant("toolu_1", "engineering", {
              task: "Build the feature",
            });
          }
          return textAssistant("CEO final");
        },
      },
      tools: [
        delegateTool("engineering", "Delegate engineering work", engineeringAgent),
      ],
    });
    const runner = createTeamRunner({
      root: rootAgent,
      mailbox: createMemoryMailbox(),
    });

    const messages = await collect(runner.query("Use engineering."));

    expect(messages.some(message =>
      message.type === "agent_message" &&
      message.source.member === "backend" &&
      message.message.type === "result" &&
      message.message.result === "Backend result"
    )).toBe(true);
    expect(messages.some(message =>
      message.type === "user" &&
      message.tool_use_result === "Engineering final with backend"
    )).toBe(true);
    expect(messages.find(message => message.type === "result")).toMatchObject({
      type: "result",
      result: "CEO final",
    });
  });
});
