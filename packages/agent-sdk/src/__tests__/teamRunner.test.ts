import { describe, expect, test } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  agentTool,
  createAgent,
  createMemoryMailbox,
  createTeamRunner,
  delegateTool,
  type ContextTraceEvent,
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
  test("agentTool handoff returns an acceptance receipt without running the target", async () => {
    let engineeringCalls = 0;
    const engineeringAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          engineeringCalls++;
          return textAssistant("Engineering should not finish synchronously");
        },
      },
    });
    let rootCalls = 0;
    const rootAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          rootCalls++;
          if (rootCalls === 1) {
            return toolUseAssistant("toolu_1", "engineering", {
              mode: "handoff",
              task: "Implement a long-running feature",
            });
          }
          const result = String((messages.at(-1)?.content as Array<{ content?: string }> | undefined)?.[0]?.content ?? "");
          expect(result).toContain('"status": "accepted"');
          expect(result).toContain('"to": "engineering"');
          return textAssistant("CEO accepted the handoff");
        },
      },
      tools: [
        agentTool("engineering", engineeringAgent, {
          description: "Hand work to engineering.",
        }),
      ],
    });
    const mailbox = createMemoryMailbox();
    const runner = createTeamRunner({ root: rootAgent, mailbox });

    const messages = await collect(runner.query("Hand work to engineering."));
    const engineeringInbox = await mailbox.inbox("engineering", { status: "all" });

    expect(engineeringCalls).toBe(0);
    expect(engineeringInbox).toHaveLength(1);
    expect(engineeringInbox[0]).toMatchObject({
      from: "manager",
      to: "engineering",
      content: "Implement a long-running feature",
      status: "pending",
      workItemRole: "delegation",
    });
    expect(messages.some(message => message.type === "team_message" && message.subtype === "sent")).toBe(true);
    expect(messages.find(message => message.type === "result")).toMatchObject({
      type: "result",
      result: "CEO accepted the handoff",
    });
  });

  test("agentTool passes authorized workspace grants to delegated agents", async () => {
    const sharedRoot = join(tmpdir(), "agent-sdk-shared-root");
    const backendRoot = join(sharedRoot, "backend");
    let childSawGrant = false;
    const backendAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ systemPrompt, messages }) {
          childSawGrant = true;
          expect(systemPrompt).toContain("Runtime workspace access for this task");
          expect(systemPrompt).toContain(backendRoot);
          expect(systemPrompt).toContain("Read-only tools may inspect any path");
          expect(String(messages.at(0)?.content ?? "")).toContain("Workspace access granted for this delegated task");
          expect(String(messages.at(0)?.content ?? "")).toContain("reason: backend implementation");
          return textAssistant("Backend done");
        },
      },
    });

    let rootCalls = 0;
    const rootAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          rootCalls++;
          if (rootCalls === 1) {
            return toolUseAssistant("toolu_1", "backend", {
              mode: "ask",
              task: "Implement the backend in the shared workspace.",
              workspaceGrants: [{
                root: backendRoot,
                access: ["write"],
                reason: "backend implementation",
              }],
            });
          }
          const result = String((messages.at(-1)?.content as Array<{ content?: string }> | undefined)?.[0]?.content ?? "");
          expect(result).toContain("Backend done");
          return textAssistant("Root final");
        },
      },
      tools: [
        agentTool("backend", backendAgent, {
          description: "Delegate backend implementation.",
        }),
      ],
    });
    const mailbox = createMemoryMailbox();
    const runner = createTeamRunner({ root: rootAgent, mailbox });

    const messages = await collect(runner.query("Use backend.", {
      permissions: {
        workspaceGrants: [{
          root: sharedRoot,
          access: ["write"],
          reason: "project shared workspace",
        }],
      },
    }));
    const backendInbox = await mailbox.inbox("backend", { status: "all" });

    expect(childSawGrant).toBe(true);
    expect(backendInbox[0]?.metadata?.workspaceGrants).toEqual([
      expect.objectContaining({
        root: backendRoot,
        access: ["write"],
        reason: "backend implementation",
      }),
    ]);
    expect(messages.find(message => message.type === "result")).toMatchObject({
      type: "result",
      result: "Root final",
    });
  });

  test("agentTool denies workspace grants the caller cannot delegate", async () => {
    const sharedRoot = join(tmpdir(), "agent-sdk-unowned-shared-root");
    let backendCalls = 0;
    const backendAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          backendCalls++;
          return textAssistant("Backend should not run");
        },
      },
    });

    let rootCalls = 0;
    const rootAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          rootCalls++;
          if (rootCalls === 1) {
            return toolUseAssistant("toolu_1", "backend", {
              mode: "ask",
              task: "Use an unowned shared workspace.",
              workspaceGrants: [{
                root: sharedRoot,
                access: ["write"],
                reason: "unowned grant",
              }],
            });
          }
          const result = String((messages.at(-1)?.content as Array<{ content?: string }> | undefined)?.[0]?.content ?? "");
          expect(result).toContain('"status": "permission_denied"');
          expect(result).toContain('"tool": "backend"');
          expect(result).toContain('"allowedWriteRoots"');
          expect(result).toContain("shared workspace grant");
          return textAssistant("Root saw the permission denial");
        },
      },
      tools: [
        agentTool("backend", backendAgent, {
          description: "Delegate backend implementation.",
        }),
      ],
    });
    const runner = createTeamRunner({ root: rootAgent, mailbox: createMemoryMailbox() });

    const messages = await collect(runner.query("Use backend."));

    expect(backendCalls).toBe(0);
    expect(messages.find(message => message.type === "result")).toMatchObject({
      type: "result",
      result: "Root saw the permission denial",
    });
  });

  test("agentTool reports runtime depth limits back to the calling AgentLike", async () => {
    let leafCalls = 0;
    const leafAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          leafCalls++;
          return textAssistant("Leaf should not run past the depth boundary");
        },
      },
    });

    let childCalls = 0;
    let childSawDepthError = false;
    const childAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          childCalls++;
          if (childCalls === 1) {
            return toolUseAssistant("toolu_child", "leaf", {
              mode: "ask",
              task: "Ask leaf for implementation details.",
            });
          }

          const content = messages.at(-1)?.content;
          const toolResult = Array.isArray(content)
            ? content.find(block => block.type === "tool_result")
            : undefined;
          expect(String(toolResult?.content ?? "")).toContain("Reached maximum delegate depth (1)");
          childSawDepthError = true;
          return textAssistant("Child adjusted after seeing the runtime boundary");
        },
      },
      tools: [
        agentTool("leaf", leafAgent, {
          description: "Ask leaf for implementation details.",
        }),
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
            return toolUseAssistant("toolu_root", "child", {
              mode: "ask",
              task: "Ask child to coordinate implementation.",
            });
          }
          return textAssistant("Root final");
        },
      },
      tools: [
        agentTool("child", childAgent, {
          description: "Ask child to coordinate implementation.",
        }),
      ],
    });

    const runner = createTeamRunner({
      root: rootAgent,
      mailbox: createMemoryMailbox(),
      maxDelegateDepth: 1,
    });

    const messages = await collect(runner.query("Use child."));

    expect(leafCalls).toBe(0);
    expect(childSawDepthError).toBe(true);
    expect(messages.find(message => message.type === "result")).toMatchObject({
      type: "result",
      result: "Root final",
    });
  });

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

  test("passes query tracer context to delegated agents with member source", async () => {
    const trace: ContextTraceEvent[] = [];
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
    const rootAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          rootCalls++;
          if (rootCalls === 1) {
            return toolUseAssistant("toolu_1", "engineering", {
              task: "Design the trace store",
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

    await collect(runner.query("Use engineering.", {
      tracer: {
        onEvent(event) {
          trace.push(event);
        },
      },
    }));

    expect(trace.some(entry =>
      entry.type === "result" &&
      entry.data.result === "Engineering result" &&
      entry.source.kind === "team_member" &&
      entry.source.member === "engineering" &&
      entry.source.mailbox === "engineering"
    )).toBe(true);
    expect(trace.some(entry =>
      entry.type === "tool_use" &&
      entry.source.kind === "root" &&
      entry.data.name === "engineering"
    )).toBe(true);
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
