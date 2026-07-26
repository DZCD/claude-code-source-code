import { describe, expect, test } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod/v4";
import {
  AbortError,
  agentTool,
  createAgent,
  createMemoryMailbox,
  createTeam,
  createTeamRunner,
  delegateTool,
  teamMember,
  tool,
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

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function createLeadThatHandoffs(memberNames: string[]) {
  let calls = 0;
  return createAgent({
    apiKey: "test-key",
    model: "claude-test",
    modelClient: {
      async createMessage() {
        calls++;
        if (calls === 1) {
          return {
            role: "assistant" as const,
            content: memberNames.map((name, index) => ({
              type: "tool_use" as const,
              id: `toolu_${index}`,
              name,
              input: { mode: "handoff", task: `Run ${name}` },
            })),
          };
        }
        return textAssistant("Lead completed after all reports");
      },
    },
  });
}

async function collect(iterable: AsyncIterable<TeamRunnerMessage>): Promise<TeamRunnerMessage[]> {
  const messages: TeamRunnerMessage[] = [];
  for await (const message of iterable) {
    messages.push(message);
  }
  return messages;
}

describe("team runner", () => {
  test("rejects invalid work-item concurrency limits", () => {
    const root = createLeadThatHandoffs([]);
    expect(() => createTeamRunner({ root, maxConcurrentWorkItems: 0 })).toThrow(
      "TeamRunnerOptions.maxConcurrentWorkItems must be a positive integer",
    );
    expect(() => createTeamRunner({ root, maxConcurrentWorkItems: 1.5 })).toThrow(
      "TeamRunnerOptions.maxConcurrentWorkItems must be a positive integer",
    );
  });

  test("rejects a conflicting tool batch before any tool runs and lets the lead split the work", async () => {
    let incrementCalls = 0;
    let memberCalls = 0;
    let memberSawRevisionTwo = false;
    let firstPolicyKinds: string[] = [];
    const member = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          memberCalls++;
          memberSawRevisionTwo = String(messages.at(-1)?.content ?? "").includes("revision 2");
          return textAssistant("Member processed revision 2");
        },
      },
    });
    let leadCalls = 0;
    const lead = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      tools: [
        tool(
          "incrementRevision",
          "Increment revision.",
          z.object({ revision: z.number() }),
          async ({ revision }) => {
            incrementCalls++;
            return { content: JSON.stringify({ revision: revision + 1 }) };
          },
        ),
      ],
      toolBatchPolicy: {
        validate({ toolCalls }) {
          if (leadCalls === 1) firstPolicyKinds = toolCalls.map(call => call.kind);
          const increment = toolCalls.find(call => call.name === "incrementRevision");
          const handoff = toolCalls.find(call => call.name === "worker");
          if (increment && handoff) {
            return {
              allowed: false as const,
              code: "invalid_tool_batch",
              message: "Update the revision before delegating work that depends on it.",
              conflictingToolCallIds: [increment.id, handoff.id],
              suggestedNextStep: "Run incrementRevision first, then hand off the new revision.",
            };
          }
          return { allowed: true as const };
        },
      },
      modelClient: {
        async createMessage({ messages }) {
          leadCalls++;
          if (leadCalls === 1) {
            return {
              role: "assistant" as const,
              content: [
                { type: "tool_use" as const, id: "tool_increment_stale", name: "incrementRevision", input: { revision: 1 } },
                { type: "tool_use" as const, id: "tool_handoff_stale", name: "worker", input: { mode: "handoff", task: "Process revision 1" } },
              ],
            };
          }
          if (leadCalls === 2) {
            const blocks = messages.at(-1)?.content;
            expect(Array.isArray(blocks) ? blocks : []).toHaveLength(2);
            for (const block of Array.isArray(blocks) ? blocks : []) {
              expect(block).toMatchObject({ type: "tool_result", is_error: true });
              expect(String("content" in block ? block.content : "")).toContain("invalid_tool_batch");
            }
            expect(incrementCalls).toBe(0);
            expect(memberCalls).toBe(0);
            return toolUseAssistant("tool_increment_fresh", "incrementRevision", { revision: 1 });
          }
          if (leadCalls === 3) {
            const blocks = messages.at(-1)?.content;
            const revisionResult = Array.isArray(blocks) ? blocks[0] : undefined;
            expect(String(revisionResult && "content" in revisionResult ? revisionResult.content : ""))
              .toContain('"revision":2');
            return toolUseAssistant("tool_handoff_fresh", "worker", {
              mode: "handoff",
              task: "Process revision 2",
            });
          }
          return textAssistant("Lead completed after revision 2 report");
        },
      },
    });
    const team = createTeam({
      name: "batch-policy",
      lead,
      members: [teamMember({ name: "worker", role: "executor", agent: member })],
    });

    const result = await team.prompt("Update the revision and delegate processing.");
    const workerInbox = await team.mailbox.inbox("batch-policy::worker", { status: "all" });

    expect(firstPolicyKinds).toEqual(["tool", "agent_tool"]);
    expect(incrementCalls).toBe(1);
    expect(memberCalls).toBe(1);
    expect(memberSawRevisionTwo).toBe(true);
    expect(workerInbox).toHaveLength(1);
    expect(workerInbox[0]?.content).toBe("Process revision 2");
    expect(result.result).toBe("Lead completed after revision 2 report");
  });

  test("does not call the lead again until accepted handoff work reports back", async () => {
    const memberStarted = deferred();
    const releaseMember = deferred();
    let memberCompleted = false;
    const member = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          memberStarted.resolve();
          await releaseMember.promise;
          memberCompleted = true;
          return textAssistant("Member completed the delegated work");
        },
      },
    });
    let leadCalls = 0;
    let leadSawCompletionReport = false;
    const lead = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          leadCalls++;
          if (leadCalls === 1) {
            return toolUseAssistant("toolu_member", "member", {
              mode: "handoff",
              task: "Complete the delegated work",
            });
          }
          const update = String(messages.at(-1)?.content ?? "");
          leadSawCompletionReport = memberCompleted &&
            update.includes("Team runtime update") &&
            update.includes("Member completed the delegated work");
          return textAssistant("Lead continued after the member report");
        },
      },
    });
    const team = createTeam({
      name: "handoff-pause",
      lead,
      members: [teamMember({ name: "member", role: "executor", agent: member })],
    });

    const run = team.prompt("Delegate the work.");
    await memberStarted.promise;
    await Promise.resolve();
    expect(leadCalls).toBe(1);
    releaseMember.resolve();
    const result = await run;

    expect(leadCalls).toBe(2);
    expect(leadSawCompletionReport).toBe(true);
    expect(result.result).toBe("Lead continued after the member report");
  });

  test("keeps accepted handoffs serial by default", async () => {
    const releaseA = deferred();
    const startedA = deferred();
    let workerBStarted = false;
    const workerA = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          startedA.resolve();
          await releaseA.promise;
          return textAssistant("A done");
        },
      },
    });
    const workerB = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          workerBStarted = true;
          return textAssistant("B done");
        },
      },
    });
    const team = createTeam({
      name: "default-serial",
      lead: createLeadThatHandoffs(["worker_a", "worker_b"]),
      members: [
        teamMember({ name: "worker_a", role: "executor", agent: workerA }),
        teamMember({ name: "worker_b", role: "executor", agent: workerB }),
      ],
    });

    const run = team.prompt("Run both.");
    await startedA.promise;
    await Promise.resolve();
    expect(workerBStarted).toBe(false);
    releaseA.resolve();
    await run;
    expect(workerBStarted).toBe(true);
  });

  test("runs different member mailboxes concurrently up to the configured limit", async () => {
    const release = deferred();
    const twoStarted = deferred();
    let inFlight = 0;
    let maxInFlight = 0;
    let started = 0;
    const makeWorker = (name: string) => createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          started++;
          inFlight++;
          maxInFlight = Math.max(maxInFlight, inFlight);
          if (started === 2) twoStarted.resolve();
          await release.promise;
          inFlight--;
          return textAssistant(`${name} done`);
        },
      },
    });
    const team = createTeam({
      name: "bounded-concurrency",
      lead: createLeadThatHandoffs(["worker_a", "worker_b", "worker_c"]),
      members: [
        teamMember({ name: "worker_a", role: "executor", agent: makeWorker("A") }),
        teamMember({ name: "worker_b", role: "executor", agent: makeWorker("B") }),
        teamMember({ name: "worker_c", role: "executor", agent: makeWorker("C") }),
      ],
      runner: { maxConcurrentWorkItems: 2 },
    });

    const run = team.prompt("Run all three.");
    await twoStarted.promise;
    await Promise.resolve();
    expect(started).toBe(2);
    expect(maxInFlight).toBe(2);
    release.resolve();
    await run;
    expect(started).toBe(3);
    expect(maxInFlight).toBe(2);
  });

  test("starts all independent members when the concurrency limit allows it", async () => {
    const release = deferred();
    const allStarted = deferred();
    const started = new Set<string>();
    const makeWorker = (name: string) => createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          started.add(name);
          if (started.size === 3) allStarted.resolve();
          await release.promise;
          return textAssistant(`${name} done`);
        },
      },
    });
    const team = createTeam({
      name: "full-concurrency",
      lead: createLeadThatHandoffs(["worker_a", "worker_b", "worker_c"]),
      members: [
        teamMember({ name: "worker_a", role: "executor", agent: makeWorker("A") }),
        teamMember({ name: "worker_b", role: "executor", agent: makeWorker("B") }),
        teamMember({ name: "worker_c", role: "executor", agent: makeWorker("C") }),
      ],
      runner: { maxConcurrentWorkItems: 3 },
    });

    const run = team.prompt("Run all three.");
    await allStarted.promise;
    expect([...started].sort()).toEqual(["A", "B", "C"]);
    release.resolve();
    await run;
  });

  test("serializes multiple concurrent handoffs addressed to the same member mailbox", async () => {
    const releaseFirst = deferred();
    const firstStarted = deferred();
    let calls = 0;
    const worker = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          calls++;
          if (calls === 1) {
            firstStarted.resolve();
            await releaseFirst.promise;
          }
          return textAssistant(`worker call ${calls} done`);
        },
      },
    });
    const team = createTeam({
      name: "same-member-serial",
      lead: createLeadThatHandoffs(["worker", "worker"]),
      members: [teamMember({ name: "worker", role: "executor", agent: worker })],
      runner: { maxConcurrentWorkItems: 2 },
    });

    const run = team.prompt("Send two tasks to the same worker.");
    await firstStarted.promise;
    await Promise.resolve();
    expect(calls).toBe(1);
    releaseFirst.resolve();
    await run;
    expect(calls).toBe(2);
  });

  test("isolates a failed handoff so other accepted work completes and the lead can decide", async () => {
    const noopTool = tool(
      "noop",
      "No-op tool used to exhaust the worker turn limit.",
      z.object({}),
      async () => ({ content: "ok" }),
    );
    const workerA = createAgent({
      apiKey: "test-key",
      name: "worker-a",
      model: "claude-test",
      maxTurns: 1,
      tools: [noopTool],
      modelClient: {
        async createMessage() {
          return toolUseAssistant("toolu_worker_a", "noop", {});
        },
      },
    });
    let workerBCalls = 0;
    const workerB = createAgent({
      apiKey: "test-key",
      name: "worker-b",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          workerBCalls++;
          return textAssistant("worker-b completed");
        },
      },
    });

    let leadCalls = 0;
    let leadSawFailure = false;
    let leadSawSuccess = false;
    const lead = createAgent({
      apiKey: "test-key",
      name: "lead",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          leadCalls++;
          if (leadCalls === 1) {
            return {
              role: "assistant" as const,
              content: [
                { type: "tool_use" as const, id: "toolu_a", name: "worker_a", input: { mode: "handoff", task: "Run task A" } },
                { type: "tool_use" as const, id: "toolu_b", name: "worker_b", input: { mode: "handoff", task: "Run task B" } },
              ],
            };
          }
          const update = String(messages.at(-1)?.content ?? "");
          leadSawFailure = update.includes("max_turns_exceeded") && update.includes("worker_a failed");
          leadSawSuccess = update.includes("worker-b completed");
          return textAssistant("Lead accepted the partial result and finished");
        },
      },
    });
    const team = createTeam({
      name: "failure-isolation",
      lead,
      members: [
        teamMember({ name: "worker_a", role: "executor", agent: workerA }),
        teamMember({ name: "worker_b", role: "executor", agent: workerB }),
      ],
      runner: { maxConcurrentWorkItems: 2 },
    });

    const messages = await collect(team.query("Run both independent tasks."));
    const workerAInbox = await team.mailbox.inbox("failure-isolation::worker_a", { status: "all" });
    const workerBInbox = await team.mailbox.inbox("failure-isolation::worker_b", { status: "all" });
    const managerInbox = await team.mailbox.inbox("manager", { status: "all" });

    expect(workerAInbox[0]?.status).toBe("failed");
    expect(workerBInbox[0]?.status).toBe("done");
    expect(workerBCalls).toBe(1);
    expect(leadSawFailure).toBe(true);
    expect(leadSawSuccess).toBe(true);
    expect(managerInbox).toHaveLength(2);
    expect(managerInbox.find(message => message.metadata?.status === "failed")?.metadata?.error).toMatchObject({
      code: "max_turns_exceeded",
    });
    expect(messages.some(message =>
      message.type === "team_message" &&
      message.subtype === "failed" &&
      message.error?.code === "max_turns_exceeded"
    )).toBe(true);
    expect(messages.find(message => message.type === "result")).toMatchObject({
      type: "result",
      result: "Lead accepted the partial result and finished",
      is_error: false,
    });
  });

  test("global abort cancels the current and remaining accepted work", async () => {
    const abortingWorker = {
      async *query() {
        yield {
          type: "result" as const,
          subtype: "error_abort" as const,
          is_error: true as const,
          result: "",
          session_id: "aborted-worker",
          num_turns: 0,
          error: new AbortError("Team run aborted"),
        };
      },
      async prompt(): Promise<never> {
        throw new AbortError("Team run aborted");
      },
    };
    let workerBCalls = 0;
    const workerB = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ signal }) {
          workerBCalls++;
          await new Promise<never>((_, reject) => {
            if (signal?.aborted) {
              reject(new AbortError("Team run aborted"));
              return;
            }
            signal?.addEventListener("abort", () => reject(new AbortError("Team run aborted")), { once: true });
          });
        },
      },
    });
    let leadCalls = 0;
    const lead = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          leadCalls++;
          if (leadCalls === 1) {
            return {
              role: "assistant" as const,
              content: [
                { type: "tool_use" as const, id: "toolu_a", name: "worker_a", input: { mode: "handoff", task: "Run A" } },
                { type: "tool_use" as const, id: "toolu_b", name: "worker_b", input: { mode: "handoff", task: "Run B" } },
              ],
            };
          }
          return textAssistant("Both accepted");
        },
      },
    });
    const team = createTeam({
      name: "abort-cleanup",
      lead,
      members: [
        teamMember({ name: "worker_a", role: "executor", agent: abortingWorker }),
        teamMember({ name: "worker_b", role: "executor", agent: workerB }),
      ],
      runner: { maxConcurrentWorkItems: 2 },
    });

    await expect(team.prompt("Run both.")).rejects.toBeInstanceOf(AbortError);
    const workerAInbox = await team.mailbox.inbox("abort-cleanup::worker_a", { status: "all" });
    const workerBInbox = await team.mailbox.inbox("abort-cleanup::worker_b", { status: "all" });

    expect(workerBCalls).toBe(1);
    expect(workerAInbox[0]?.status).toBe("cancelled");
    expect(workerBInbox[0]?.status).toBe("cancelled");
  });

  test("agentTool handoff returns a receipt to the caller and runner waits for final delivery", async () => {
    let engineeringCalls = 0;
    const engineeringAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          engineeringCalls++;
          return textAssistant("Engineering completed the implementation");
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
          if (rootCalls === 2) {
            const update = String(messages.at(-1)?.content ?? "");
            expect(update).toContain("Team runtime update");
            expect(update).toContain("Engineering completed the implementation");
            return textAssistant("CEO final delivery with engineering results");
          }
          throw new Error(`Unexpected root call ${rootCalls}`);
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

    expect(engineeringCalls).toBe(1);
    expect(rootCalls).toBe(2);
    expect(engineeringInbox).toHaveLength(1);
    expect(engineeringInbox[0]).toMatchObject({
      from: "manager",
      to: "engineering",
      content: "Implement a long-running feature",
      status: "done",
      workItemRole: "delegation",
    });
    expect(messages.some(message => message.type === "team_message" && message.subtype === "sent")).toBe(true);
    expect(messages.some(message => message.type === "team_message" && message.subtype === "replied")).toBe(true);
    expect(messages.some(message =>
      message.type === "user" &&
      String(message.tool_use_result).includes('"phase": "queued"') &&
      String(message.tool_use_result).includes('"completion_pending": true') &&
      String(message.tool_use_result).includes('"work_item_id"')
    )).toBe(true);
    expect(messages.some(message =>
      message.type === "agent_message" &&
      message.source.kind === "root" &&
      message.message.type === "result" &&
      message.message.result.includes("Delegated work was queued")
    )).toBe(true);
    expect(messages.find(message => message.type === "result")).toMatchObject({
      type: "result",
      result: "CEO final delivery with engineering results",
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

  test("traces one team invocation as one root with lead and member runs beneath it", async () => {
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
              mode: "handoff",
              task: "Design the trace store",
            });
          }
          return textAssistant("CEO final");
        },
      },
      tools: [
        agentTool("engineering", engineeringAgent, {
          description: "Delegate engineering work",
        }),
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

    const runStarts = trace.filter(entry => entry.type === "run_start");
    const invocationRoot = runStarts.find(entry => entry.data.runtime === "team");
    expect(invocationRoot).toBeDefined();
    expect(invocationRoot?.parent_run_id).toBeUndefined();
    expect(runStarts).toHaveLength(4);
    expect(runStarts.filter(entry => entry.parent_run_id === undefined)).toEqual([
      invocationRoot!,
    ]);
    expect(runStarts.filter(entry => entry !== invocationRoot).every(entry =>
      entry.parent_run_id === invocationRoot?.run_id
    )).toBe(true);
    expect(new Set(trace.map(entry => entry.session_id))).toEqual(new Set([
      invocationRoot!.session_id,
    ]));
    expect(new Set(runStarts
      .filter(entry => entry !== invocationRoot)
      .map(entry => entry.data.agent_session_id)).size).toBe(2);
    expect(trace.find(entry =>
      entry.type === "result" &&
      entry.run_id === invocationRoot?.run_id
    )).toMatchObject({
      data: {
        subtype: "success",
        result: "CEO final",
      },
    });

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
