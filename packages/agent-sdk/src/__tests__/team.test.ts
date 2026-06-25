import { describe, expect, test } from "bun:test";
import {
  createAgent,
  createMemoryMailbox,
  createTeam,
  teamMember,
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

describe("team", () => {
  test("claimNext only claims messages for the addressed member", async () => {
    const mailbox = createMemoryMailbox();
    await mailbox.send("manager", "engineering::researcher", "Research this.");
    await mailbox.send("manager", "engineering::tester", "Test this.");

    const claimed = await mailbox.claimNext("engineering::researcher");
    const researcherInbox = await mailbox.inbox("engineering::researcher", { status: "all" });
    const testerInbox = await mailbox.inbox("engineering::tester", { status: "all" });

    expect(claimed).toMatchObject({
      to: "engineering::researcher",
      status: "processing",
    });
    expect(researcherInbox[0]).toMatchObject({ status: "processing" });
    expect(testerInbox[0]).toMatchObject({ status: "pending" });
  });

  test("supervisor can send work to a team member mailbox", async () => {
    let supervisorCalls = 0;
    const supervisorClient: ModelClient = {
      async createMessage() {
        supervisorCalls++;
        if (supervisorCalls === 1) {
          return toolUseAssistant("toolu_1", "team_send", {
            to: "researcher",
            content: "Research the team runtime design.",
          });
        }
        return textAssistant("delegated");
      },
    };

    const team = createTeam({
      name: "engineering",
      supervisor: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: supervisorClient,
      }),
      members: [
        teamMember({
          name: "researcher",
          role: "executor",
          focus: "Research agent architecture",
          agent: createAgent({
            apiKey: "test-key",
            model: "claude-test",
            modelClient: { async createMessage() { return textAssistant("idle"); } },
          }),
        }),
      ],
      mailbox: createMemoryMailbox(),
    });

    const events = await collect(team.query("Ask the researcher to inspect this."));
    const inbox = await team.mailbox.inbox("engineering::researcher");

    expect(events.at(-1)).toMatchObject({ type: "result", result: "delegated" });
    expect(team.tools.map(tool => tool.name)).toContain("team_send");
    expect(inbox).toHaveLength(1);
    expect(inbox[0]).toMatchObject({
      from: "manager",
      to: "engineering::researcher",
      content: "Research the team runtime design.",
      status: "pending",
      workItemRole: "upstream_request",
    });
    expect(inbox[0]?.workItemId).toBe(inbox[0]?.id);
  });

  test("member can read and reply while preserving work item context", async () => {
    const mailbox = createMemoryMailbox();
    const team = createTeam({
      name: "engineering",
      supervisor: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("supervisor idle"); } },
      }),
      members: [
        teamMember({
          name: "researcher",
          role: "executor",
          focus: "Research agent architecture",
          agent: createAgent({
            apiKey: "test-key",
            model: "claude-test",
            modelClient: {
              async createMessage({ messages }) {
                const last = messages.at(-1);
                if (last?.role === "user" && typeof last.content === "string") {
                  return toolUseAssistant("toolu_1", "team_read", { message_id: "first" });
                }
                return toolUseAssistant("toolu_2", "team_reply", {
                  message_id: "first",
                  content: "Research complete.",
                });
              },
            },
            maxTurns: 3,
          }),
        }),
      ],
      mailbox,
    });
    const sent = await team.send("manager", "researcher", "Please research the SDK.");
    await mailbox.updateStatus(sent.id, "pending");

    const member = team.members[0]!;
    await member.agent.prompt("Handle your team inbox.");

    const supervisorInbox = await mailbox.inbox("manager");
    const original = await mailbox.get(sent.id);

    expect(original?.status).toBe("done");
    expect(supervisorInbox).toHaveLength(1);
    expect(supervisorInbox[0]).toMatchObject({
      from: "engineering::researcher",
      to: "manager",
      content: "Research complete.",
      threadId: sent.threadId,
      parentMessageId: sent.id,
      workItemId: sent.workItemId,
      workItemRole: "upstream_report",
      upstreamMessageId: sent.id,
    });
  });

  test("drain runs the addressed member and preserves explicit routing", async () => {
    const mailbox = createMemoryMailbox();
    let researcherCalls = 0;
    let testerCalls = 0;
    const team = createTeam({
      name: "engineering",
      supervisor: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("supervisor idle"); } },
      }),
      members: [
        teamMember({
          name: "researcher",
          role: "executor",
          agent: createAgent({
            apiKey: "test-key",
            model: "claude-test",
            modelClient: {
              async createMessage({ messages }) {
                researcherCalls++;
                if (researcherCalls > 1) {
                  return textAssistant("reported upstream");
                }
                const text = String(messages.at(-1)?.content ?? "");
                expect(text).toContain("message_id: first");
                return toolUseAssistant("toolu_1", "team_reply", {
                  message_id: "first",
                  content: "Research handled.",
                });
              },
            },
          }),
        }),
        teamMember({
          name: "tester",
          role: "executor",
          agent: createAgent({
            apiKey: "test-key",
            model: "claude-test",
            modelClient: {
              async createMessage() {
                testerCalls++;
                return textAssistant("tester should not run");
              },
            },
          }),
        }),
      ],
      mailbox,
    });
    await team.send("manager", "researcher", "Only researcher should handle this.");

    const result = await team.drain();
    const supervisorInbox = await mailbox.inbox("manager");

    expect(result).toMatchObject({
      processed: 1,
      failed: 0,
    });
    expect(researcherCalls).toBe(2);
    expect(testerCalls).toBe(0);
    expect(supervisorInbox[0]).toMatchObject({
      from: "engineering::researcher",
      to: "manager",
      content: "Research handled.",
      workItemRole: "upstream_report",
    });
  });

  test("drain marks a message failed and reports upstream when member ends without feedback", async () => {
    const mailbox = createMemoryMailbox();
    const team = createTeam({
      name: "engineering",
      supervisor: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("supervisor idle"); } },
      }),
      members: [
        teamMember({
          name: "researcher",
          role: "executor",
          agent: createAgent({
            apiKey: "test-key",
            model: "claude-test",
            modelClient: { async createMessage() { return textAssistant("I handled it in plain text."); } },
          }),
        }),
      ],
      mailbox,
    });
    const sent = await team.send("manager", "researcher", "Please reply through mailbox.");

    const result = await team.drain();
    const original = await mailbox.get(sent.id);
    const supervisorInbox = await mailbox.inbox("manager");

    expect(result.failed).toBe(1);
    expect(original?.status).toBe("failed");
    expect(supervisorInbox[0]).toMatchObject({
      from: "engineering::researcher",
      to: "manager",
      workItemRole: "followup",
    });
    expect(supervisorInbox[0]?.content).toContain("ended without team_reply or team_followup");
  });
});
