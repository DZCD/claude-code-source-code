import { describe, expect, test } from "bun:test";
import {
  createAgent,
  createMemoryMailbox,
  createTeam,
  teamMember,
  type AgentLikeEvent,
  type ModelClient,
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

async function collect(iterable: AsyncIterable<AgentLikeEvent>): Promise<AgentLikeEvent[]> {
  const messages: AgentLikeEvent[] = [];
  for await (const message of iterable) {
    messages.push(message);
  }
  return messages;
}

describe("team", () => {
  test("lead tools default to member AgentLike tools without raw mailbox controls", async () => {
    const team = createTeam({
      name: "engineering",
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("lead idle"); } },
      }),
      members: [
        teamMember({
          name: "researcher",
          role: "executor",
          agent: createAgent({
            apiKey: "test-key",
            model: "claude-test",
            modelClient: { async createMessage() { return textAssistant("research idle"); } },
          }),
        }),
      ],
    });

    expect(team.tools.map(tool => tool.name)).toEqual(["researcher"]);
    expect(team.tools[0]?.description).toContain("workspace paths");
    expect(team.tools[0]?.description).toContain("natural language");
  });

  test("teamMember does not keep a workspace identity field", async () => {
    const member = teamMember({
      name: "researcher",
      role: "executor",
      workspace: "/tmp/should-not-belong-to-member",
      agent: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("idle"); } },
      }),
    } as Parameters<typeof teamMember>[0] & { workspace: string });

    expect("workspace" in member).toBe(false);
  });

  test("team.query automatically drives member delegation without an explicit runner", async () => {
    const researcherAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          return textAssistant("Research complete.");
        },
      },
    });
    let leadCalls = 0;
    const team = createTeam({
      name: "engineering",
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: {
          async createMessage() {
            leadCalls++;
            if (leadCalls === 1) {
              return toolUseAssistant("toolu_1", "researcher", {
                mode: "ask",
                task: "Research the team runtime design.",
              });
            }
            return textAssistant("Engineering final.");
          },
        },
      }),
      members: [
        teamMember({
          name: "researcher",
          role: "executor",
          focus: "Research agent architecture",
          agent: researcherAgent,
        }),
      ],
      mailbox: createMemoryMailbox(),
    });

    const events = await collect(team.query("Ask the researcher to inspect this."));

    expect(events.some(event => event.type === "team_message" && event.subtype === "sent")).toBe(true);
    expect(events.some(event => event.type === "team_message" && event.subtype === "replied")).toBe(true);
    expect(events.some(event =>
      event.type === "agent_message" &&
      event.source.member === "researcher" &&
      event.message.type === "result" &&
      event.message.result === "Research complete."
    )).toBe(true);
    expect(events.some(event =>
      event.type === "user" &&
      event.tool_use_result === "Research complete."
    )).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: "result", result: "Engineering final." });
  });

  test("nested teams are driven when talking directly to the parent team", async () => {
    const backendAgent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage() {
          return textAssistant("Backend complete.");
        },
      },
    });
    let engineeringCalls = 0;
    const engineeringTeam = createTeam({
      name: "engineering",
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: {
          async createMessage() {
            engineeringCalls++;
            if (engineeringCalls === 1) {
              return toolUseAssistant("toolu_2", "backend", {
                mode: "ask",
                task: "Design the storage API.",
              });
            }
            return textAssistant("Engineering final with backend.");
          },
        },
      }),
      members: [
        teamMember({
          name: "backend",
          role: "executor",
          focus: "Backend implementation",
          agent: backendAgent,
        }),
      ],
    });
    let companyCalls = 0;
    const companyTeam = createTeam({
      name: "company",
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: {
          async createMessage() {
            companyCalls++;
            if (companyCalls === 1) {
              return toolUseAssistant("toolu_1", "engineering", {
                mode: "ask",
                task: "Plan the RAG feature.",
              });
            }
            return textAssistant("CEO final.");
          },
        },
      }),
      members: [
        teamMember({
          name: "engineering",
          role: "head",
          focus: "Own engineering delivery",
          agent: engineeringTeam,
        }),
      ],
    });

    const events = await collect(companyTeam.query("Design a RAG feature."));

    expect(events.some(event =>
      event.type === "agent_message" &&
      event.source.member === "backend" &&
      event.message.type === "result" &&
      event.message.result === "Backend complete."
    )).toBe(true);
    expect(events.some(event =>
      event.type === "user" &&
      event.tool_use_result === "Engineering final with backend."
    )).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: "result", result: "CEO final." });
  });

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

  test("mailbox allows work to return to a previous responsible member", async () => {
    const mailbox = createMemoryMailbox();
    const initial = await mailbox.send("manager", "engineering::researcher", "Research this.");
    await mailbox.updateStatus(initial.id, "done");
    const question = await mailbox.send("engineering::researcher", "manager", "Need product context.", {
      threadId: initial.threadId,
      parentMessageId: initial.id,
      workItemId: initial.workItemId,
      upstreamMessageId: initial.id,
      workItemRole: "followup",
    });
    const returned = await mailbox.send("manager", "engineering::researcher", "Here is the missing context.", {
      threadId: initial.threadId,
      parentMessageId: question.id,
      workItemId: initial.workItemId,
      upstreamMessageId: initial.id,
      workItemRole: "upstream_request",
    });

    const claimed = await mailbox.claimNext("engineering::researcher");
    const managerInbox = await mailbox.inbox("manager");

    expect(managerInbox[0]).toMatchObject({
      from: "engineering::researcher",
      to: "manager",
      content: "Need product context.",
    });
    expect(claimed).toMatchObject({
      id: returned.id,
      from: "manager",
      to: "engineering::researcher",
      status: "processing",
      threadId: initial.threadId,
      workItemId: initial.workItemId,
    });
  });

  test("lead can send work to a team member mailbox", async () => {
    let leadCalls = 0;
    const leadClient: ModelClient = {
      async createMessage() {
        leadCalls++;
        if (leadCalls === 1) {
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
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: leadClient,
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
      exposeLeadMailboxTools: true,
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
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("lead idle"); } },
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

    const leadInbox = await mailbox.inbox("manager");
    const original = await mailbox.get(sent.id);

    expect(original?.status).toBe("done");
    expect(leadInbox).toHaveLength(1);
    expect(leadInbox[0]).toMatchObject({
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
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("lead idle"); } },
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
                expect(text).toContain("Write durable deliverables in your own workspace");
                expect(text).toContain("workspace paths");
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
    const leadInbox = await mailbox.inbox("manager");

    expect(result).toMatchObject({
      processed: 1,
      failed: 0,
    });
    expect(researcherCalls).toBe(2);
    expect(testerCalls).toBe(0);
    expect(leadInbox[0]).toMatchObject({
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
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("lead idle"); } },
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
    const leadInbox = await mailbox.inbox("manager");

    expect(result.failed).toBe(1);
    expect(original?.status).toBe("failed");
    expect(leadInbox[0]).toMatchObject({
      from: "engineering::researcher",
      to: "manager",
      workItemRole: "followup",
    });
    expect(leadInbox[0]?.content).toContain("ended without team_reply or team_followup");
  });

  test("team can be nested as a member agent of another team", async () => {
    const companyMailbox = createMemoryMailbox();
    const engineeringTeam = createTeam({
      name: "engineering",
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("engineering team handled it"); } },
      }),
      members: [],
      mailbox: createMemoryMailbox(),
    });
    const companyTeam = createTeam({
      name: "company",
      lead: createAgent({
        apiKey: "test-key",
        model: "claude-test",
        modelClient: { async createMessage() { return textAssistant("ceo idle"); } },
      }),
      members: [
        teamMember({
          name: "engineering",
          role: "head",
          focus: "Own engineering work",
          agent: engineeringTeam,
        }),
      ],
      mailbox: companyMailbox,
    });
    const sent = await companyTeam.send("manager", "engineering", "Handle this through the engineering team.");

    const result = await companyTeam.drain();
    const original = await companyMailbox.get(sent.id);
    const leadInbox = await companyMailbox.inbox("manager");

    expect(companyTeam.memberTools.engineering).toEqual([]);
    expect(result).toMatchObject({ processed: 1, failed: 0 });
    expect(original?.status).toBe("done");
    expect(leadInbox[0]).toMatchObject({
      from: "company::engineering",
      to: "manager",
      content: "engineering team handled it",
      workItemRole: "upstream_report",
    });
  });
});
