import { Database } from "bun:sqlite";
import { describe, expect, test } from "bun:test";
import {
  createSQLiteMailbox,
  type TeamMailbox,
} from "../index.js";

async function roundTrip(mailbox: TeamMailbox) {
  const first = await mailbox.send("manager", "engineering::researcher", "Research SQLite mailbox.");
  const second = await mailbox.send("engineering::researcher", "manager", "Research complete.", {
    threadId: first.threadId,
    parentMessageId: first.id,
    workItemId: first.workItemId,
    upstreamMessageId: first.upstreamMessageId,
    workItemRole: "upstream_report",
  });

  return { first, second };
}

describe("SQLite mailbox", () => {
  test("persists team messages and work item context", async () => {
    const database = new Database(":memory:");
    const mailbox = createSQLiteMailbox({ database });

    const { first, second } = await roundTrip(mailbox);
    const supervisorInbox = await mailbox.inbox("manager");
    const storedFirst = await mailbox.get(first.id);

    expect(supervisorInbox).toHaveLength(1);
    expect(supervisorInbox[0]).toMatchObject({
      id: second.id,
      from: "engineering::researcher",
      to: "manager",
      content: "Research complete.",
      threadId: first.threadId,
      parentMessageId: first.id,
      workItemId: first.workItemId,
      workItemRole: "upstream_report",
      upstreamMessageId: first.id,
    });
    expect(storedFirst).toMatchObject({
      status: "pending",
      workItemRole: "upstream_request",
    });
  });

  test("can reopen an existing SQLite database", async () => {
    const database = new Database(":memory:");
    const mailbox = createSQLiteMailbox({ database });
    const sent = await mailbox.send("manager", "engineering::researcher", "Persist me.");

    const reopened = createSQLiteMailbox({ database });
    const message = await reopened.get(sent.id);

    expect(message).toMatchObject({
      id: sent.id,
      content: "Persist me.",
      workItemId: sent.id,
    });
  });

  test("claimNext atomically marks one addressed message processing", async () => {
    const database = new Database(":memory:");
    const mailbox = createSQLiteMailbox({ database });
    await mailbox.send("manager", "engineering::researcher", "Research this.");
    await mailbox.send("manager", "engineering::tester", "Test this.");

    const claimed = await mailbox.claimNext("engineering::researcher");
    const next = await mailbox.claimNext("engineering::researcher");
    const testerInbox = await mailbox.inbox("engineering::tester");

    expect(claimed).toMatchObject({
      to: "engineering::researcher",
      status: "processing",
    });
    expect(next).toBeUndefined();
    expect(testerInbox).toHaveLength(1);
    expect(testerInbox[0]).toMatchObject({ status: "pending" });
  });
});
