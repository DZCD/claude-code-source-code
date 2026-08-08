import { describe, expect, test } from "bun:test";
import { appendFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod/v4";
import {
  createAgent,
  createJsonlHistoryStore,
  defineHistoryStore,
  tool,
  type HistoryStore,
  type ModelClient,
  type ModelMessage,
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

type RecordedCall = { method: "append" | "replace"; messages: ModelMessage[] };

function createRecordingStore(initial: ModelMessage[] = []) {
  const calls: RecordedCall[] = [];
  const stored: ModelMessage[] = structuredClone(initial);
  const store: HistoryStore = {
    load: () => structuredClone(stored),
    append(message) {
      calls.push({ method: "append", messages: [structuredClone(message)] });
      stored.push(structuredClone(message));
    },
    replace(messages) {
      calls.push({ method: "replace", messages: structuredClone(messages) });
      stored.length = 0;
      stored.push(...structuredClone(messages));
    },
  };
  return { store, calls, stored };
}

describe("HistoryStore", () => {
  test("notifies append in order for prompt, assistant, and tool_result writes", async () => {
    const { store, calls } = createRecordingStore();
    let turn = 0;
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: store,
      tools: [tool("noop", "No-op", z.object({}), async () => ({ content: "ok" }))],
      modelClient: {
        async createMessage() {
          turn++;
          return turn === 1 ? toolUseAssistant("toolu_1", "noop", {}) : textAssistant("done");
        },
      },
    });

    const result = await agent.prompt("start the task");
    expect(result.subtype).toBe("success");

    expect(calls.map(call => call.method)).toEqual(["append", "append", "append", "append"]);
    const written = calls.flatMap(call => call.messages);
    expect(written[0]).toEqual({ role: "user", content: "start the task" });
    expect(written[1]).toMatchObject({ role: "assistant", content: [{ type: "tool_use", name: "noop" }] });
    expect(written[2]).toMatchObject({
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "ok" }],
    });
    expect(written[3]).toMatchObject({ role: "assistant", content: [{ type: "text", text: "done" }] });

    // The store saw exactly what the agent holds.
    expect(await agent.getHistory()).toEqual(written);
  });

  test("compaction persists through replace, matching getHistory()", async () => {
    const { store, calls, stored } = createRecordingStore();
    let turn = 0;
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: store,
      autoCompact: { thresholdTokens: 1_000, keepRecentMessages: 2 },
      tools: [tool("noop", "No-op", z.object({}), async () => ({ content: "ok" }))],
      modelClient: {
        async createMessage({ systemPrompt }) {
          if (systemPrompt?.includes("create a detailed summary")) {
            return { ...textAssistant("SUMMARY OF EARLIER WORK"), usage: { input_tokens: 900, output_tokens: 50 } };
          }
          turn++;
          if (turn <= 2) {
            return {
              ...toolUseAssistant(`toolu_${turn}`, "noop", {}),
              usage: { input_tokens: 5_000, output_tokens: 10 },
            };
          }
          return { ...textAssistant("done"), usage: { input_tokens: 100, output_tokens: 10 } };
        },
      },
    });

    const messages = await collect(agent.query("start the task"));
    expect(messages.some(message => message.type === "system" && message.subtype === "compaction")).toBe(true);

    const replacements = calls.filter(call => call.method === "replace");
    expect(replacements.length).toBeGreaterThanOrEqual(1);
    // The replacement mirrors the live history at that point; later turns
    // append again on top of it.
    const replaced = replacements.at(-1)!.messages;
    expect(JSON.stringify(replaced)).toContain("SUMMARY OF EARLIER WORK");
    const history = await agent.getHistory();
    expect(replaced).toEqual(history.slice(0, replaced.length));
    // Applying every append/replace leaves the store identical to the agent.
    expect(stored).toEqual(history);
  });

  test("seeds a new Agent from the store on the first query", async () => {
    const seed: ModelMessage[] = [
      { role: "user", content: "My name is Ada." },
      textAssistant("Nice to meet you, Ada."),
    ];
    const { store } = createRecordingStore(seed);
    const requests: ModelMessage[][] = [];
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: store,
      modelClient: {
        async createMessage({ messages }) {
          requests.push(messages);
          return textAssistant("done");
        },
      },
    });

    await agent.prompt("What is my name?");

    expect(requests[0]!.slice(0, 2)).toEqual(seed);
    expect(requests[0]!.at(-1)).toEqual({ role: "user", content: "What is my name?" });
  });

  test("getHistory returns a copy that cannot pollute the live history", async () => {
    const requests: ModelMessage[][] = [];
    const modelClient: ModelClient = {
      async createMessage({ messages }) {
        requests.push(structuredClone(messages));
        return textAssistant("done");
      },
    };
    const agent = createAgent({ apiKey: "test-key", model: "claude-test", modelClient });

    await agent.prompt("first");
    const snapshot = await agent.getHistory();
    snapshot[0]!.content = "HACKED";
    (snapshot as ModelMessage[]).length = 0;

    await agent.prompt("second");
    const secondRequest = requests.at(-1)!;
    expect(JSON.stringify(secondRequest)).not.toContain("HACKED");
    expect(secondRequest[0]).toEqual({ role: "user", content: "first" });
    expect(secondRequest.at(-1)).toEqual({ role: "user", content: "second" });
  });

  test("a failing store is swallowed by default and the query completes", async () => {
    const store: HistoryStore = {
      load: () => [],
      append() {
        throw new Error("disk full");
      },
      replace() {
        throw new Error("disk full");
      },
    };
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: store,
      modelClient: { async createMessage() { return textAssistant("done"); } },
    });

    const result = await agent.prompt("hi");
    expect(result.subtype).toBe("success");
    expect(await agent.getHistory()).toHaveLength(2);
  });

  test("failOnError propagates store write failures out of query()", async () => {
    const store = defineHistoryStore({
      failOnError: true,
      load: () => [],
      append() {
        throw new Error("disk full");
      },
      replace() {},
    });
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: store,
      modelClient: { async createMessage() { return textAssistant("done"); } },
    });

    await expect(agent.prompt("hi")).rejects.toThrow("disk full");
  });

  test("a failed load follows the same failOnError rule", async () => {
    const makeStore = (failOnError?: boolean): HistoryStore => defineHistoryStore({
      failOnError,
      load() {
        throw new Error("corrupt store");
      },
      append() {},
      replace() {},
    });
    const modelClient: ModelClient = {
      async createMessage() { return textAssistant("done"); },
    };

    const tolerant = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: makeStore(),
      modelClient,
    });
    const result = await tolerant.prompt("hi");
    expect(result.subtype).toBe("success");

    const strict = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: makeStore(true),
      modelClient,
    });
    await expect(strict.prompt("hi")).rejects.toThrow("corrupt store");
  });

  test("replaceHistory sends the next query the replacement, not the seeded history", async () => {
    const seed: ModelMessage[] = [
      { role: "user", content: "stale question" },
      textAssistant("stale answer"),
    ];
    const { store, calls, stored } = createRecordingStore(seed);
    const requests: ModelMessage[][] = [];
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: store,
      modelClient: {
        async createMessage({ messages }) {
          requests.push(structuredClone(messages));
          return textAssistant("done");
        },
      },
    });

    // Replaced before the first query: the lazy load must not seed over it.
    const replacement: ModelMessage[] = [
      { role: "user", content: "My name is Ada." },
      textAssistant("Nice to meet you, Ada."),
    ];
    await agent.replaceHistory(replacement);
    await agent.prompt("What is my name?");

    expect(calls.map(call => call.method)).toEqual(["replace", "append", "append"]);
    expect(calls[0]!.messages).toEqual(replacement);
    expect(requests[0]).toEqual([
      ...replacement,
      { role: "user", content: "What is my name?" },
    ]);
    // The store and the agent agree after the query.
    expect(stored).toEqual(await agent.getHistory());
  });

  test("replaceHistory mutating its argument afterwards cannot pollute the live history", async () => {
    const requests: ModelMessage[][] = [];
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient: {
        async createMessage({ messages }) {
          requests.push(structuredClone(messages));
          return textAssistant("done");
        },
      },
    });

    const replacement: ModelMessage[] = [{ role: "user", content: "first" }];
    await agent.replaceHistory(replacement);
    replacement[0]!.content = "HACKED";

    await agent.prompt("second");
    expect(JSON.stringify(requests[0])).not.toContain("HACKED");
    expect(requests[0]).toEqual([
      { role: "user", content: "first" },
      { role: "user", content: "second" },
    ]);
  });

  test("a failing store replace inside replaceHistory follows the failOnError rule", async () => {
    const makeStore = (failOnError?: boolean): HistoryStore => defineHistoryStore({
      failOnError,
      load: () => [],
      append() {},
      replace() {
        throw new Error("disk full");
      },
    });

    const tolerant = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: makeStore(),
      modelClient: { async createMessage() { return textAssistant("done"); } },
    });
    await tolerant.replaceHistory([{ role: "user", content: "hi" }]);
    expect(await tolerant.getHistory()).toEqual([{ role: "user", content: "hi" }]);

    const strict = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      historyStore: makeStore(true),
      modelClient: { async createMessage() { return textAssistant("done"); } },
    });
    await expect(strict.replaceHistory([{ role: "user", content: "hi" }])).rejects.toThrow("disk full");
  });
});

describe("defineHistoryStore", () => {
  test("requires load, append, and replace methods", () => {
    expect(() => defineHistoryStore({} as never)).toThrow("load");
    expect(() => defineHistoryStore({ load: () => [] } as never)).toThrow("append");
    expect(() => defineHistoryStore({ load: () => [], append() {} } as never)).toThrow("replace");
  });

  test("exposes methods only, not the failOnError flag", () => {
    const store = defineHistoryStore({
      failOnError: true,
      load: () => [],
      append() {},
      replace() {},
    });
    expect("failOnError" in store).toBe(false);
  });
});

describe("createJsonlHistoryStore", () => {
  test("round-trips messages across store instances", async () => {
    const dir = await mkdtemp(join(tmpdir(), "history-store-"));
    try {
      const path = join(dir, "history.jsonl");
      const history: ModelMessage[] = [
        { role: "user", content: "hello" },
        textAssistant("hi there"),
        { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "ok" }] },
      ];
      const writer = createJsonlHistoryStore({ path });
      for (const message of history) {
        await writer.append(message);
      }

      const reader = createJsonlHistoryStore({ path });
      expect(await reader.load()).toEqual(history);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("replace rewrites the file", async () => {
    const dir = await mkdtemp(join(tmpdir(), "history-store-"));
    try {
      const path = join(dir, "history.jsonl");
      const store = createJsonlHistoryStore({ path });
      await store.append({ role: "user", content: "old" });
      await store.append(textAssistant("old answer"));

      const compacted: ModelMessage[] = [{ role: "user", content: "summary handoff" }];
      await store.replace(compacted);

      const reader = createJsonlHistoryStore({ path });
      expect(await reader.load()).toEqual(compacted);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("load skips malformed lines and keeps the rest", async () => {
    const dir = await mkdtemp(join(tmpdir(), "history-store-"));
    try {
      const path = join(dir, "history.jsonl");
      const store = createJsonlHistoryStore({ path });
      await store.append({ role: "user", content: "good" });
      await appendFile(path, '{"role":"user","content":"torn', "utf8");
      await appendFile(path, '{"role":"system","content":"not a model message"}\n', "utf8");

      expect(await createJsonlHistoryStore({ path }).load()).toEqual([
        { role: "user", content: "good" },
      ]);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("load on a missing file returns an empty history", async () => {
    const dir = await mkdtemp(join(tmpdir(), "history-store-"));
    try {
      const store = createJsonlHistoryStore({ path: join(dir, "missing.jsonl") });
      expect(await store.load()).toEqual([]);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
