import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  createAgent,
  loadSkill,
  skill,
  type ModelClient,
  type ModelMessage,
} from "../index.js";

const tempDirs: string[] = [];

async function tempWorkspace(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "agent-sdk-skills-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map(dir => rm(dir, { recursive: true, force: true })));
});

function textAssistant(text: string) {
  return {
    role: "assistant" as const,
    content: [{ type: "text" as const, text }],
  };
}

describe("skills", () => {
  test("creates an in-memory skill definition", () => {
    const markdown = "# Code Review\n\nAlways mention risks first.";

    const definition = skill({
      name: "code-review",
      description: "Review code changes",
      instructions: markdown,
    });

    expect(definition).toEqual({
      name: "code-review",
      description: "Review code changes",
      instructions: markdown,
    });
  });

  test("loads a skill from a SKILL.md file with frontmatter", async () => {
    const dir = await tempWorkspace();
    await writeFile(
      join(dir, "SKILL.md"),
      [
        "---",
        "name: pdf",
        "description: Read and inspect PDF documents",
        "---",
        "",
        "# PDF",
        "",
        "Use page rendering before claiming layout is correct.",
      ].join("\n"),
      "utf8",
    );

    const definition = await loadSkill(dir);

    expect(definition.name).toBe("pdf");
    expect(definition.description).toBe("Read and inspect PDF documents");
    expect(definition.instructions).toContain("Use page rendering");
    expect(definition.path).toBe(dir);
  });

  test("injects matching skill instructions into the model request", async () => {
    let seenMessages: ModelMessage[] = [];
    const modelClient: ModelClient = {
      async createMessage({ messages }) {
        seenMessages = messages;
        return textAssistant("reviewed");
      },
    };
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient,
      skills: [
        skill({
          name: "code-review",
          description: "Review code changes and pull requests",
          instructions: "Always list bugs before summaries.",
        }),
      ],
    });

    await agent.prompt("Please review this pull request.");

    expect(seenMessages[0]?.role).toBe("user");
    expect(String(seenMessages[0]?.content)).toContain("<skill name=\"code-review\">");
    expect(String(seenMessages[0]?.content)).toContain("Always list bugs before summaries.");
    expect(seenMessages[1]).toMatchObject({
      role: "user",
      content: "Please review this pull request.",
    });
  });

  test("does not inject unrelated skills", async () => {
    let seenMessages: ModelMessage[] = [];
    const modelClient: ModelClient = {
      async createMessage({ messages }) {
        seenMessages = messages;
        return textAssistant("hello");
      },
    };
    const agent = createAgent({
      apiKey: "test-key",
      model: "claude-test",
      modelClient,
      skills: [
        skill({
          name: "pdf",
          description: "Read PDF documents",
          instructions: "Render pages before answering.",
        }),
      ],
    });

    await agent.prompt("Say hello.");

    expect(seenMessages).toEqual([
      {
        role: "user",
        content: "Say hello.",
      },
    ]);
  });
});
