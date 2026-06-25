import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createClaudeCodeTools } from "../index.js";

const tempDirs: string[] = [];

async function tempWorkspace(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "agent-sdk-tools-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map(dir => rm(dir, { recursive: true, force: true })));
});

function findTool(name: string, cwd: string) {
  const found = createClaudeCodeTools({ cwd }).find(tool => tool.name === name);
  if (!found) throw new Error(`Tool ${name} not found`);
  return found;
}

describe("Claude Code built-in tools", () => {
  test("exports Claude Code style built-in tool names", async () => {
    const cwd = await tempWorkspace();

    const names = createClaudeCodeTools({ cwd }).map(tool => tool.name);

    expect(names).toEqual(["Read", "Write", "Edit", "LS", "Glob", "Grep", "Bash"]);
  });

  test("Read reads files with optional offset and limit", async () => {
    const cwd = await tempWorkspace();
    await writeFile(join(cwd, "notes.txt"), "one\ntwo\nthree\n");

    const result = await findTool("Read", cwd).handler(
      { file_path: "notes.txt", offset: 2, limit: 1 },
      { toolUseId: "toolu_1" },
    );

    expect(result.content).toBe("two");
  });

  test("Write and Edit update files inside the workspace", async () => {
    const cwd = await tempWorkspace();

    await findTool("Write", cwd).handler(
      { file_path: "src/app.txt", content: "hello world" },
      { toolUseId: "toolu_1" },
    );
    await findTool("Edit", cwd).handler(
      { file_path: "src/app.txt", old_string: "world", new_string: "sdk" },
      { toolUseId: "toolu_2" },
    );

    expect(await readFile(join(cwd, "src/app.txt"), "utf8")).toBe("hello sdk");
  });

  test("file tools reject paths outside allowed directories", async () => {
    const cwd = await tempWorkspace();
    const outside = join(await tempWorkspace(), "secret.txt");
    await writeFile(outside, "secret");

    await expect(
      findTool("Read", cwd).handler({ file_path: outside }, { toolUseId: "toolu_1" }),
    ).rejects.toThrow("outside allowed directories");
  });

  test("LS, Glob, and Grep inspect workspace files", async () => {
    const cwd = await tempWorkspace();
    await findTool("Write", cwd).handler(
      { file_path: "src/app.ts", content: "export const answer = 42\n" },
      { toolUseId: "toolu_1" },
    );
    await findTool("Write", cwd).handler(
      { file_path: "README.md", content: "hello\n" },
      { toolUseId: "toolu_2" },
    );

    const ls = await findTool("LS", cwd).handler({ path: "." }, { toolUseId: "toolu_3" });
    const glob = await findTool("Glob", cwd).handler({ pattern: "**/*.ts" }, { toolUseId: "toolu_4" });
    const grep = await findTool("Grep", cwd).handler({ pattern: "answer", path: "." }, { toolUseId: "toolu_5" });

    expect(String(ls.content)).toContain("README.md");
    expect(String(glob.content)).toContain("src/app.ts");
    expect(String(grep.content)).toContain("src/app.ts:1:export const answer = 42");
  });

  test("Bash executes commands in the workspace", async () => {
    const cwd = await tempWorkspace();

    const result = await findTool("Bash", cwd).handler(
      { command: "printf sdk", timeout_ms: 2000 },
      { toolUseId: "toolu_1" },
    );

    expect(result.content).toContain("sdk");
  });
});
