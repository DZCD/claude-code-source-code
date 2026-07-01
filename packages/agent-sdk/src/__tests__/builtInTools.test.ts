import { afterEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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

  test("read-only file tools can inspect outside allowed directories", async () => {
    const cwd = await tempWorkspace();
    const outside = await tempWorkspace();
    await mkdir(join(outside, "src"));
    const outsideFile = join(outside, "secret.txt");
    await writeFile(outsideFile, "secret");
    await writeFile(join(outside, "src/app.ts"), "export const answer = 42\n");

    const read = await findTool("Read", cwd).handler({ file_path: outsideFile }, { toolUseId: "toolu_1" });
    const ls = await findTool("LS", cwd).handler({ path: outside }, { toolUseId: "toolu_2" });
    const glob = await findTool("Glob", cwd).handler(
      { path: outside, pattern: "*.txt" },
      { toolUseId: "toolu_3" },
    );
    const grep = await findTool("Grep", cwd).handler(
      { path: outside, pattern: "answer" },
      { toolUseId: "toolu_4" },
    );

    expect(read.content).toBe("secret");
    expect(String(ls.content)).toContain("secret.txt");
    expect(String(glob.content)).toContain("secret.txt");
    expect(String(grep.content)).toContain("src/app.ts:1:export const answer = 42");
  });

  test("write tools reject paths outside allowed directories", async () => {
    const cwd = await tempWorkspace();
    const outside = join(await tempWorkspace(), "secret.txt");
    await writeFile(outside, "secret");

    await expect(
      findTool("Write", cwd).handler(
        { file_path: outside, content: "nope" },
        { toolUseId: "toolu_1" },
      ),
    ).rejects.toThrow("outside allowed write roots");
    await expect(
      findTool("Edit", cwd).handler(
        { file_path: outside, old_string: "secret", new_string: "nope" },
        { toolUseId: "toolu_2" },
      ),
    ).rejects.toThrow("outside allowed write roots");
  });

  test("write tools can use runtime workspace grants", async () => {
    const cwd = await tempWorkspace();
    const shared = await tempWorkspace();
    const sharedFile = join(shared, "server.js");

    await findTool("Write", cwd).handler(
      { file_path: sharedFile, content: "shared" },
      {
        toolUseId: "toolu_1",
        permissions: {
          workspaceGrants: [{
            root: shared,
            access: ["write"],
            reason: "shared backend workspace",
          }],
        },
      },
    );
    const result = await findTool("Read", cwd).handler(
      { file_path: sharedFile },
      { toolUseId: "toolu_2" },
    );

    expect(result.content).toBe("shared");
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

  test("Bash rejects obvious writes outside allowed roots", async () => {
    const cwd = await tempWorkspace();
    const outside = join(await tempWorkspace(), "escape.txt");

    await expect(
      findTool("Bash", cwd).handler(
        { command: `printf nope > ${outside}`, timeout_ms: 2000 },
        { toolUseId: "toolu_1" },
      ),
    ).rejects.toThrow("outside allowed write roots");
  });

  test("Bash allows obvious reads outside allowed roots", async () => {
    const cwd = await tempWorkspace();
    const outside = join(await tempWorkspace(), "outside.txt");
    await writeFile(outside, "visible");

    const result = await findTool("Bash", cwd).handler(
      { command: `cat ${outside}`, timeout_ms: 2000 },
      { toolUseId: "toolu_1" },
    );

    expect(result.content).toContain("visible");
  });

  test("Bash allows obvious writes inside runtime workspace grants", async () => {
    const cwd = await tempWorkspace();
    const shared = await tempWorkspace();
    const output = join(shared, "out.txt");

    await findTool("Bash", cwd).handler(
      { command: `printf sdk > ${output}`, timeout_ms: 2000 },
      {
        toolUseId: "toolu_1",
        permissions: {
          workspaceGrants: [{
            root: shared,
            access: ["write"],
            reason: "shared shell workspace",
          }],
        },
      },
    );

    expect(await readFile(output, "utf8")).toBe("sdk");
  });
});
