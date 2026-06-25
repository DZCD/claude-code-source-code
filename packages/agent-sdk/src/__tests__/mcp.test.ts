import { describe, expect, test } from "bun:test";
import {
  connectMCPStreamableHTTPServer,
  createMCPTools,
  type MCPClient,
} from "../index.js";

describe("MCP tools", () => {
  test("maps MCP tools into SDK tool definitions", async () => {
    const client: MCPClient = {
      async listTools() {
        return {
          tools: [
            {
              name: "search",
              description: "Search documents",
              inputSchema: {
                type: "object",
                properties: { query: { type: "string" } },
                required: ["query"],
              },
            },
          ],
        };
      },
      async callTool() {
        return { content: [{ type: "text", text: "unused" }] };
      },
    };

    const tools = await createMCPTools(client);

    expect(tools).toHaveLength(1);
    expect(tools[0]).toMatchObject({
      name: "search",
      description: "Search documents",
      jsonSchema: {
        type: "object",
        properties: { query: { type: "string" } },
        required: ["query"],
      },
    });
  });

  test("calls the underlying MCP tool and returns text content", async () => {
    const calls: unknown[] = [];
    const client: MCPClient = {
      async listTools() {
        return {
          tools: [
            {
              name: "search",
              inputSchema: { type: "object" },
            },
          ],
        };
      },
      async callTool(input) {
        calls.push(input);
        return {
          content: [
            { type: "text", text: "one" },
            { type: "text", text: "two" },
          ],
        };
      },
    };

    const [search] = await createMCPTools(client);
    const result = await search.handler({ query: "sdk" }, { toolUseId: "toolu_1" });

    expect(calls).toEqual([{ name: "search", arguments: { query: "sdk" } }]);
    expect(result.content).toBe("one\ntwo");
  });

  test("supports prefixing tool names", async () => {
    const client: MCPClient = {
      async listTools() {
        return {
          tools: [
            {
              name: "search",
              inputSchema: { type: "object" },
            },
          ],
        };
      },
      async callTool() {
        return { content: [{ type: "text", text: "ok" }] };
      },
    };

    const [search] = await createMCPTools(client, { namePrefix: "docs" });

    expect(search.name).toBe("docs_search");
    const result = await search.handler({}, { toolUseId: "toolu_1" });
    expect(result.content).toBe("ok");
  });

  test("marks MCP tool errors as thrown handler errors", async () => {
    const client: MCPClient = {
      async listTools() {
        return {
          tools: [
            {
              name: "explode",
              inputSchema: { type: "object" },
            },
          ],
        };
      },
      async callTool() {
        return {
          isError: true,
          content: [{ type: "text", text: "boom" }],
        };
      },
    };

    const [explode] = await createMCPTools(client);

    await expect(explode.handler({}, { toolUseId: "toolu_1" })).rejects.toThrow("boom");
  });

  test("exports remote Streamable HTTP connector", () => {
    expect(typeof connectMCPStreamableHTTPServer).toBe("function");
  });
});
