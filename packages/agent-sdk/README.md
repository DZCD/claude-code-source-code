# Claude Agent SDK

A lightweight npm package for building Claude-powered agents without installing
the Claude Code CLI runtime.

This first version supports Anthropic direct API calls, an in-memory agent loop,
custom tools, permission callbacks, and stable SDK-style events.

## Install

```bash
npm install @npm-while1/claude-agent-sdk zod
```

## Minimal Usage

The examples use DeepSeek's Anthropic-compatible endpoint.

```ts
import { createAgent } from "@npm-while1/claude-agent-sdk";

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
});

for await (const message of agent.query("Say hello")) {
  console.log(message);
}
```

Pass `{ stream: false }` to disable model streaming for a query:

```ts
const result = await agent.prompt("Say hello", { stream: false });
```

## DeepSeek Anthropic-compatible API

DeepSeek exposes an Anthropic-compatible endpoint. Configure `baseURL` and use a
DeepSeek model name:

```ts
const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
});
```

## Custom Tool

```ts
import { createAgent, tool } from "@npm-while1/claude-agent-sdk";
import { z } from "zod/v4";

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  tools: [
    tool(
      "calculator",
      "Evaluate a simple arithmetic expression",
      z.object({ expr: z.string() }),
      async input => ({ content: String(Function(`return ${input.expr}`)()) }),
    ),
  ],
});

const result = await agent.prompt("What is 2+2?");
console.log(result.result);
```

## Permission Callback

```ts
const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  tools: [dangerousTool],
  permission: async request => {
    if (request.toolName === "danger") {
      return { behavior: "deny", message: "Blocked by policy" };
    }
    return { behavior: "allow" };
  },
});
```

Denied tools are returned to Claude as error `tool_result` blocks so the model
can explain or choose another path.

## Skills

Skills are reusable instruction bundles. They are lighter than Claude Code
runtime plugins: the SDK reads skill instructions and injects matching skills
into the model request, but it does not depend on the Claude Code runtime.

```ts
import { createAgent, loadSkill, skill } from "@npm-while1/claude-agent-sdk";

const codeReview = skill({
  name: "code-review",
  description: "Review code changes and pull requests",
  instructions: "Always list bugs and risks before summaries.",
});

const pdf = await loadSkill("./skills/pdf");

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  skills: [codeReview, pdf],
});
```

`loadSkill(path)` expects a `SKILL.md` file:

```md
---
name: pdf
description: Read and inspect PDF documents
---

Render pages before claiming layout is correct.
```

## MCP Tools

The SDK can expose MCP server tools as agent tools. The first version supports
stdio MCP servers, remote Streamable HTTP servers, OAuth providers, and a
generic `MCPClient` adapter.

```ts
import {
  connectMCPStdioServer,
  createAgent,
} from "@npm-while1/claude-agent-sdk";

const mcp = await connectMCPStdioServer(
  {
    command: "node",
    args: ["./mcp-server.js"],
  },
  {
    namePrefix: "docs",
  },
);

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  tools: mcp.tools,
});

try {
  const result = await agent.prompt("Search the docs for installation steps.");
  console.log(result.result);
} finally {
  await mcp.close();
}
```

Use `createMCPTools(client)` if your host application already manages an MCP
client connection.

Connect a remote Streamable HTTP MCP server:

```ts
import {
  connectMCPStreamableHTTPServer,
  createAgent,
} from "@npm-while1/claude-agent-sdk";

const mcp = await connectMCPStreamableHTTPServer("https://mcp.example.com/mcp", {
  namePrefix: "remote",
  requestInit: {
    headers: {
      "X-Workspace": "demo",
    },
  },
});

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  tools: mcp.tools,
});
```

Pass an official MCP `OAuthClientProvider` when the remote server requires OAuth:

```ts
const mcp = await connectMCPStreamableHTTPServer("https://mcp.example.com/mcp", {
  authProvider,
});
```

## Sub-agents and Multi-agent Delegation

The SDK supports lightweight supervisor/sub-agent delegation. A sub-agent is an
`Agent` wrapped as a delegate tool. The supervisor can call that tool, receive
the sub-agent result, and continue its own loop.

```ts
import {
  createAgent,
  createMultiAgent,
  createSubAgent,
} from "@npm-while1/claude-agent-sdk";

const researcher = createSubAgent({
  name: "researcher",
  description: "Research SDK implementation details",
  agent: createAgent({
    apiKey: process.env.DEEPSEEK_API_KEY,
    baseURL: "https://api.deepseek.com/anthropic",
    model: "deepseek-v4-flash",
  }),
});

const team = createMultiAgent({
  supervisor: createAgent({
    apiKey: process.env.DEEPSEEK_API_KEY,
    baseURL: "https://api.deepseek.com/anthropic",
    model: "deepseek-v4-flash",
  }),
  subAgents: [researcher],
});

const result = await team.prompt("Use the researcher to inspect the SDK design.");
```

The supervisor receives a tool named `delegate_researcher`. Use the normal
permission callback on the supervisor if your host wants to approve delegation.

## Team Mailbox Collaboration

Use `createTeam()` when you want longer-lived team members that coordinate
through mailbox messages instead of a single delegate call.

```ts
import {
  createAgent,
  createMemoryMailbox,
  createTeam,
  teamMember,
} from "@npm-while1/claude-agent-sdk";

const team = createTeam({
  name: "engineering",
  supervisor: createAgent({
    apiKey: process.env.DEEPSEEK_API_KEY,
    baseURL: "https://api.deepseek.com/anthropic",
    model: "deepseek-v4-flash",
  }),
  members: [
    teamMember({
      name: "researcher",
      role: "executor",
      focus: "Research agent architecture",
      agent: createAgent({
        apiKey: process.env.DEEPSEEK_API_KEY,
        baseURL: "https://api.deepseek.com/anthropic",
        model: "deepseek-v4-flash",
      }),
    }),
  ],
  mailbox: createMemoryMailbox(),
});

await team.prompt("Ask the researcher to inspect the SDK design.");
```

`createTeam()` injects `team_send`, `team_inbox`, `team_read`, `team_reply`,
`team_followup`, and `team_status`. The default mailbox is in memory.

For durable local storage, pass a SQLite-like database. `better-sqlite3` works
without the SDK taking a hard dependency on it:

```ts
import Database from "better-sqlite3";
import {
  createSQLiteMailbox,
  createTeam,
} from "@npm-while1/claude-agent-sdk";

const mailbox = createSQLiteMailbox({
  database: new Database("team-mailbox.db"),
});

const team = createTeam({
  name: "engineering",
  supervisor,
  members,
  mailbox,
});
```

Hosts can also provide their own `TeamMailbox` adapter for Redis, Cloudflare D1,
Durable Objects, or another queue/storage backend.

### Team runtime drain

Mailbox routing is explicit: a pending message belongs to its `to` mailbox and
must be handled by that member's agent. `claimNext(mailboxId)` only claims one
pending message for that mailbox and marks it `processing`.

```ts
const message = await team.mailbox.claimNext("engineering::researcher");
```

Use `team.drain()` to let the runtime advance already-routed work:

```ts
const result = await team.drain({
  maxRounds: 5,
  maxMessages: 20,
});
```

`drain()` iterates members, claims pending messages from each member's own
mailbox, and prompts that member agent. It does not re-route work. The member
must call `team_reply` for a final result or `team_followup` for progress. If a
member ends without either, the runtime marks the original message `failed` and
sends a diagnostic follow-up to the upstream mailbox.

## Claude Code-style Built-in Tools

The SDK includes an opt-in set of Claude Code-style tools:

- `Read`
- `Write`
- `Edit`
- `LS`
- `Glob`
- `Grep`
- `Bash`

```ts
import { createAgent, createClaudeCodeTools } from "@npm-while1/claude-agent-sdk";

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  tools: createClaudeCodeTools({
    cwd: process.cwd(),
    allowedDirectories: [process.cwd()],
  }),
  permission: async request => {
    if (request.toolName === "Bash" || request.toolName === "Write" || request.toolName === "Edit") {
      return { behavior: "deny", message: "This host did not approve write or shell access." };
    }
    return { behavior: "allow" };
  },
});
```

These tools are not enabled by default. They can read, write, edit, search, and
execute shell commands in the configured workspace, so production hosts should
pair them with a permission callback.

## Multi-turn Session

```ts
const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
});

await agent.prompt("My name is Ada.");
const result = await agent.prompt("What is my name?");
console.log(result.result);
```

The SDK stores conversation state in memory for the lifetime of the `Agent`
instance. Persistent transcripts and resume support are intentionally out of
scope for the first release.
