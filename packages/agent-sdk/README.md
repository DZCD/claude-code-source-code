# AgentLattice

AgentLattice is a TypeScript framework for building coordinated agent systems
with tools, skills, tracing, supervisor delegation, and mailbox-backed teams.

Install it from npm as `agent-lattice`. It works with Anthropic and
Anthropic-compatible providers such as DeepSeek, without installing the Claude
Code CLI runtime.

## Install

```bash
npm install agent-lattice zod
```

## Minimal Usage

The examples use DeepSeek's Anthropic-compatible endpoint.

```ts
import { createAgent } from "agent-lattice";

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
});

for await (const message of agent.query("Say hello")) {
  console.log(message);
}
```

`createAgent()` includes a private workspace and the built-in file/shell tools
by default. Use `createBareAgent()` when your host wants to provide every prompt
and tool explicitly:

```ts
import { createBareAgent, createBuiltinTools } from "agent-lattice";

const agent = createBareAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  systemPrompt: "You are a concise engineering assistant.",
  tools: createBuiltinTools({
    cwd: process.cwd(),
    allowedDirectories: [process.cwd()],
  }),
});
```

Pass `{ stream: false }` to disable model streaming for a query:

```ts
const result = await agent.prompt("Say hello", { stream: false });
```

Pass `outputFormat` when you want the model to produce JSON matching a schema:

```ts
const jsonResult = await agent.prompt("Return JSON only.", {
  outputFormat: "json",
});

const result = await agent.prompt("Return the answer to 2 + 2.", {
  outputFormat: {
    type: "json_schema",
    schema: {
      type: "object",
      properties: {
        answer: { type: "number" },
      },
      required: ["answer"],
      additionalProperties: false,
    },
  },
});
```

When `outputFormat` is set, the SDK sends structured output parameters to the
provider and returns the final text unchanged. Parse or validate the returned
JSON in your application when you need a typed value.

Pass `thinkingConfig` on the agent to configure reasoning for every query, or
on an individual query to override the agent default:

```ts
const agent = createAgent({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: "claude-sonnet-4-6",
  thinkingConfig: { type: "adaptive" },
});

const result = await agent.prompt("Solve this carefully.", {
  thinkingConfig: { type: "enabled", budgetTokens: 8_000 },
});
```

Use `{ type: "adaptive" }` for models that support adaptive thinking. Use
`{ type: "enabled", budgetTokens }` for models that require a fixed budget, or
`{ type: "disabled" }` to omit thinking from the provider request. A fixed
budget is capped at `maxTokens - 1` to satisfy the Anthropic API constraint.
When omitted, the SDK does not send a thinking configuration.

## Multimodal Input

Pass Anthropic-compatible content blocks for image or document prompts:

```ts
const result = await agent.prompt([
  { type: "text", text: "Summarize this screenshot." },
  {
    type: "image",
    source: {
      type: "base64",
      media_type: "image/png",
      data: imageBase64,
    },
  },
]);

console.log(result.result);
```

## JSONL Context Tracing

Pass a `ContextTracer` to observe an agent run without changing the agent loop.
The built-in JSONL tracer writes one structured event per line:

```ts
import { createAgent, createJsonlContextTracer } from "agent-lattice";

const tracer = createJsonlContextTracer({
  path: ".agent-runs/session.jsonl",
});

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  tracer,
});

await agent.prompt("Remember that my name is Ada.");
```

Each JSONL entry includes `session_id`, `run_id`, `seq`, `source`, `type`, and
`data`. Agent runs record transcript and context events such as `run_start`,
`user_message`, `model_request`, `assistant_message`, `tool_use`,
`tool_result`, and `result`. For team runners, pass the tracer per query to
propagate it into delegated agents:

```ts
for await (const event of team.query("Ask engineering to investigate.", {
  tracer,
})) {
  console.log(event);
}
```

## LangSmith Context Tracing

Pass LangSmith's `RunTree` constructor to the SDK tracer. The SDK depends on
`langsmith` directly and uses its official `RunTree` / `RunTreeConfig` types for
this adapter.

Configure LangSmith with its standard environment variables:

```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your-langsmith-api-key>
LANGSMITH_PROJECT=<your-langsmith-project>
# Required only for org-scoped or multi-workspace API keys.
LANGSMITH_WORKSPACE_ID=<your-langsmith-workspace-id>
```

```ts
import { RunTree } from "langsmith/run_trees";
import {
  createAgent,
  createCompositeContextTracer,
  createJsonlContextTracer,
  createLangSmithContextTracer,
} from "agent-lattice";

const tracer = createCompositeContextTracer([
  createJsonlContextTracer({ path: ".agent-runs/session.jsonl" }),
  createLangSmithContextTracer({
    RunTree,
    projectName: process.env.LANGSMITH_PROJECT,
    workspaceId: process.env.LANGSMITH_WORKSPACE_ID,
    tags: ["local-debug"],
  }),
]);

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  tracer,
});

try {
  await agent.prompt("Trace this run.", { stream: false });
} finally {
  await tracer.close?.();
}
```

Close or flush the LangSmith tracer before a short-lived test process exits so
LangSmith receives the final root run patch.

If you prefer explicit values over environment variables, pass them to the SDK
tracer. `workspaceId` is optional and only selects a LangSmith workspace; it is
not the tracing project name.

```ts
import { RunTree } from "langsmith/run_trees";
import { createLangSmithContextTracer } from "agent-lattice";

const tracer = createLangSmithContextTracer({
  RunTree,
  apiKey: process.env.LANGSMITH_API_KEY,
  apiUrl: process.env.LANGSMITH_ENDPOINT,
  projectName: process.env.LANGSMITH_PROJECT,
  // Optional: only when LangSmith requires an explicit workspace.
  workspaceId: process.env.LANGSMITH_WORKSPACE_ID,
});
```

LangSmith receives one root `chain` run per SDK query, child `llm` runs for
model turns, child `tool` runs for SDK tool calls, and run events for auxiliary
trace events.

Custom sinks can implement the same interface for SQLite, OpenTelemetry, object
storage, or host-specific observability. The functions below are application
code you provide, not SDK exports:

```ts
const tracer = {
  async onEvent(event) {
    // TODO: Replace with your own storage/logging code.
  },
};
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
import { createAgent, tool } from "agent-lattice";
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

## Business Context For Tools

Pass host application data through `context`. The SDK gives that context to
tool handlers, but does not automatically put it into the model transcript. The
model sees the data only if a tool returns it.

```ts
import { createBareAgent, tool } from "agent-lattice";
import { z } from "zod/v4";

type QcContext = {
  patientRecordId: string;
  scoringStandardId: string;
};

const readPatientRecordInput = z.object({});
const qcTool = tool<QcContext>();

const readPatientRecord = qcTool(
  "read_patient_record",
  "Read the current patient record",
  readPatientRecordInput,
  async (_input, { context }) => {
    return {
      content: JSON.stringify({
        patientRecordId: context?.patientRecordId,
        scoringStandardId: context?.scoringStandardId,
      }),
    };
  },
);

const agent = createBareAgent<QcContext>({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  tools: [readPatientRecord],
});

const result = await agent.prompt("Review the current patient record.", {
  context: {
    patientRecordId: "ocr_123",
    scoringStandardId: "tumor-treatment-process-v1",
  },
});
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
import { createAgent, loadSkill, skill } from "agent-lattice";

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
} from "agent-lattice";

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
} from "agent-lattice";

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

## AgentLike Composition

`Agent` and `Team` satisfy the same `AgentLike` shape:

```ts
type AgentLike<TContext = unknown> = {
  query(prompt, options?): AsyncGenerator<SDKMessage | TeamRunnerMessage>;
  prompt(prompt, options?): Promise<SDKResultMessage>;
};
```

That means a team can be used anywhere a callable agent is expected. From the
outside, a team is an agent; inside, it can contain a whole organization.

## Team Mailbox Collaboration

Use `createTeam()` when you want to talk to one `AgentLike` while it coordinates
with named members internally. The team automatically injects member
`agentTool()` tools and drives the mailbox runtime when you call `team.query()`
or `team.prompt()`.

```ts
import {
  createAgent,
  createMemoryMailbox,
  createTeam,
  teamMember,
} from "agent-lattice";

const researcher = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  systemPrompt: "You research agent SDK architecture and report concise findings.",
});

const team = createTeam({
  name: "engineering",
  lead: createAgent({
    apiKey: process.env.DEEPSEEK_API_KEY,
    baseURL: "https://api.deepseek.com/anthropic",
    model: "deepseek-v4-flash",
    systemPrompt: "You lead engineering work. Delegate research tasks to researcher.",
  }),
  members: [
    teamMember({
      name: "researcher",
      role: "executor",
      focus: "Research agent architecture",
      agent: researcher,
    }),
  ],
  mailbox: createMemoryMailbox(),
});

for await (const event of team.query("Ask the researcher to inspect the SDK design.")) {
  console.log(event);
}
```

`team.query()` streams both the lead agent's normal SDK messages and team
runtime events such as `team_message`, `team_agent`, and nested `agent_message`
events. `team.prompt()` consumes that stream and returns only the final result.

`createTeam()` injects member AgentLike tools into the lead. Those tools expose
an explicit action contract: `mode: "ask"` waits for the member result,
`mode: "handoff"` returns an acceptance receipt to the lead while the team
runtime continues the accepted mailbox work, and `mode: "observe"` reports
unsupported unless a host runtime provides observation support. In the default
`team.query()` and `team.prompt()` path, a handoff receipt is not the final
delivery: the runtime waits for the member's upstream reply, feeds it back to
the lead, and keeps going until the root lead returns the final result or the
run terminates.

Team member tools can also request explicit shared workspace write grants:

```ts
for await (const event of team.query(
  "Ask backend to implement the API in the shared repo.",
  {
    permissions: {
      workspaceGrants: [{
        root: "/work/shared/txt-notebook-app",
        access: ["write"],
        reason: "Project shared workspace",
      }],
    },
  },
)) {
  console.log(event);
}
```

When the lead calls a member tool, it may include `workspaceGrants` scoped to
that member, for example `/work/shared/txt-notebook-app/backend`. The runtime
only accepts write grants that are covered by the caller's current permissions. The
accepted grants are written to mailbox metadata, included in the child agent's
task/system context, and enforced by the built-in write tools. Read-only tools
such as `Read`, `LS`, `Glob`, and `Grep` can inspect any path the host process
can read and do not require workspace grants. If a write tool is denied, the
model receives a structured `permission_denied` tool result with the requested
path, allowed roots, and a deterministic suggested next step.
Grant `access` values are operation categories, not tool names; `write` covers
`Write`, `Edit`, and obvious Bash writes.

Managers should choose one workspace strategy explicitly when delegating:
ask the member to write deliverables in its own private workspace and report
paths, or provide `workspaceGrants: [{ root, access: ["write"], reason }]` for
every shared or manager-owned root named as a write destination.

Advanced mailbox controls remain available through `team.send()`,
`team.drain()`, and `team.mailbox`. Member agents that can accept tools receive
`team_send`, `team_inbox`, `team_read`, `team_reply`, `team_followup`, and
`team_status` so they can process assigned mailbox work. The lead does not
receive raw mailbox tools by default; pass `exposeLeadMailboxTools: true` only
when the lead should manually operate the team mailbox.

For durable local storage, pass a SQLite-like database. `better-sqlite3` works
without the SDK taking a hard dependency on it:

```ts
import Database from "better-sqlite3";
import {
  createAgent,
  createSQLiteMailbox,
  createTeam,
} from "agent-lattice";

const mailbox = createSQLiteMailbox({
  database: new Database("team-mailbox.db"),
});

const team = createTeam({
  name: "engineering",
  lead: createAgent({
    apiKey: process.env.DEEPSEEK_API_KEY,
    baseURL: "https://api.deepseek.com/anthropic",
    model: "deepseek-v4-flash",
  }),
  members: [],
  mailbox,
});
```

Hosts can also provide their own `TeamMailbox` adapter for Redis, Cloudflare D1,
Durable Objects, or another queue/storage backend.

### Nested teams

Because `teamMember().agent` accepts any `AgentLike`, a `Team` can be a member
of another `Team`:

```ts
import {
  createAgent,
  createTeam,
  teamMember,
} from "agent-lattice";

const createDeepSeekAgent = (systemPrompt: string) => createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  systemPrompt,
});

const ceoAgent = createDeepSeekAgent(
  [
    "You are the CEO agent.",
    "Clarify product goals, decide which department owns the work, and ask for concise progress reports.",
    "Do not implement engineering details yourself.",
  ].join("\n"),
);
const engineeringHeadAgent = createDeepSeekAgent(
  [
    "You are the engineering head agent.",
    "Break engineering goals into backend and frontend work, route tasks to the right executor, and report outcomes upstream.",
    "Keep architecture decisions explicit.",
  ].join("\n"),
);
const backendAgent = createDeepSeekAgent(
  [
    "You are the backend executor agent.",
    "Handle APIs, data models, storage, integrations, and server-side correctness.",
    "Escalate product or UI decisions instead of guessing.",
  ].join("\n"),
);
const frontendAgent = createDeepSeekAgent(
  [
    "You are the frontend executor agent.",
    "Handle UI flows, client state, accessibility, and browser behavior.",
    "Escalate API contract questions instead of inventing them.",
  ].join("\n"),
);

const engineeringTeam = createTeam({
  name: "engineering",
  lead: engineeringHeadAgent,
  members: [
    teamMember({ name: "backend", role: "executor", agent: backendAgent }),
    teamMember({ name: "frontend", role: "executor", agent: frontendAgent }),
  ],
});

const companyTeam = createTeam({
  name: "company",
  lead: ceoAgent,
  members: [
    teamMember({
      name: "engineering",
      role: "head",
      focus: "Own engineering delivery",
      agent: engineeringTeam,
    }),
  ],
});
```

Use this pattern to model CEO -> Head Team -> Executor Agent without hard-coding
that hierarchy into the SDK.

### Routing loops

The SDK does not block routing loops by default. A task can move from a manager
to a member, back to the manager for context, and then back to the same member.
That is normal organizational flow, not necessarily a runtime error.

Use `maxTurns`, permission callbacks, mailbox status, and host-level monitoring
to control cost and risk. If your application needs a strict hierarchy, expose
only the allowed members at each layer and enforce routing with permission
callbacks or a host-level policy.

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

## Agent Workspace Tools

`createAgent()` includes a private workspace and these built-in tools by
default:

- `Read`
- `Write`
- `Edit`
- `LS`
- `Glob`
- `Grep`
- `Bash`

Use `createBuiltinTools()` or `createAgentWorkspaceTools()` when you want to
assemble the tool list yourself, especially with `createBareAgent()`:

```ts
import { createBareAgent, createBuiltinTools } from "agent-lattice";

const agent = createBareAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  systemPrompt: "Use the configured project directory for file work.",
  tools: createBuiltinTools({
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

`Read`, `LS`, `Glob`, and `Grep` are read-only observation tools and are not
gated by workspace grants. `Write`, `Edit`, and obvious Bash writes are gated to
the configured workspace roots and task-scoped shared workspace grants, so
production hosts should pair write and shell access with a permission callback.
Shell redirects to `/dev/null` are treated as discard targets, not workspace
writes.

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
