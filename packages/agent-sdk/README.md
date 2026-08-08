# AgentLattice

AgentLattice is a TypeScript framework for building coordinated agent systems
with tools, skills, tracing, supervisor delegation, and mailbox-backed teams.

Install it from npm as `agent-lattice`. It works with Anthropic and
Anthropic-compatible providers such as DeepSeek, without installing the Claude
Code CLI runtime.

> **Integrating this SDK from an AI agent?** Start at
> <https://docs.claude-code-sdk.com/llms.txt> for an index of the documentation,
> where every page is served as clean Markdown. Read
> <https://docs.claude-code-sdk.com/llms-full.txt> to take it all in one request.

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

`agent.query()` yields `stream_event` messages while the model is still
responding, so a host can render output incrementally:

```ts
for await (const message of agent.query("Say hello")) {
  if (message.type === "stream_event") {
    const event = message.event as { type: string; delta?: { text?: string } };
    if (event.type === "content_block_delta" && event.delta?.text) {
      process.stdout.write(event.delta.text);
    }
  }
}
```

The SDK produces the next event only after the loop takes the current one.
Slow work in the loop body delays later events without dropping or reordering
them, so keep expensive handling off the loop itself.

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

For Kimi K3 through an Anthropic-compatible endpoint or gateway, use
`reasoningEffort` to send the provider's top-level `reasoning_effort` parameter:

```ts
const agent = createAgent({
  apiKey: process.env.MOONSHOT_API_KEY,
  baseURL: process.env.KIMI_ANTHROPIC_BASE_URL,
  model: "kimi-k3",
  reasoningEffort: "high",
});

const result = await agent.prompt("Solve this carefully.", {
  reasoningEffort: "low",
});
```

The supported values are `"low"`, `"high"`, and `"max"`. A query-level value
overrides the agent default. When omitted, the SDK does not send
`reasoning_effort`, so the provider applies its own default (`"max"` for Kimi
K3). Kimi K3 does not accept `thinkingConfig`; use `reasoningEffort` instead.
The Kimi Open Platform endpoint at `https://api.moonshot.cn/v1` uses the OpenAI
Chat Completions protocol and is not a valid `baseURL` for the SDK's built-in
Anthropic client.

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

LangSmith receives one root `chain` run per SDK query. For an `Agent` query,
model turns and SDK tool calls appear as child `llm` and `tool` runs. For a
`Team` query, the root represents the complete Team invocation; the initial
Lead run, delegated Member runs, and later Lead runs all appear beneath that
root and share one trace session. Each Agent keeps its own SDK session identity,
recorded as `agent_session_id` metadata, so tracing does not change Agent state
or returned SDK messages.

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

## End The Run From A Tool

*Requires 0.16.0 or later.*

A tool that has the final answer can end the run itself by returning
`endTurn: true`. The SDK finishes with `subtype: "success"` and uses that
tool's text content as the result, without calling the model again:

```ts
const finish = tool(
  "finish",
  "Submit the final answer and end the run",
  z.object({ answer: z.string() }),
  async ({ answer }) => ({ content: answer, endTurn: true }),
);
```

`endTurn` does not cancel the other tools of the same batch — they already
started concurrently, their `tool_result` blocks still enter the history, and
`onToolResult` hooks and trace events run for them as usual. Only the next
model call is skipped. When several tools in a batch set `endTurn`, the first
one's content becomes the result text.

## Concurrent Tool Calls

The model requests concurrency by returning multiple `tool_use` blocks in one
assistant message. The SDK makes the final safety decision. By default, only
tools whose parsed input passes `isConcurrencySafe(input)` run together:

```ts
const search = tool(
  "search",
  "Search documents",
  z.object({ query: z.string() }),
  async ({ query }) => {
    // App code: replace with your database or search client.
    return { content: await documentIndex.search(query) };
  },
  { isConcurrencySafe: () => true },
);

const agent = createAgent({
  model: "claude-sonnet-4-6",
  tools: [search],
  toolConcurrency: { mode: "safe", maxConcurrency: 8 },
});
```

`safe` is the default mode, `maxConcurrency` defaults to `10`, and tools without
an `isConcurrencySafe` declaration stay sequential. Use `mode: "all"` only when
every tool in the Agent is safe to overlap. Use `mode: "sequential"` to disable
tool concurrency even for tools marked safe.

When concurrency is available, the SDK tells the model to batch independent
calls and to use separate assistant responses when a later call needs an earlier
result. Runtime safety checks and `toolBatchPolicy` remain authoritative.

The SDK waits for the complete batch before requesting the model again. Tools
may finish in any order, while the `tool_result` blocks sent to the model remain
in the original `tool_use` order. One tool failure does not discard the other
results. On abort, running handlers receive the shared `AbortSignal`, queued
handlers do not start, and the SDK waits for handlers that already started to
settle.

## Tool Batch Policy

Use `toolBatchPolicy` when some tools must not run in the same model response.
The policy sees the complete batch before any tool executes. If it rejects the
batch, no tool runs and every tool call receives a structured `is_error: true`
result.

```ts
const lead = createAgent({
  model: "claude-sonnet-4-6",
  tools: [incrementRevision],
  toolBatchPolicy: {
    validate({ toolCalls }) {
      const incrementsRevision = toolCalls.find(
        call => call.name === "incrementRevision",
      );
      const handoff = toolCalls.find(
        call => call.kind === "agent_tool" &&
          (call.input as { mode?: string }).mode === "handoff",
      );
      if (incrementsRevision && handoff) {
        return {
          allowed: false,
          code: "invalid_tool_batch",
          message: "Update the revision before delegating dependent work.",
          conflictingToolCallIds: [incrementsRevision.id, handoff.id],
          suggestedNextStep: "Run incrementRevision first, then hand off the new revision.",
        };
      }
      return { allowed: true };
    },
  },
});
```

The policy may be synchronous or asynchronous. If it throws, the SDK rejects
the whole batch with `tool_batch_policy_error`; no tool has executed. Without a
policy, tool execution is unchanged. A policy prevents known bad combinations
inside one model response, but it does not replace database transactions or
revision checks against concurrent external updates.

## Automatic Context Compaction

History only grows, so a long-running agent eventually exceeds the model's
context window. Enable `autoCompact` to replace the older part of the
conversation with a model-written summary:

```ts
const agent = createAgent({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: "claude-sonnet-4-6",
  autoCompact: true, // or { thresholdTokens: 150_000, keepRecentMessages: 8 }
});
```

Compaction runs between turns, once a response reports more input tokens than
`thresholdTokens` (default `100000`). Everything except the last
`keepRecentMessages` messages (default `6`) is summarized, and the history is
rebuilt as that summary followed by the retained messages. The summary is
wrapped in an instruction telling the model that compaction just happened and to
continue from it, so the next turn resumes the task instead of restarting it.

Unlike the `onModelRequest` hook, which shapes a single request, this **rewrites
the stored conversation** — that is what makes the saving persist, but the
replaced turns are gone.

The cut point never separates a `tool_result` from the `tool_use` that produced
it, because the model API rejects that. If no safe cut leaves anything to
summarize, compaction is skipped.

Compaction costs a model call. Its tokens are folded into `result.usage`, and a
`system` message with `subtype: "compaction"` reports what happened:

```ts
for await (const message of agent.query("Refactor this module.")) {
  if (message.type === "system" && message.subtype === "compaction") {
    console.log(`compacted ${message.compacted_messages} messages`, message.usage);
  }
}
```

The trigger depends on reported usage, so a custom `ModelClient` that omits
`usage` never compacts. Override the instruction with `prompt`, or read the
built-in one from `DEFAULT_COMPACTION_PROMPT`.

The threshold is a forecast, so a single large tool result can still carry a
request past the window. Compaction then runs as a recovery — summarize, then
retry the same turn — on either `stop_reason: "model_context_window_exceeded"`
or an API error naming a too-long prompt. It is attempted once per query; if
summarizing fails or there is nothing left to summarize, the original failure
surfaces unchanged.

`stop_reason: "max_tokens"` deliberately does **not** trigger compaction. It
means the *output* hit `maxTokens`, not that the input was too large — the model
had room to read and ran out of room to write, so compacting the history would
not make the answer complete. Raise `maxTokens` instead.

## Hooks

`permission` and `toolBatchPolicy` decide whether something runs. Hooks decide
what it looks like — redacting tool output, trimming context before a request,
or injecting retrieved documents:

```ts
const agent = createAgent({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: "claude-sonnet-4-6",
  tools: [queryDatabase],
  hooks: {
    async onToolResult({ toolName, result, error }) {
      if (toolName !== "queryDatabase") return; // undefined: leave unchanged
      return { ...result, content: await redact(result.content) };
    },
    onModelRequest({ messages, turn }) {
      if (messages.length < 40) return;
      return { messages: compact(messages) };
    },
  },
});
```

`onToolResult` sees every result on its way to the model, including handler
failures, aborted calls, and calls blocked by `toolBatchPolicy`. `onModelRequest`
shapes a single request; the stored conversation is untouched, so trimming
context does not destroy history.

A hook returns a replacement or nothing, and must not mutate what it receives. A
hook that throws propagates out of `query()` rather than becoming an error
`result` — a redaction hook that failed quietly would leak the data it exists to
protect. Hooks run before the matching trace event, so traces record what was
actually sent.

Compose independent concerns with `createCompositeAgentHooks([a, b, c])`, which
chains them in order, each receiving the previous one's output.

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

## Agent Specs (Templates) And Sessions

*Requires 0.15.0 or later.*

`createAgent()` returns a live session: one conversation, one history, one
workspace. `defineAgent()` returns an `AgentSpec` — a template carrying the
same options but no state. `spawn()` creates an independent session from it:

```ts
import { agentTool, defineAgent } from "agent-lattice";

const reviewerSpec = defineAgent({
  name: "reviewer",
  model: "claude-sonnet-4-5",
  systemPrompt: "You are a senior code reviewer...",
});

// Register the spec: every tool call spawns a fresh session with no memory
// of previous calls. This is the safe default for reuse.
const lead = createAgent({
  model: "claude-sonnet-4-5",
  tools: [
    agentTool("review", reviewerSpec, {
      description: "Ask the reviewer to audit a change.",
    }),
  ],
});

// Register a spawned session instead when the target should remember earlier
// tasks across calls — continuity is an explicit opt-in.
const reviewSession = reviewerSpec.spawn();
```

The same union applies to `delegateTool()`. The generated tool description
states which semantics a target has, so the calling agent knows whether each
task must be self-contained. Existing code that passes an `AgentLike` keeps
its current behavior: a long-lived session with history.

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

After one model response queues one or more handoffs, the SDK does not call the
lead model again immediately. It first runs those members, collects their
completed or failed reports, and only then calls the lead again. All handoffs
from the current tool batch are queued before the lead pauses. The receipt keeps
`status: "accepted"` for compatibility and also includes `phase: "queued"`,
`completion_pending: true`, `message_id`, `work_item_id`, and `thread_id`.

Accepted handoffs use work-item failure isolation by default. If one member
returns an agent error such as `MaxTurnsError` or `APIError`, the runtime marks
that work item `failed`, sends a failure report to the lead, and continues the
other accepted handoffs. The lead receives successful and failed reports
together and decides whether to retry, revise the task, accept a partial result,
or finish. A run-wide `AbortError` still stops the runner; the runtime marks the
current and remaining accepted work `cancelled` before propagating the abort.

Handoff work is serial by default. Set a bounded concurrency limit on the team
when independent members should run at the same time:

```ts
const team = createTeam({
  name: "research",
  lead,
  members,
  runner: { maxConcurrentWorkItems: 4 },
});
```

You can also pass `maxConcurrentWorkItems` to `createTeamRunner()`. The limit
must be a positive integer and defaults to `1`. It applies across different
member mailboxes; work addressed to the same mailbox remains serial because an
Agent may keep mutable conversation state. Runtime events are emitted as work
actually progresses, while the reports injected back into the lead stay in the
original handoff order.

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

An `Agent` is a conversation, not a reusable client. Because the history is
instance state, starting a query while another is still running would interleave
both conversations; the SDK rejects the second one with `ConcurrentQueryError`.
Create one Agent per concurrent conversation — in a server, per request or per
user session rather than a shared module-level instance. Sequential reuse, as
above, is the intended pattern.

## Deadlines

`QueryOptions.signal` bounds a whole query — every model request, tool call, and
turn together. `requestTimeoutMs` bounds each single model request, so an agent
that legitimately runs many tool-using turns does not have to fit them all into
one budget:

```ts
const agent = createAgent({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: "claude-sonnet-4-6",
  requestTimeoutMs: 120_000,
});

const result = await agent.prompt("Audit this repository.", {
  signal: AbortSignal.timeout(600_000),
  requestTimeoutMs: 60_000, // overrides the agent default for this query
});
```

A request deadline produces `subtype: "error_timeout"` with a `TimeoutError`,
distinct from the `"error_abort"` of a caller-initiated cancellation, so hosts
can retry timeouts without retrying deliberate cancellations.

Both limits are enforced by the SDK rather than delegated. `ModelRequest` carries
`signal` and `timeoutMs` so a client can cancel its own work, but the agent loop
also races the call, so a `ModelClient` that honours neither cannot stall the
loop indefinitely. Losing that race abandons the call rather than cancelling it.

## Interrupting A Query

*Requires 0.16.0 or later.*

`agent.interrupt()` ends the current query without tearing the conversation
down. Where `QueryOptions.signal` terminates the query with `"error_abort"`,
`interrupt()` aborts only the in-flight model request and finishes with
`subtype: "interrupted"` — normal control flow, so `is_error` stays `false`.
As on abort, the partial assistant message is dropped, but every completed
turn stays in the history, so the host can continue the same Agent with a new
`query()` that injects its own message:

```ts
const pending = agent.prompt("Draft the release notes.");
agent.interrupt(); // e.g. the user typed a correction
const result = await pending;
// result.subtype === "interrupted"

await agent.prompt("Actually, skip 0.15.x and cover 0.16.0 only.");
```

An interrupt that lands while a tool batch is executing takes effect once the
batch completes: its tool results are written to history first, and the query
ends `"interrupted"` before the next model call. `interrupt()` is a no-op when
no query is running.

## Token Usage And Truncation

Every `result` message reports `usage`, summed over the model requests in that
query, plus the `stop_reason` of the last response:

```ts
const result = await agent.prompt("Summarize this file.");
console.log(result.usage);
// { input_tokens: 1200, output_tokens: 512, cache_read_input_tokens: 800 }

if (result.stop_reason === "max_tokens") {
  // subtype is still "success", but result is a fragment, not an answer.
}
```

`stop_reason: "max_tokens"` means the model hit its output budget mid-response.
The SDK does not treat that as an error, so checking this field is the only way
to distinguish a complete answer from a truncated one.

Usage comes from the model client. The built-in Anthropic client fills it in from
the response, including the streaming path; a custom `ModelClient` that omits
`usage` produces zeroed counts rather than an error.

Assistant messages also carry provider response metadata: `providerResponseId`
is the provider-assigned response id and `model` is the model that actually
served the response, which may differ from the requested `AgentOptions.model`.
The built-in Anthropic client fills both in on streaming and non-streaming
requests; a custom `ModelClient` may set them on the `AssistantModelMessage` it
returns. Both fields are optional and absent when the client does not report
them.

*Requires 0.16.0 or later.*
