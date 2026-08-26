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

`Agent` is a type-only export — instances come from these factories (or
`AgentSpec.spawn()`), never from `new Agent()`, because the factories also
generate the session id and install the workspace. (Breaking in 0.17.0: the
`Agent` class constructor is no longer exported.)

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

`query()` yields these SDK messages:

| Type | When | What it carries |
| --- | --- | --- |
| `system` | Query start | Session init metadata (model, tools, `session_id`). |
| `stream_event` | While the model is responding | Raw provider stream event for incremental rendering. |
| `assistant` | After each model turn is assembled | The `AssistantModelMessage` with text / `tool_use` blocks and provider metadata. |
| `user` | After a whole tool batch finishes | Tool results as `ToolResultBlock[]`; the prompt is never echoed. |
| `result` | Once, at the end of the query | Final text, `subtype` (`"success"`, `"interrupted"`, or an error variant), optional `structuredResult`, and token usage. |

For the exact per-event guarantees see
[Streaming Events](https://docs.claude-code-sdk.com/concepts/streaming-events/).

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
`{ type: "disabled" }` to turn thinking off. A fixed
budget is capped at `maxTokens - 1` to satisfy the Anthropic API constraint.
When omitted, the SDK does not send a thinking configuration.

`{ type: "disabled" }` is sent to the provider explicitly as
`thinking: { "type": "disabled" }` rather than omitted, because some
Anthropic-compatible providers (for example DeepSeek's
`https://api.deepseek.com/anthropic` endpoint) default thinking to on —
omitting the field would leave it enabled.

On DeepSeek, the on/off switch is the only thinking control that works:
DeepSeek accepts `budget_tokens` but ignores the value, treats `adaptive` as
plain enabled thinking, and does not support `reasoning_effort` at all. Its own
thinking-strength knob is `output_config.effort`, which the SDK does not expose
yet. See
[Provider Compatibility](https://docs.claude-code-sdk.com/reference/provider-compatibility/)
for the full matrix.

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
`tool_result`, and `result`. When the model client reports token usage,
`assistant_message` events carry it as `data.message.usage`, and the `result`
event carries the query's summed usage as `data.usage` (since 0.23.1). For
team runners, pass the tracer per query to propagate it into delegated agents:

```ts
for await (const event of team.query("Ask engineering to investigate.", {
  tracer,
})) {
  console.log(event);
}
```

## LangSmith Context Tracing

The SDK depends on `langsmith` directly and uses its official `RunTree` /
`RunTreeConfig` types for this adapter, so the tracer works out of the box —
no constructor wiring needed.

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
import {
  createAgent,
  createCompositeContextTracer,
  createJsonlContextTracer,
  createLangSmithContextTracer,
} from "agent-lattice";

const tracer = createCompositeContextTracer([
  createJsonlContextTracer({ path: ".agent-runs/session.jsonl" }),
  createLangSmithContextTracer({
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
import { createLangSmithContextTracer } from "agent-lattice";

const tracer = createLangSmithContextTracer({
  apiKey: process.env.LANGSMITH_API_KEY,
  apiUrl: process.env.LANGSMITH_ENDPOINT,
  projectName: process.env.LANGSMITH_PROJECT,
  // Optional: only when LangSmith requires an explicit workspace.
  workspaceId: process.env.LANGSMITH_WORKSPACE_ID,
});
```

`RunTree` defaults to the bundled langsmith constructor since 0.17.0; pass
`RunTree` or `runTree` only to inject a custom runtime or a test fake.

LangSmith receives one root `chain` run per SDK query. For an `Agent` query,
model turns and SDK tool calls appear as child `llm` and `tool` runs. For a
`Team` query, the root represents the complete Team invocation; the initial
Lead run, delegated Member runs, and later Lead runs all appear beneath that
root and share one trace session. Each Agent keeps its own SDK session identity,
recorded as `agent_session_id` metadata, so tracing does not change Agent state
or returned SDK messages.

When the model client reports token usage, each `llm` run ends with
`usage_metadata` in its outputs (`input_tokens`, `output_tokens`,
`total_tokens`, plus cache buckets under `input_token_details`), so LangSmith
shows token consumption and inferred cost per model turn. Anthropic cache
tokens are additive, so they are summed into `input_tokens` the same way
LangSmith's own Anthropic wrapper does. *Requires 0.23.1 or later.*

## Langfuse Context Tracing

*Requires 0.19.0 or later.*

The Langfuse adapter targets the current Langfuse JS SDK generation
(`@langfuse/tracing` v5), which is OpenTelemetry-based. Register the
`LangfuseSpanProcessor` once at process startup, then create the tracer —
no other wiring needed.

Configure Langfuse with its standard environment variables:

```bash
LANGFUSE_PUBLIC_KEY=<your-langfuse-public-key>
LANGFUSE_SECRET_KEY=<your-langfuse-secret-key>
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com # or your self-hosted host
```

```bash
npm install @langfuse/otel @opentelemetry/sdk-trace-node
```

```ts
// instrumentation: register the span processor before agents run.
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

export const langfuseSpanProcessor = new LangfuseSpanProcessor();

const tracerProvider = new NodeTracerProvider({
  spanProcessors: [langfuseSpanProcessor],
});
tracerProvider.register();
```

```ts
import {
  createAgent,
  createCompositeContextTracer,
  createJsonlContextTracer,
  createLangfuseContextTracer,
} from "agent-lattice";
import { langfuseSpanProcessor } from "./instrumentation";

const tracer = createCompositeContextTracer([
  createJsonlContextTracer({ path: ".agent-runs/session.jsonl" }),
  createLangfuseContextTracer({
    // Drained by tracer.flush()/close() so spans reach Langfuse before a
    // short-lived process exits.
    spanProcessor: langfuseSpanProcessor,
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

Langfuse receives one trace per SDK query: the agent run is a root `chain`
observation carrying the trace name, session id, and tags; model turns appear
as child `generation` observations and SDK tool calls as child `tool`
observations. For a `Team` query, delegated runs nest as child `chain`
observations under the team root, so one handoff invocation stays one trace.

When the model client reports token usage, each `generation` observation ends
with `usageDetails` (`input`, `output`, `cache_creation_input_tokens`,
`cache_read_input_tokens`, `total`), so Langfuse shows token consumption and
inferred cost per model turn. Anthropic `input_tokens` already excludes cache
tokens, matching Langfuse's mutually-exclusive usage buckets. *Requires 0.23.1
or later.*

`startObservation` defaults to the bundled `@langfuse/tracing` function; pass
`startObservation` only to inject a custom runtime or a test fake.

Custom sinks can implement the same interface for SQLite, OpenTelemetry, object
storage, or host-specific observability. A `ContextTracer` port object exposes
methods only — `failOnError` is bound when the factory creates the tracer, not
set as a field afterwards. Implement a custom sink with `defineContextTracer()`.
*Requires 0.17.0 or later.* (Breaking in 0.17.0: the public `failOnError` field
was removed from `ContextTracer`; custom tracers created before 0.17.0 must go
through `defineContextTracer()`.)

```ts
import { defineContextTracer } from "agent-lattice";

const tracer = defineContextTracer({
  async onEvent(event) {
    // TODO: Replace with your own storage/logging code.
  },
});
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

Pass an explicit `deepseek-*` model name — unknown names are silently mapped to
`deepseek-v4-flash`. For which SDK options DeepSeek actually honors (thinking
budgets are ignored; `reasoningEffort` does not apply; structured output is not
supported), see
[Provider Compatibility](https://docs.claude-code-sdk.com/reference/provider-compatibility/).

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

A tool can also return `structuredResult` next to `endTurn: true` to carry a
structured payload to `SDKResultMessage.structuredResult`
(*requires 0.20.0 or later*). Without `endTurn`, `structuredResult` is
ignored.

## Structured Output Via submit_output

*Requires 0.20.0 or later.*

Set `AgentOptions.outputSchema` when a run must deliver a typed result rather
than free text. The SDK injects a built-in `submit_output` tool (exported as
`SUBMIT_OUTPUT_TOOL_NAME`) whose input schema is your schema converted to JSON
Schema — a zod schema works directly, since `OutputSchema<T>` is just
`{ parse(input: unknown): T }`:

```ts
import { createAgent } from "agent-lattice";
import { z } from "zod/v4";

const reviewSchema = z.object({
  approved: z.boolean(),
  issues: z.array(z.string()),
});

const reviewer = createAgent({
  model: "claude-sonnet-4-6",
  systemPrompt: "Review the change and submit your verdict.",
  outputSchema: reviewSchema,
});

const result = await reviewer.prompt("Review the patch in this workspace.");
if (result.subtype === "success") {
  console.log(result.structuredResult); // { approved: false, issues: [...] }
}
```

The structure is enforced by the harness, not by prompt discipline:

- The model submits its answer by calling `submit_output`. The payload is
  validated against the schema first; a validation failure goes back into the
  loop as an error `tool_result`, so the model can fix it and retry. A valid
  submission ends the run with `subtype: "success"` and the payload on
  `SDKResultMessage.structuredResult`.
- If the model ends its turn without calling `submit_output`, the run fails
  with `subtype: "error_missing_output"` and a `MissingOutputError`. There is
  no fallback that parses the final text as JSON.
- `submit_output` must be the only tool call in its batch. A batch that mixes
  it with other calls — or contains two submissions — is rejected with code
  `submit_output_exclusive_batch` and the loop continues.
- The name is reserved: registering your own `submit_output` tool while
  `outputSchema` is set throws from `createAgent`/`addTools`.

Unlike `outputFormat` (which relies on the provider's
`response_format`/`json_schema` support — DeepSeek ignores it, see
[Provider Compatibility](https://docs.claude-code-sdk.com/reference/provider-compatibility/)),
`submit_output` only requires a model that can call tools, so it ports to any
tool-capable provider.

The same schema composes with `agentTool()` for parent/child delegation: give
it to the child agent, and an `ask` call returns the child's validated output
as a JSON string. Since 0.21.0 the parent side inherits the child's declared
schema, so the `agentTool()` copy can be omitted. Passing
`AgentToolOptions.outputSchema` explicitly is still allowed, but it must match
the target's declaration (compared by reference, then by derived JSON Schema
structure) — a mismatch throws at assembly time, so drift fails fast instead
of at call time. The error message suggests sharing one schema instance or
omitting the `agentTool()` copy:

```ts
import { agentTool, createAgent } from "agent-lattice";

const child = createAgent({
  model: "claude-sonnet-4-5",
  systemPrompt: "You review code and submit a structured verdict.",
  outputSchema: reviewSchema, // child submits via submit_output
});

const parent = createAgent({
  model: "claude-sonnet-4-6",
  tools: [
    agentTool("review", child, {
      description: "Ask the reviewer to audit a change.",
      // outputSchema is inherited from the child (0.21.0+); an explicit copy
      // must match the child's declaration or agentTool() throws.
    }),
  ],
});
```

If the child ends without submitting or submits a payload that fails the
schema, the tool returns an `is_error` `tool_result` starting with
`child_output_invalid:`, so the parent model sees the failure and can retry.

*Behavior change in 0.21.0:* where the child declares an `outputSchema` and
the parent does not, the `ask` tool result changed from the fixed text
`"Structured output submitted."` to the validated JSON. That is the intended
fix and ships in a minor under 0.x. Host-defined `AgentLike` adapters carry no
readable declaration, so nothing is inherited or cross-checked for them; an
explicit `outputSchema` still applies. Whenever a schema is in effect, the
generated tool description states that the tool returns the target's
validated structured output as JSON.

Two more contract combinations are pinned down since 0.23.0:

- **Target declares no `outputSchema`, `agentTool()` declares one
  explicitly.** Assembly allows it (nothing to cross-check against), and an
  `ask` call validates the child's `structuredResult` against the parent-side
  schema and returns it as JSON. This fits children that end through a custom
  `endTurn` + `structuredResult` tool performing domain validation beyond the
  schema (for example reference truthfulness), with the contract declared by
  the parent alone.
- **Neither side declares a schema.** When the child ends with a
  `structuredResult`, the `ask` tool result is its JSON as-is (unvalidated)
  instead of falling back to the text content and dropping it. The trust
  level is the same as the text result; the schema's job is validation only,
  not gating the structured channel. This applies on both the direct path and
  the team runtime delegate path.

*Behavior change in 0.23.0:* existing code where the child ends with
`endTurn` + `structuredResult` and the parent declares no schema now receives
the structured JSON from `ask` instead of the text content.

### Typed delegation

*Requires 0.21.0 or later.*

`AgentToolOptions.inputSchema` replaces the default `{mode, task,
expectedOutput, acceptanceCriteria, workspaceGrants}` input shape with your
own schema (a zod schema works directly), and `mapInput` projects the
validated input into the child prompt:

```ts
const judge = createAgent({
  model: "claude-sonnet-4-5",
  systemPrompt: "You judge a case and submit a structured verdict.",
  outputSchema: verdictSchema,
});

const judgeTool = agentTool("judge", judge, {
  description: "Judge a case from its summary and documents.",
  inputSchema: z.object({
    caseSummary: z.string(),
    documents: z.array(z.object({ title: z.string(), content: z.string() })),
  }),
  mapInput: input => renderJudgeTask(input.caseSummary, input.documents),
});
```

- The parent's arguments are parsed against `inputSchema` before anything
  runs; invalid input is rejected as an error `tool_result` in the parent's
  loop — the same semantics as a plain `tool()` call — and the child is never
  invoked.
- `inputSchema` and `mapInput` must come as a pair: `agentTool()` throws at
  assembly time when one is missing.
- Typed delegation is ask-only: there is no `mode` field and no
  `workspaceGrants`.
- `mapInput` may return a string or `ContentBlock[]`; `ContentBlock[]` is only
  supported for direct `ask` calls — inside a team runtime the projected
  prompt must be a string, or the call fails at runtime.
- Because typed input no longer matches `AgentToolInput`, `agentTool()` now
  returns `ToolDefinition<any>`.

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

## Strict Option Validation And Tool Metadata

*Requires 0.22.0 or later.*

The option objects of `createAgent()`/`createBareAgent()`/`defineAgent()`
(`AgentOptions`), `agentTool()` (`AgentToolOptions`), `delegateTool()`
(`DelegateToolOptions`), and `tool()` (`ToolOptions`) are validated strictly:
an unknown key throws at assembly time —

```
AgentOptions: unknown option "bogusOption". Check for a typo, or upgrade the SDK if this option was added in a newer version.
```

(`agentTool()`/`delegateTool()` prefix the message with `agentTool("<name>"):` /
`delegateTool("<name>"):` instead.) The point is to fail fast on the old-SDK +
new-API combination: before 0.22.0 an unknown option was silently ignored, so
calling a newer API on an older install "worked" with the feature absent.

*Behavior change in 0.22.0:* extra keys that used to be silently ignored —
for example host fields spread into an options object — now throw. If your
host assembles options by spreading wider objects, strip the extra fields when
upgrading.

Separately, `ToolOptions.metadata` and `AgentToolOptions.metadata` accept a
`Record<string, unknown>` that is passed through to `ToolDefinition.metadata`:

```ts
const search = tool(
  "search",
  "Search documents",
  z.object({ query: z.string() }),
  async ({ query }) => ({ content: await documentIndex.search(query) }),
  { metadata: { contractVersion: 3 } },
);
```

The SDK never reads or interprets `metadata`, and it is never shown to the
model — it is host-owned, machine-readable annotation (for example a contract
version). When not passed, the key is absent from the `ToolDefinition`.

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
  interrupt(): boolean;
};
```

`interrupt()` ends the in-flight model request with an `"interrupted"` result
(see [Interrupting A Query](#interrupting-a-query)); on a `Team` or
`TeamRunner` it delegates to the lead/root agent. It returns `true` when a
query was interrupted and `false` when idle. *Requires 0.16.0 or later; the
`boolean` return requires 0.17.0 or later.*

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

Pass `workspace: false` to opt out of the built-in workspace entirely — no
built-in file/shell tools and no workspace prompt section, equivalent to
`createBareAgent()` (*requires 0.23.0 or later*). Unlike `createBareAgent()`,
the option also works through `defineAgent()`, so
`defineAgent({ workspace: false })` spawns bare sessions — handy for
typed-delegation specialists that should have no filesystem or shell surface
at all.

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
instance. To persist it or resume a conversation in another process, attach a
`HistoryStore` — see [Persistent History And Resume](#persistent-history-and-resume).

An `Agent` is a conversation, not a reusable client. Because the history is
instance state, starting a query while another is still running would interleave
both conversations; the SDK rejects the second one with `ConcurrentQueryError`.
Create one Agent per concurrent conversation — in a server, per request or per
user session rather than a shared module-level instance. Sequential reuse, as
above, is the intended pattern.

## Persistent History And Resume

*Requires 0.16.0 or later.*

Pass a `HistoryStore` to seed an Agent's history from durable storage and have
every later write mirrored back. `createJsonlHistoryStore()` persists one JSON
message per line:

```ts
import { createAgent, createJsonlHistoryStore } from "agent-lattice";

const historyStore = createJsonlHistoryStore({
  path: ".agent-sessions/ada.jsonl",
});

const agent = createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
  historyStore,
});

// The first query lazily loads any history the store already holds, then the
// new prompt continues from it. Every user, assistant, and tool_result message
// is appended to the file as it lands.
const result = await agent.prompt("What is my name?");

// A copy of the live history, safe to inspect or mutate.
const transcript = await agent.getHistory();
```

The store contract is three methods — `load()`, `append(message)`, and
`replace(messages)` — each synchronous or returning a promise:

- `load()` runs once per Agent lifetime, lazily before the first query (the
  constructor cannot be async). Resuming across processes is simply a new
  Agent over the same store; the "one Agent, one conversation" rule is
  unchanged.
- `append(message)` follows every message added to the history.
- `replace(messages)` follows compaction, which rewrites the whole history —
  a store must support full replacement, not just appends.

The JSONL store's `load()` skips malformed lines rather than failing, so a
torn final write does not lose the rest of the transcript. By default a
failing store is swallowed and the conversation continues in memory only,
mirroring the tracer's failure semantics; bind `failOnError: true` when the
store is created (via `createJsonlHistoryStore()` options or
`defineHistoryStore()`) to propagate store errors out of `query()` instead.

Implement a custom store with `defineHistoryStore()`, which validates the
three methods and binds `failOnError`; the returned port object exposes
methods only. *Requires 0.17.0 or later.* (Breaking in 0.17.0: the public
`failOnError` field was removed from `HistoryStore`; custom stores created
before 0.17.0 must go through `defineHistoryStore()`.)

```ts
import { defineHistoryStore } from "agent-lattice";

const historyStore = defineHistoryStore({
  load: () => loadMessagesFromYourDatabase(),
  append: message => appendMessageToYourDatabase(message),
  replace: messages => replaceMessagesInYourDatabase(messages),
});
```

To rewrite the history from the host instead of from compaction, use
`agent.replaceHistory(messages)`. *Requires 0.17.0 or later.* It is idle-only:
calling it while a query is running throws `ConcurrentQueryError`, the same
guard as a concurrent `query()`. When a `historyStore` is configured the store
is replaced too, so persistence stays in sync, and the replacement also
suppresses the lazy `load()` — seeding never overwrites what the host just
installed. The SDK does not validate the content: the host owns it, and the
history must be well-formed (e.g. no dangling `tool_use` without its matching
`tool_result`).

```ts
await agent.replaceHistory([
  { role: "user", content: "My name is Ada." },
  { role: "assistant", content: [{ type: "text", text: "Nice to meet you, Ada." }] },
]);
```

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

*Requires 0.16.0 or later; `interrupt()` returns `boolean` from 0.17.0.*

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
ends `"interrupted"` before the next model call. `interrupt()` returns `true`
when a query was running and is now interrupted, and `false` when idle — a
`false` tells the host there is nothing to wait for, so it can send its next
query directly.

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

When a response containing tool calls is truncated at `max_tokens`, the SDK
does not execute those calls: the last `tool_use` input may be incomplete, and
a truncated value can even survive JSON parsing with its meaning changed.
Every call in the batch gets an error `tool_result` explaining the truncation
and asking the model to reissue the call with a shorter output, and the loop
continues. *Requires 0.18.0 or later.*

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
