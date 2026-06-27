# AgentLike Recursive Tool Runtime Design

Date: 2026-06-27

## Purpose

Design a small, composable runtime model for `claude-team-agent-sdk` where every callable intelligent unit shares one `AgentLike` interface and can be invoked either directly by an application or recursively through model tool use.

The goal is to make this flow natural:

```text
User -> AgentLike A -> tool_use(agentTool B) -> AgentLike B -> tool_use(agentTool C) -> AgentLike C
```

without making `Team`, mailbox tools, or supervisor-specific APIs special cases in the public mental model.

## Current Signals From Real Testing

The external live test harness at `/Users/duzicong/code/duzicong/claude-code-sdk-test` showed that nested teams can finish successfully, but the intermediate trace exposed design pressure:

- Lead agents receive too many low-level mailbox tools.
- `manager` is local to a team, but models can misread it as the global parent or CEO.
- Message ids repeat across nested teams, so `message_id: "first"` is ambiguous to humans and models.
- Synchronous delegation returns a tool result and also leaves an upstream mailbox report, causing duplicated context.
- `team_inbox` returns full message bodies by default, causing context growth.
- `team_send(to:self)` can create accidental work that is later claimed before the intended new task.
- Mailbox mutations created by tools are not always emitted as first-class team events, reducing replay/debug fidelity.

The new design must keep the success path while narrowing the footguns.

## Core Abstraction

`AgentLike` is the only required callable shape:

```ts
export type AgentLike = {
  query(input: AgentInput, options?: AgentRunOptions): AsyncGenerator<AgentEvent>;
  prompt(input: AgentInput, options?: AgentRunOptions): Promise<AgentResult>;
};
```

These must satisfy `AgentLike`:

- a single model-backed `Agent`
- a `Team`
- a nested `Team`
- a remote or host-provided adapter
- a future worker-backed mailbox runtime

This invariant should hold:

> If something is an `AgentLike`, the application can talk to it directly, and another agent can call it as a tool.

## Tool Bridge

`agentTool()` is the adapter from model tool use to an `AgentLike` call:

```ts
const engineering = createTeam(...);

const ceo = createAgent({
  tools: [
    agentTool("engineering", engineering, {
      description: "Delegate engineering delivery.",
      wait: "result",
    }),
  ],
});
```

`agentTool()` creates a normal model tool, but its handler delegates execution to another `AgentLike`.

The handler must not contain team-specific logic. It should call the child through a runtime context so the same recursion works for agents, teams, and adapters.

## Runtime Context

Recursive execution is owned by the runtime, not by tool handlers:

```ts
export type AgentRunContext = {
  traceId: string;
  path: string[];
  depth: number;
  maxDepth: number;
  signal?: AbortSignal;
  stream?: boolean;
  eventSink: AgentEventSink;
  mailbox?: TeamMailbox;
};
```

When `agentTool("backend", backendAgent)` is called from `company.engineering`, the child context path becomes:

```text
company.engineering.backend
```

Every emitted event should carry this source path or an equivalent structured source object. This is required for logs, UI rendering, and debugging.

## Wait Modes

`agentTool()` supports three execution modes:

```ts
type AgentToolWaitMode = "result" | "accepted" | "detached";
```

`wait: "result"` is the default. The parent model blocks until the child `AgentLike` returns a final result, and that result becomes the tool result.

`wait: "accepted"` creates or records work and immediately returns an acceptance payload. A mailbox runner or host service can continue the task.

`wait: "detached"` starts background work and returns a lightweight handle. This is for advanced hosts that own lifecycle, retries, and observation.

MVP should implement `result` cleanly first and keep the other two modes as explicit extension points if they are already present.

## Team As Composition

`createTeam()` should primarily be a composition helper:

```ts
const engineering = createTeam({
  name: "engineering",
  lead: engineeringLead,
  members: [
    teamMember({ name: "backend", role: "executor", agent: backend }),
    teamMember({ name: "frontend", role: "executor", agent: frontend }),
  ],
});
```

Internally, members become `agentTool()` tools on the lead:

```ts
lead.addTools([
  agentTool("backend", backend, ...),
  agentTool("frontend", frontend, ...),
]);
```

From the outside, `engineering` is still one `AgentLike`.

## Mailbox Role

Mailbox is a runtime coordination ledger, not the default public mental model.

Default `createTeam()` should avoid exposing all low-level mailbox tools to every lead. The default lead surface should be:

- member delegation tools
- optional status/summary tools if needed

Low-level tools such as raw `team_send`, `team_reply`, `team_followup`, `team_read`, and `team_inbox` should be either:

- reserved for mailbox worker mode
- scoped to members that are actively processing mailbox messages
- exposed only through an explicit advanced option

This avoids models using mailbox as a confusing chat system.

## Parent Reporting

Agents should not have to guess whether `manager` means local team manager or global parent.

When an `AgentLike` is invoked through `agentTool(wait: "result")`, parent reporting should usually be automatic:

1. child returns final result
2. runtime converts it to the parent tool result
3. optional trace event records the delegation result

For mailbox worker mode, use explicit parent semantics such as:

```ts
team_report({ content: "..." })
```

or a runtime-scoped reply method that knows the active parent message. Avoid requiring models to address `"manager"` manually.

## Inbox Behavior

`team_inbox` should default to summaries:

```text
id: msg_9
from: engineering::frontend
status: pending
thread: third
preview: "Here is the index.html..."
```

Full content should require `team_read(message_id)`.

This protects context budgets and makes model behavior less noisy.

## Event Model

Every recursive call and mailbox mutation should produce structured events:

```ts
type AgentInvocationEvent = {
  type: "agent_invocation";
  subtype: "started" | "finished" | "failed";
  source: AgentSource;
  target?: AgentSource;
};

type AgentToolEvent = {
  type: "agent_tool";
  subtype: "called" | "result" | "accepted" | "failed";
  source: AgentSource;
  tool: string;
  target?: AgentSource;
};

type MailboxEvent = {
  type: "team_message";
  subtype: "sent" | "claimed" | "read" | "replied" | "followup" | "done" | "failed";
  source: AgentSource;
  mailbox: string;
  message: TeamMessage;
};
```

Token-level model stream events can still exist, but `stream: false` must suppress token deltas while preserving invocation, tool, mailbox, assistant, and result events.

## Budgets And Cycles

Cycles are allowed. A task can move from `A -> B -> A` when context or ownership requires it.

The runtime should prevent runaway execution through budgets, not hard topology bans:

- `maxTurns`
- `maxDepth`
- `maxDelegations`
- `timeoutMs`
- `AbortSignal`

Errors should preserve the trace path that failed.

## Public API Target

The preferred public shape is:

```ts
const backend = createAgent(...);
const frontend = createAgent(...);

const engineering = createTeam({
  name: "engineering",
  lead: createAgent(...),
  members: [
    teamMember({ name: "backend", role: "executor", agent: backend }),
    teamMember({ name: "frontend", role: "executor", agent: frontend }),
  ],
});

const company = createAgent({
  tools: [
    agentTool("engineering", engineering, {
      description: "Own engineering delivery.",
    }),
  ],
});

for await (const event of company.query("Build a small text notebook app.", { stream: false })) {
  console.log(event);
}
```

`createTeam()` can remain a convenience for nesting:

```ts
const companyTeam = createTeam({
  name: "company",
  lead: ceo,
  members: [
    teamMember({ name: "engineering", role: "head", agent: engineering }),
  ],
});
```

Both `company` and `companyTeam` are `AgentLike`.

## Migration Plan

1. Introduce or rename the public bridge to `agentTool()`.
2. Rebuild existing team member delegate tools on top of `agentTool()`.
3. Narrow default `createTeam()` mailbox tool exposure.
4. Make `team_inbox` return summaries by default.
5. Add explicit parent-report semantics for mailbox worker mode.
6. Emit events for all mailbox mutations, including tool-created messages.
7. Add regression tests based on the real log issues.
8. Update docs around AgentLike, recursive tool use, teams, and streaming.

## Test Plan

Unit tests:

- `AgentLike` single agent still supports `query()` and `prompt()`.
- `agentTool()` calls a child `AgentLike` and returns child result as tool result.
- nested `agentTool()` calls preserve source path.
- `createTeam()` exposes members as agent tools.
- lead does not receive raw mailbox tools by default.
- `team_inbox` summary does not include full long content.
- mailbox worker mode can still read and reply to assigned messages.
- all mailbox sends/replies/followups emit events.
- cycles are allowed within budget and fail with a clear budget error when exceeded.

Integration tests:

- run the external DeepSeek team harness with `{ stream: false }`
- assert final result success
- assert no `team_reply does not belong to manager` error in the normal happy path
- assert logs remain compact enough to inspect

## Non-Goals

- Do not hard-code CEO -> Head -> Executor into the SDK core.
- Do not forbid cyclic delegation by default.
- Do not require users to understand mailbox concepts for basic teams.
- Do not make Team a separate execution model from AgentLike.

## Spec Self-Review

- No TBD or placeholder sections remain.
- The design keeps `AgentLike` as the core public invariant.
- Team is reduced to composition plus optional mailbox coordination.
- Runtime recursion, not tool handler special casing, owns nested execution.
- The design addresses every issue observed in the live team log without forcing strict hierarchy.

