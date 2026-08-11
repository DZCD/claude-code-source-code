import Anthropic from "@anthropic-ai/sdk";
import type {
  DocumentBlockParam,
  ImageBlockParam,
  TextBlockParam,
} from "@anthropic-ai/sdk/resources/messages.mjs";
import type { BetaJSONOutputFormat } from "@anthropic-ai/sdk/resources/beta/messages/messages.mjs";
import { Client as MCPProtocolClient } from "@modelcontextprotocol/sdk/client/index.js";
import type { OAuthClientProvider } from "@modelcontextprotocol/sdk/client/auth.js";
import {
  StdioClientTransport,
  type StdioServerParameters,
} from "@modelcontextprotocol/sdk/client/stdio.js";
import {
  StreamableHTTPClientTransport,
  type StreamableHTTPClientTransportOptions,
} from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type {
  RunEvent,
  RunTree,
  RunTreeConfig,
} from "langsmith/run_trees";
import { execFile } from "node:child_process";
import { mkdirSync } from "node:fs";
import { appendFile, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import {
  dirname,
  basename,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from "node:path";
import { toJSONSchema } from "zod/v4";
import { z } from "zod/v4";

export type TextBlock = TextBlockParam;
export type ImageBlock = ImageBlockParam;
export type DocumentBlock = DocumentBlockParam;
export type ToolUseBlock = {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
};
export type ToolResultBlock = {
  type: "tool_result";
  tool_use_id: string;
  content: string | ContentBlock[];
  is_error?: boolean;
};

type ToolExecutionOutcome = {
  block: ToolResultBlock;
  error?: Error;
  endTurn?: boolean;
};
export type ThinkingBlock = {
  type: "thinking";
  thinking: string;
  signature: string;
};
export type RedactedThinkingBlock = {
  type: "redacted_thinking";
  data: string;
};
export type ContentBlock =
  | TextBlock
  | ImageBlock
  | DocumentBlock
  | ToolUseBlock
  | ToolResultBlock
  | ThinkingBlock
  | RedactedThinkingBlock;

export type AgentLikeEvent = SDKMessage | TeamRunnerMessage;

export type AgentLike<TContext = unknown> = {
  query(prompt: string | ContentBlock[], options?: QueryOptions<TContext>): AsyncGenerator<AgentLikeEvent>;
  prompt(prompt: string | ContentBlock[], options?: QueryOptions<TContext>): Promise<SDKResultMessage>;
  /**
   * End the in-flight model request with an "interrupted" result; follow up
   * with a new query(). Returns `true` when a query was interrupted, `false`
   * when idle — a `false` means the host can send its next query directly.
   */
  interrupt(): boolean;
};

export type DelegateWaitMode = "result" | "accepted";

export type AgentRuntimeSource = {
  kind: "root" | "team_member" | "agent";
  name?: string;
  team?: string;
  member?: string;
  mailbox?: string;
};

export type AgentRuntimeDelegateInput = {
  name: string;
  description?: string;
  agent: AgentLike<any>;
  task: string;
  wait?: DelegateWaitMode;
  targetMailboxId?: string;
  workspaceGrants?: WorkspaceGrantInput[];
};

export type AgentRuntimeDelegateResult = {
  status: "completed" | "accepted" | "failed";
  content: string;
  request: TeamMessage;
  reply?: TeamMessage;
  result?: SDKResultMessage;
  error?: AgentRuntimeFailure;
  workspaceGrants?: WorkspaceGrant[];
};

export type AgentRuntimeFailure = {
  code:
    | "max_turns_exceeded"
    | "api_error"
    | "tool_execution_error"
    | "permission_denied"
    | "agent_error";
  message: string;
  name: string;
};

// Access values are operation categories, not tool names. "write" covers
// Write, Edit, and obvious shell writes.
export type WorkspaceAccess = "read" | "write" | "execute";

export type WorkspaceGrantInput = {
  root: string;
  access?: WorkspaceAccess[];
  reason?: string;
  expiresAt?: number;
};

export type WorkspaceGrant = WorkspaceGrantInput & {
  kind: "workspace";
  root: string;
  access: WorkspaceAccess[];
  grantor?: AgentRuntimeSource;
  grantee?: AgentRuntimeSource;
  workItemId?: string;
};

export type RuntimePermissions = {
  workspaceGrants?: WorkspaceGrantInput[];
};

export type PermissionDenialReasonCode =
  | "outside_allowed_roots"
  | "grant_not_authorized"
  | "expired_grant";

export type PermissionDenial = {
  status: "permission_denied";
  tool: string;
  operation: WorkspaceAccess;
  requestedPath?: string;
  reasonCode: PermissionDenialReasonCode;
  reason: string;
  allowedRoots: string[];
  suggestedNextStep: string;
};

export type AgentRuntimeContext = {
  source: AgentRuntimeSource;
  permissions: RuntimePermissions;
  delegate(input: AgentRuntimeDelegateInput): Promise<AgentRuntimeDelegateResult>;
  emit(message: TeamRunnerMessage): void;
  shouldPauseAfterToolBatch?(): boolean;
};

export type ContextTraceEventType =
  | "run_start"
  | "user_message"
  | "model_request"
  | "assistant_message"
  | "tool_use"
  | "tool_result"
  | "team_message"
  | "result"
  | "compaction"
  | "error";

export type ContextTraceEvent = {
  version: 1;
  timestamp: string;
  session_id: string;
  run_id: string;
  parent_run_id?: string;
  seq: number;
  source: AgentRuntimeSource;
  type: ContextTraceEventType;
  data: Record<string, unknown>;
};

/**
 * Ports (ContextTracer, HistoryStore) expose methods only. The `failOnError`
 * flag is bound by the factory that creates the port and carried as internal
 * metadata, so a port object handed to the SDK never mixes configuration with
 * its callable surface.
 */
const kFailOnError = Symbol("agent-lattice.failOnError");

function bindFailOnError<TPort extends object>(port: TPort, failOnError: boolean | undefined): TPort {
  if (failOnError) {
    Object.assign(port, { [kFailOnError]: true });
  }
  return port;
}

function shouldPropagatePortError(port: object): boolean {
  return (port as Record<symbol, unknown>)[kFailOnError] === true;
}

export type ContextTracer = {
  onEvent(event: ContextTraceEvent): Promise<void> | void;
  flush?(): Promise<void>;
  close?(): Promise<void>;
};

export type JsonlContextTracerOptions = {
  path?: string;
  dir?: string;
  redact?: (event: ContextTraceEvent) => ContextTraceEvent | undefined;
  failOnError?: boolean;
};

export type LangSmithKVMap = Record<string, unknown>;

export type LangSmithRunTreeConfig = RunTreeConfig;
export type LangSmithRunEvent = RunEvent;
export type LangSmithRunTreeLike = RunTree;
export type LangSmithRunTreeConstructor = new (config: LangSmithRunTreeConfig) => LangSmithRunTreeLike;
export type LangSmithWriteReplicaConfig = NonNullable<LangSmithRunTreeConfig["replicas"]>[number];

export type LangSmithContextTracerOptions = {
  RunTree?: LangSmithRunTreeConstructor;
  runTree?: (config: LangSmithRunTreeConfig) => LangSmithRunTreeLike;
  projectName?: string;
  name?: string;
  apiUrl?: string;
  apiKey?: string;
  workspaceId?: string;
  client?: LangSmithRunTreeConfig["client"];
  tags?: string[];
  metadata?: LangSmithKVMap;
  redact?: (event: ContextTraceEvent) => ContextTraceEvent | undefined;
  failOnError?: boolean;
};

type ToolCapableAgentLike<TContext = unknown> = AgentLike<TContext> & {
  addTools(tools: Array<ToolDefinition<any, TContext>>): void;
};

export type ModelMessage = {
  role: "user" | "assistant";
  content: string | ContentBlock[];
};

/**
 * Persistence adapter for an Agent's conversation history. `load()` runs once
 * per Agent lifetime, before the first query; `append()` follows every message
 * added to the history; `replace()` follows compaction, which rewrites the
 * whole history — a store must support full replacement. Methods may be
 * synchronous or return a promise; the Agent awaits them to preserve order.
 * Like `ContextTracer`, a failing store is swallowed by default and the
 * conversation continues in memory only; pass `failOnError` to the factory
 * that creates the store (`createJsonlHistoryStore`, `defineHistoryStore`)
 * to propagate store errors out of `query()` instead.
 *
 * The objects passed to `append()` and `replace()` are guaranteed to carry at
 * least `role` and `content`. Runtime objects may carry extra fields (`usage`,
 * `stopReason`, `providerResponseId`, `model`) that are not part of the
 * contract and must not be relied on; hosts that need response metadata for
 * reconciliation or analytics should use the ContextTracer's
 * `assistant_message` events instead.
 */
export type HistoryStore = {
  load(): ModelMessage[] | Promise<ModelMessage[]>;
  append(message: ModelMessage): void | Promise<void>;
  replace(messages: ModelMessage[]): void | Promise<void>;
};

export type JsonlHistoryStoreOptions = {
  path: string;
  failOnError?: boolean;
};

export type TokenUsage = {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
};

/**
 * Why the model stopped. `"max_tokens"` means the response was cut off mid-way:
 * the text is a fragment, not an answer. Left open because providers add values.
 */
export type StopReason =
  | "end_turn"
  /** Output hit `maxTokens`. The text is a fragment; compaction does not help. */
  | "max_tokens"
  | "stop_sequence"
  | "tool_use"
  | "pause_turn"
  | "refusal"
  /** The context window ran out mid-request. This is what compaction is for. */
  | "model_context_window_exceeded"
  | (string & {});

export type AssistantModelMessage = {
  role: "assistant";
  content: ContentBlock[];
  /** Absent when a custom ModelClient does not report it. */
  usage?: TokenUsage;
  stopReason?: StopReason;
  /** The provider-assigned response id. Absent when a custom ModelClient does not report it. */
  providerResponseId?: string;
  /**
   * The model that actually served this response, which may differ from
   * `ModelRequest.model`. Absent when a custom ModelClient does not report it.
   */
  model?: string;
};

export type ModelToolDefinition = {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
};

export type ModelRequest = {
  model: string;
  systemPrompt?: string;
  maxTokens: number;
  messages: ModelMessage[];
  tools: ModelToolDefinition[];
  stream: boolean;
  outputFormat?: OutputFormat;
  thinkingConfig?: ThinkingConfig;
  reasoningEffort?: ReasoningEffort;
  /** Deadline for this single request. Clients should honour it; the SDK also enforces it. */
  timeoutMs?: number;
  onStreamEvent?: (event: Record<string, unknown>) => void;
  signal?: AbortSignal;
};

export interface ModelClient {
  createMessage(request: ModelRequest): Promise<AssistantModelMessage>;
}

export type ToolResult = {
  content: string | ContentBlock[];
  /** End the run after this tool batch: finish with subtype "success" instead of calling the model again. */
  endTurn?: boolean;
};

export type ToolKind = "tool" | "agent_tool";

export type ToolBatchCall = {
  id: string;
  name: string;
  input: unknown;
  kind: ToolKind;
};

export type ToolBatchPolicyContext<TContext = unknown> = {
  source?: AgentRuntimeSource;
  toolCalls: ToolBatchCall[];
  context?: TContext;
  signal?: AbortSignal;
};

export type ToolBatchPolicyRejection = {
  allowed: false;
  code: string;
  message: string;
  conflictingToolCallIds?: string[];
  suggestedNextStep?: string;
};

export type ToolBatchPolicyResult = { allowed: true } | ToolBatchPolicyRejection;

export type ToolBatchPolicy<TContext = unknown> = {
  validate(
    context: ToolBatchPolicyContext<TContext>,
  ): ToolBatchPolicyResult | Promise<ToolBatchPolicyResult>;
};

export type ToolExecutionContext<TContext = unknown> = {
  signal?: AbortSignal;
  toolUseId: string;
  context?: TContext;
  agentRuntime?: AgentRuntimeContext;
  permissions?: RuntimePermissions;
};

export type ToolResultHookContext<TContext = unknown> = {
  toolName: string;
  toolUseId: string;
  /** Raw input as the model sent it, before the tool schema parsed it. */
  input: unknown;
  /** What the SDK would send back to the model. */
  result: ToolResultBlock;
  /** Present when the handler failed, was denied, or was cancelled. */
  error?: Error;
  context?: TContext;
  source?: AgentRuntimeSource;
  signal?: AbortSignal;
};

export type ModelRequestHookContext<TContext = unknown> = {
  /** What the SDK would send. Not the stored history; see AgentHooks. */
  messages: ModelMessage[];
  systemPrompt?: string;
  /** 1 for the first model request of the query. */
  turn: number;
  context?: TContext;
  source?: AgentRuntimeSource;
  signal?: AbortSignal;
};

export type ModelRequestHookResult = {
  messages?: ModelMessage[];
  systemPrompt?: string;
};

export const DEFAULT_COMPACTION_PROMPT = `Your task is to create a detailed summary of the conversation so far,
paying close attention to the user's explicit requests and your previous actions.

This summary should be thorough in capturing:
- technical details
- code patterns
- architectural decisions
- files that were modified
- commands that were run
- errors encountered
- solutions attempted
- important context needed to continue the work

Preserve:
- user's intent
- important constraints
- decisions already made
- reasoning behind decisions
- unresolved issues
- next steps

The summary will replace the conversation history, so include everything
necessary for another Claude instance to continue the task successfully.

Output only the summary.`;

export type AutoCompactOptions = {
  /**
   * Compact once a model response reports more input tokens than this.
   * Defaults to 100000, chosen to leave headroom on a 200k-token model.
   */
  thresholdTokens?: number;
  /** Trailing messages left verbatim after the summary. Defaults to 6. */
  keepRecentMessages?: number;
  /** Replaces DEFAULT_COMPACTION_PROMPT. */
  prompt?: string;
  /** Model used for the summary. Defaults to the agent's model. */
  model?: string;
  /** Output cap for the summary. Defaults to 8192. */
  maxTokens?: number;
};

/**
 * Lifecycle callbacks that can rewrite what crosses the agent loop's boundaries,
 * as opposed to `permission` and `toolBatchPolicy`, which can only allow or deny.
 *
 * A hook returns a replacement, or nothing to leave the value unchanged; it must
 * not mutate what it receives.
 *
 * A hook that throws propagates out of `query()` rather than being swallowed the
 * way a tracer error is, or being reported as an error `result`. A hook failure
 * is host code failing, like `ConcurrentQueryError`, not an upstream failure the
 * loop can describe to the model — and a redaction hook that failed quietly
 * would leak the data it exists to protect.
 *
 * Hooks run before the matching trace event, so traces record what was actually
 * sent. `onModelRequest` shapes one request only and never edits the stored
 * conversation, so trimming context for a long turn does not destroy history.
 */
export type AgentHooks<TContext = unknown> = {
  onToolResult?(
    context: ToolResultHookContext<TContext>,
  ): ToolResultBlock | void | Promise<ToolResultBlock | void>;
  onModelRequest?(
    context: ModelRequestHookContext<TContext>,
  ): ModelRequestHookResult | void | Promise<ModelRequestHookResult | void>;
};

export type ToolHandler<TInput = unknown, TContext = unknown> = (
  input: TInput,
  context: ToolExecutionContext<TContext>,
) => Promise<ToolResult> | ToolResult;

export type ToolOptions<TInput = unknown> = {
  isConcurrencySafe?: (input: TInput) => boolean;
};

export type ToolDefinition<TInput = unknown, TContext = unknown> = {
  name: string;
  description: string;
  kind?: ToolKind;
  inputSchema: unknown;
  jsonSchema: Record<string, unknown>;
  parse(input: unknown): TInput;
  handler: ToolHandler<TInput, TContext>;
  isConcurrencySafe?: (input: TInput) => boolean;
};

export type ToolConcurrencyMode = "safe" | "all" | "sequential";

export type ToolConcurrencyOptions = {
  mode?: ToolConcurrencyMode;
  maxConcurrency?: number;
};

export type SkillDefinition = {
  name: string;
  description: string;
  instructions: string;
  path?: string;
};

export type SkillInput = {
  name: string;
  description: string;
  instructions: string;
  path?: string;
};

export type MCPContentBlock =
  | { type: "text"; text: string }
  | { type: string; [key: string]: unknown };

export type MCPTool = {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
};

export type MCPListToolsResult = {
  tools: MCPTool[];
};

export type MCPCallToolResult = {
  content?: MCPContentBlock[];
  structuredContent?: unknown;
  isError?: boolean;
};

export type MCPClient = {
  listTools(): Promise<MCPListToolsResult>;
  callTool(input: { name: string; arguments?: Record<string, unknown> }): Promise<MCPCallToolResult>;
};

export type MCPToolsOptions = {
  namePrefix?: string;
};

export type MCPStdioServerOptions = MCPToolsOptions & {
  clientName?: string;
  clientVersion?: string;
};

export type MCPStdioConnection = {
  client: MCPClient;
  tools: Array<ToolDefinition<any, any>>;
  close(): Promise<void>;
};

export type MCPStreamableHTTPServerOptions = MCPToolsOptions & {
  clientName?: string;
  clientVersion?: string;
  authProvider?: OAuthClientProvider;
  requestInit?: RequestInit;
  fetch?: StreamableHTTPClientTransportOptions["fetch"];
  reconnectionOptions?: StreamableHTTPClientTransportOptions["reconnectionOptions"];
  sessionId?: string;
};

export type MCPStreamableHTTPConnection = MCPStdioConnection & {
  finishAuth(authorizationCode: string): Promise<void>;
  terminateSession(): Promise<void>;
  sessionId: string | undefined;
};

export type TeamMemberRole = "lead" | "head" | "executor";

export type TeamMemberDefinition = {
  name: string;
  role: TeamMemberRole;
  focus?: string;
  agent: AgentLike<any>;
  mailboxId?: string;
};

export type TeamMemberInput = TeamMemberDefinition;

export type TeamMessageStatus = "pending" | "processing" | "done" | "failed" | "cancelled" | "read";

export type TeamMessageRole =
  | "upstream_request"
  | "delegation"
  | "executor_result"
  | "downstream_reply"
  | "upstream_report"
  | "followup"
  | "message";

export type TeamMessage = {
  id: string;
  from: string;
  to: string;
  content: string;
  status: TeamMessageStatus;
  createdAt: number;
  threadId: string;
  parentMessageId?: string;
  workItemId?: string;
  workItemRole?: TeamMessageRole;
  upstreamMessageId?: string;
  metadata?: Record<string, unknown>;
};

export type TeamSendOptions = {
  threadId?: string;
  parentMessageId?: string;
  workItemId?: string;
  workItemRole?: TeamMessageRole;
  upstreamMessageId?: string;
  metadata?: Record<string, unknown>;
};

export type TeamMailbox = {
  send(from: string, to: string, content: string, options?: TeamSendOptions): Promise<TeamMessage>;
  inbox(mailboxId: string, options?: { status?: TeamMessageStatus | "all" }): Promise<TeamMessage[]>;
  get(messageId: string): Promise<TeamMessage | undefined>;
  claimNext(mailboxId: string): Promise<TeamMessage | undefined>;
  updateStatus(messageId: string, status: TeamMessageStatus): Promise<boolean>;
};

export type SQLiteStatementLike = {
  run(...params: unknown[]): unknown;
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
};

export type SQLiteDatabaseLike = {
  prepare(sql: string): SQLiteStatementLike;
  exec?(sql: string): unknown;
};

export type SQLiteMailboxOptions = {
  database: SQLiteDatabaseLike;
  tableName?: string;
};

export type TeamRunnerConfig = {
  maxDelegateDepth?: number;
  maxConcurrentWorkItems?: number;
};

export type TeamOptions = {
  name: string;
  lead: Agent<any>;
  members: TeamMemberDefinition[];
  mailbox?: TeamMailbox;
  exposeLeadMailboxTools?: boolean;
  runner?: TeamRunnerConfig;
};

export type Team = {
  name: string;
  lead: Agent<any>;
  members: TeamMemberDefinition[];
  mailbox: TeamMailbox;
  tools: Array<ToolDefinition<any, any>>;
  memberTools: Record<string, Array<ToolDefinition<any, any>>>;
  readonly runnerOptions?: TeamRunnerConfig;
  send(from: string, to: string, content: string, options?: TeamSendOptions): Promise<TeamMessage>;
  drain(options?: TeamDrainOptions): Promise<TeamDrainResult>;
  query(prompt: string | ContentBlock[], options?: QueryOptions): AsyncGenerator<TeamRunnerMessage>;
  prompt(prompt: string | ContentBlock[], options?: QueryOptions): Promise<SDKResultMessage>;
  /** Interrupts the lead agent's in-flight model request; returns `true` when a query was interrupted, `false` when idle. */
  interrupt(): boolean;
};

export type TeamDrainOptions = {
  maxRounds?: number;
  maxMessages?: number;
  signal?: AbortSignal;
};

export type TeamDrainResult = {
  processed: number;
  failed: number;
  rounds: number;
};

export type TeamRunnerOptions = TeamRunnerConfig & {
  team?: Team;
  root?: AgentLike<any>;
  mailbox?: TeamMailbox;
  source?: AgentRuntimeSource;
};

export type TeamRunner = {
  root: AgentLike<any>;
  mailbox: TeamMailbox;
  query(prompt: string | ContentBlock[], options?: QueryOptions): AsyncGenerator<TeamRunnerMessage>;
  prompt(prompt: string | ContentBlock[], options?: QueryOptions): Promise<SDKResultMessage>;
  /** Interrupts the root agent's in-flight model request; returns `true` when a query was interrupted, `false` when idle. */
  interrupt(): boolean;
};

type AcceptedDelegateWork = {
  delegateInput: AgentRuntimeDelegateInput;
  request: TeamMessage;
  callerMailbox: string;
  targetMailbox: string;
  callerSource: AgentRuntimeSource;
  targetSource: AgentRuntimeSource;
  workspaceGrants: WorkspaceGrant[];
  depth: number;
};

export type PermissionRequest = {
  toolName: string;
  input: Record<string, unknown>;
  toolUseId: string;
};

export type PermissionDecision =
  | { behavior: "allow" }
  | { behavior: "deny"; message: string };

export type AgentWorkspaceOptions =
  | string
  | {
      cwd: string;
      allowedDirectories?: string[];
      bashTimeoutMs?: number;
    };

export type AgentOptions<TContext = unknown> = {
  apiKey?: string;
  baseURL?: string;
  name?: string;
  model: string;
  systemPrompt?: string;
  maxTokens?: number;
  maxTurns?: number;
  thinkingConfig?: ThinkingConfig;
  reasoningEffort?: ReasoningEffort;
  /** Deadline for each single model request, in milliseconds. Unset means no SDK-side limit. */
  requestTimeoutMs?: number;
  tools?: Array<ToolDefinition<any, TContext>>;
  toolBatchPolicy?: ToolBatchPolicy<TContext>;
  /** Lifecycle callbacks that rewrite tool results and outgoing model requests. */
  hooks?: AgentHooks<TContext>;
  /**
   * Replaces older conversation history with a model-written summary once it
   * grows past a threshold. Off unless set; `true` uses the defaults.
   */
  autoCompact?: boolean | AutoCompactOptions;
  toolConcurrency?: ToolConcurrencyOptions;
  skills?: SkillDefinition[];
  workspace?: AgentWorkspaceOptions;
  permission?: (request: PermissionRequest) => Promise<PermissionDecision> | PermissionDecision;
  modelClient?: ModelClient;
  tracer?: ContextTracer;
  /**
   * Persistence adapter for the conversation history. Seeded via `load()` once
   * before the first query, then notified on every history write. Compaction
   * calls `replace()`, so the store must support full replacement.
   */
  historyStore?: HistoryStore;
};

export type BareAgentOptions<TContext = unknown> = Omit<AgentOptions<TContext>, "workspace">;

const BARE_AGENT_OPTIONS = Symbol("bareAgentOptions");

type InternalAgentOptions<TContext = unknown> = AgentOptions<TContext> & {
  [BARE_AGENT_OPTIONS]?: {
    installWorkspace?: boolean;
  };
};

export type AgentWorkspaceToolsOptions = {
  cwd?: string;
  allowedDirectories?: string[];
  bashTimeoutMs?: number;
};

export type QueryOptions<TContext = unknown> = {
  stream?: boolean;
  outputFormat?: OutputFormat;
  thinkingConfig?: ThinkingConfig;
  reasoningEffort?: ReasoningEffort;
  /** Overrides the agent's per-request deadline for this query. */
  requestTimeoutMs?: number;
  signal?: AbortSignal;
  context?: TContext;
  agentRuntime?: AgentRuntimeContext;
  permissions?: RuntimePermissions;
  tracer?: ContextTracer;
};

type QueryTraceContext = {
  sessionId: string;
  parentRunId: string;
};

const QUERY_TRACE_CONTEXT = Symbol("queryTraceContext");

type InternalQueryOptions<TContext = unknown> = QueryOptions<TContext> & {
  [QUERY_TRACE_CONTEXT]?: QueryTraceContext;
};

export type OutputFormat = "json" | BetaJSONOutputFormat;

export type ThinkingConfig =
  | { type: "adaptive" }
  | { type: "enabled"; budgetTokens: number }
  | { type: "disabled" };

export type ReasoningEffort = "low" | "high" | "max";

export type SDKSystemInitMessage = {
  type: "system";
  subtype: "init";
  model: string;
  tools: string[];
  session_id: string;
};

export type SDKSystemCompactionMessage = {
  type: "system";
  subtype: "compaction";
  session_id: string;
  /** Messages replaced by the summary. */
  compacted_messages: number;
  /** Messages kept verbatim after it. */
  retained_messages: number;
  /** Input tokens of the turn that triggered compaction. */
  trigger_input_tokens: number;
  /** Cost of producing the summary, already folded into the result usage. */
  usage: TokenUsage;
};

export type SDKAssistantMessage = {
  type: "assistant";
  message: AssistantModelMessage;
  session_id: string;
};

/**
 * Emitted only after a whole tool batch finishes — one event per batch, never
 * one per tool, and never for the prompt (the prompt is not echoed; subscribe
 * a `ContextTracer` for a full transcript). `message.content` is always
 * `ToolResultBlock[]`; the declared `ModelMessage` type is wider than this
 * guarantee. `tool_use_result` is a convenience view: a single tool's result
 * `content`, or the array of result blocks when the batch had several tools.
 */
export type SDKUserMessage = {
  type: "user";
  message: ModelMessage;
  session_id: string;
  tool_use_result?: string | ContentBlock[];
  error?: Error;
};

export type SDKStreamEventMessage = {
  type: "stream_event";
  event: Record<string, unknown>;
  session_id: string;
};

export type SDKResultMessage = {
  type: "result";
  /**
   * `"interrupted"` is not an error: `Agent.interrupt()` ends the query this
   * way, keeping completed turns in history so a follow-up query can continue
   * the conversation. `is_error` stays `false` for it.
   */
  subtype: "success" | "interrupted" | "error" | "error_max_turns" | "error_abort" | "error_timeout";
  is_error: boolean;
  result: string;
  session_id: string;
  num_turns: number;
  error?: Error;
  /** Summed over every model request in the query. Zeroed when unreported. */
  usage: TokenUsage;
  /**
   * From the last model response. Check for `"max_tokens"`: `subtype` is still
   * `"success"` there, but `result` is a truncated fragment.
   */
  stop_reason?: StopReason;
};

export type SDKMessage =
  | SDKSystemInitMessage
  | SDKSystemCompactionMessage
  | SDKStreamEventMessage
  | SDKAssistantMessage
  | SDKUserMessage
  | SDKResultMessage;

export type TeamRunnerSource = AgentRuntimeSource;

export type TeamRunnerTeamMessage = {
  type: "team_message";
  subtype: "sent" | "claimed" | "replied" | "followup" | "done" | "failed" | "cancelled";
  source: TeamRunnerSource;
  mailbox: string;
  message: TeamMessage;
  error?: AgentRuntimeFailure;
};

export type TeamRunnerAgentMessage = {
  type: "agent_message";
  source: TeamRunnerSource;
  message: SDKMessage;
};

export type TeamRunnerAgentLifecycleMessage = {
  type: "team_agent";
  subtype: "started" | "finished";
  source: TeamRunnerSource;
};

export type TeamRunnerStreamEventMessage = {
  type: "stream_event";
  source: TeamRunnerSource;
  event: Record<string, unknown>;
  session_id: string;
};

export type TeamRunnerMessage =
  | SDKMessage
  | TeamRunnerTeamMessage
  | TeamRunnerAgentMessage
  | TeamRunnerAgentLifecycleMessage
  | TeamRunnerStreamEventMessage;

export class AgentSDKError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = new.target.name;
    this.cause = options?.cause;
  }
}

export class APIError extends AgentSDKError {}
export class ToolExecutionError extends AgentSDKError {}
export class MaxTurnsError extends AgentSDKError {}
export class AbortError extends AgentSDKError {}
/** A second query was started on an Agent that was still running one. */
export class ConcurrentQueryError extends AgentSDKError {}
/** A model request exceeded `requestTimeoutMs`. */
export class TimeoutError extends AgentSDKError {}

export class ToolBatchRejectedError extends AgentSDKError {
  readonly rejection: ToolBatchPolicyRejection;

  constructor(rejection: ToolBatchPolicyRejection) {
    super(rejection.message);
    this.rejection = rejection;
  }
}

export class ToolPermissionDeniedError extends AgentSDKError {
  readonly denial: PermissionDenial;

  constructor(denial: PermissionDenial) {
    super(denial.reason);
    this.denial = denial;
  }
}

export function tool<TContext = unknown>(): <TSchema>(
  name: string,
  description: string,
  inputSchema: TSchema,
  handler: ToolHandler<InferInput<TSchema>, TContext>,
  options?: ToolOptions<InferInput<TSchema>>,
) => ToolDefinition<InferInput<TSchema>, TContext>;

export function tool<TSchema, TContext = unknown>(
  name: string,
  description: string,
  inputSchema: TSchema,
  handler: ToolHandler<InferInput<TSchema>, TContext>,
  options?: ToolOptions<InferInput<TSchema>>,
): ToolDefinition<InferInput<TSchema>, TContext>;

export function tool<TSchema, TContext = unknown>(
  name?: string,
  description?: string,
  inputSchema?: TSchema,
  handler?: ToolHandler<InferInput<TSchema>, TContext>,
  options?: ToolOptions<InferInput<TSchema>>,
):
  | ToolDefinition<InferInput<TSchema>, TContext>
  | (<TFactorySchema>(
    name: string,
    description: string,
    inputSchema: TFactorySchema,
    handler: ToolHandler<InferInput<TFactorySchema>, TContext>,
    options?: ToolOptions<InferInput<TFactorySchema>>,
  ) => ToolDefinition<InferInput<TFactorySchema>, TContext>) {
  if (name === undefined) {
    return <TFactorySchema>(
      factoryName: string,
      factoryDescription: string,
      factoryInputSchema: TFactorySchema,
      factoryHandler: ToolHandler<InferInput<TFactorySchema>, TContext>,
      factoryOptions?: ToolOptions<InferInput<TFactorySchema>>,
    ): ToolDefinition<InferInput<TFactorySchema>, TContext> =>
      createToolDefinition(factoryName, factoryDescription, factoryInputSchema, factoryHandler, factoryOptions);
  }

  if (description === undefined || inputSchema === undefined || handler === undefined) {
    throw new Error("tool(name, description, inputSchema, handler) requires all arguments");
  }

  return createToolDefinition(name, description, inputSchema, handler, options);
}

function createToolDefinition<TSchema, TContext = unknown>(
  name: string,
  description: string,
  inputSchema: TSchema,
  handler: ToolHandler<InferInput<TSchema>, TContext>,
  options?: ToolOptions<InferInput<TSchema>>,
): ToolDefinition<InferInput<TSchema>, TContext> {
  return {
    name,
    description,
    inputSchema,
    jsonSchema: schemaToJSONSchema(inputSchema),
    parse(input: unknown): InferInput<TSchema> {
      return parseWithSchema(inputSchema, input) as InferInput<TSchema>;
    },
    handler,
    ...(options?.isConcurrencySafe ? { isConcurrencySafe: options.isConcurrencySafe } : {}),
  };
}

export type DelegateToolOptions = {
  wait?: DelegateWaitMode;
  targetMailboxId?: string;
  workspaceGrants?: WorkspaceGrantInput[];
};

export type AgentToolMode = "ask" | "handoff" | "observe";

export const agentToolInputSchema = z.object({
  mode: z.enum(["ask", "handoff", "observe"]),
  task: z.string(),
  expectedOutput: z.string().optional(),
  acceptanceCriteria: z.array(z.string()).optional(),
  workspaceGrants: z.array(z.object({
    root: z.string(),
    access: z.array(z.enum(["write"])).optional(),
    reason: z.string().optional(),
    expiresAt: z.number().optional(),
  })).optional(),
});

export type AgentToolInput = z.infer<typeof agentToolInputSchema>;

export type AgentToolOptions = {
  description: string;
  targetMailboxId?: string;
};

/**
 * An AgentLike is a live session: it keeps its conversation history across
 * calls. An AgentSpec is a template: each call spawns a fresh session with no
 * memory of previous calls. Prefer a spec unless the parent explicitly wants
 * continuity.
 */
export type AgentToolTarget = AgentLike<any> | AgentSpec<any>;

export function agentTool(
  name: string,
  agent: AgentToolTarget,
  options: AgentToolOptions,
): ToolDefinition<AgentToolInput> {
  const toolName = sanitizeToolName(name);
  const definition = tool(
    toolName,
    formatAgentToolDescription(options.description, isAgentSpec(agent)),
    agentToolInputSchema,
    async (input, toolContext) => {
      const task = formatAgentToolTask(input);
      // Resolve per invocation: a spec spawns a new session every call, so no
      // history leaks between unrelated tasks.
      const target = resolveAgentTarget(agent);
      if (input.mode === "observe") {
        throw new Error(`agentTool("${toolName}") mode=observe is not supported. Available modes: ask, handoff.`);
      }

      if (input.workspaceGrants?.length && !toolContext.agentRuntime) {
        throw new Error(`agentTool("${toolName}") workspaceGrants require an AgentRuntime so grants can be authorized and reported.`);
      }

      if (input.mode === "handoff") {
        if (!toolContext.agentRuntime) {
          throw new Error(`agentTool("${toolName}") mode=handoff requires an AgentRuntime. Available modes without AgentRuntime: ask.`);
        }
        const result = await toolContext.agentRuntime.delegate({
          name: toolName,
          description: options.description,
          agent: target,
          task,
          wait: "accepted",
          targetMailboxId: options.targetMailboxId,
          workspaceGrants: input.workspaceGrants,
        });
        return { content: result.content };
      }

      if (toolContext.agentRuntime) {
        const result = await toolContext.agentRuntime.delegate({
          name: toolName,
          description: options.description,
          agent: target,
          task,
          wait: "result",
          targetMailboxId: options.targetMailboxId,
          workspaceGrants: input.workspaceGrants,
        });
        return { content: result.content };
      }

      const result = await target.prompt(task, {
        signal: toolContext.signal,
        context: toolContext.context,
      });
      if (result.is_error) {
        throw result.error ?? new Error(result.result || `agentTool("${toolName}") target returned an error`);
      }
      return { content: result.result };
    },
  );
  return { ...definition, kind: "agent_tool" };
}

export function delegateTool(
  name: string,
  description: string,
  agent: AgentToolTarget,
  options: DelegateToolOptions = {},
): ToolDefinition<{ task: string }> {
  const toolName = sanitizeToolName(name);
  const definition = tool(
    toolName,
    description,
    z.object({
      task: z.string(),
    }),
    async (input, toolContext) => {
      if (!toolContext.agentRuntime) {
        throw new Error(`delegateTool("${toolName}") requires an AgentRuntime. Use createTeamRunner().query() or createTeamRunner().prompt().`);
      }
      const result = await toolContext.agentRuntime.delegate({
        name: toolName,
        description,
        agent: resolveAgentTarget(agent),
        task: input.task,
        wait: options.wait,
        targetMailboxId: options.targetMailboxId,
        workspaceGrants: options.workspaceGrants,
      });
      return { content: result.content };
    },
  );
  return { ...definition, kind: "agent_tool" };
}

export function createAgent<TContext = unknown>(options: AgentOptions<TContext>): Agent<TContext> {
  return new Agent<TContext>(options);
}

export function createBareAgent<TContext = unknown>(options: BareAgentOptions<TContext>): Agent<TContext> {
  return new Agent<TContext>({
    ...options,
    [BARE_AGENT_OPTIONS]: { installWorkspace: false },
  } as InternalAgentOptions<TContext>);
}

/**
 * A template describing an agent's identity: model, prompt, tools, skills,
 * and workspace policy. A spec carries no conversation state; `spawn()`
 * creates an independent session (an Agent) that owns its own history and
 * workspace. Register a spec wherever a capability should be reused without
 * leaking memory between tasks; spawn a session when continuity is wanted.
 */
export type AgentSpec<TContext = unknown> = {
  readonly name?: string;
  readonly options: AgentOptions<TContext>;
  spawn(overrides?: Partial<AgentOptions<TContext>>): Agent<TContext>;
};

export function defineAgent<TContext = unknown>(options: AgentOptions<TContext>): AgentSpec<TContext> {
  if (!options?.model) {
    throw new Error("AgentOptions.model is required");
  }
  const specOptions = Object.freeze({ ...options });
  return {
    name: options.name,
    options: specOptions,
    spawn(overrides) {
      return new Agent<TContext>({ ...specOptions, ...overrides });
    },
  };
}

export function isAgentSpec(target: AgentLike<any> | AgentSpec<any>): target is AgentSpec<any> {
  return typeof (target as AgentSpec<any>).spawn === "function";
}

function resolveAgentTarget(target: AgentLike<any> | AgentSpec<any>): AgentLike<any> {
  return isAgentSpec(target) ? target.spawn() : target;
}

export function createBuiltinTools(options: AgentWorkspaceToolsOptions = {}): Array<ToolDefinition<any, any>> {
  return createAgentWorkspaceTools(options);
}

const DEFAULT_AGENT_WORKSPACE_ROOT = join(homedir(), ".agent", "workspaces");

type NormalizedAgentWorkspace = {
  cwd: string;
  toolsOptions: AgentWorkspaceToolsOptions;
};

function applyAgentWorkspaceOptions<TContext>(options: AgentOptions<TContext>, sessionId: string): AgentOptions<TContext> {
  const workspace = normalizeAgentWorkspaceOptions(options, sessionId);
  mkdirSync(workspace.cwd, { recursive: true });
  const workspaceTools = createAgentWorkspaceTools(workspace.toolsOptions);
  return {
    ...options,
    systemPrompt: joinPromptSections([
      options.systemPrompt,
      formatAgentWorkspaceInstructions(workspace.cwd),
    ]),
    tools: mergeToolsByName(workspaceTools, options.tools ?? []),
  };
}

function normalizeAgentWorkspaceOptions<TContext>(options: AgentOptions<TContext>, sessionId: string): NormalizedAgentWorkspace {
  const workspace = options.workspace;
  if (!workspace) {
    const name = options.name?.trim() ? options.name : `agent-${sessionId}`;
    const cwd = join(DEFAULT_AGENT_WORKSPACE_ROOT, sanitizeWorkspaceSegment(name));
    return {
      cwd,
      toolsOptions: { cwd },
    };
  }
  if (typeof workspace === "string") {
    const cwd = resolve(workspace);
    return {
      cwd,
      toolsOptions: { cwd },
    };
  }
  const cwd = resolve(workspace.cwd);
  return {
    cwd,
    toolsOptions: {
      cwd,
      ...(workspace.allowedDirectories ? { allowedDirectories: workspace.allowedDirectories } : {}),
      ...(workspace.bashTimeoutMs ? { bashTimeoutMs: workspace.bashTimeoutMs } : {}),
    },
  };
}

function formatAgentWorkspaceInstructions(cwd: string): string {
  return [
    "Workspace:",
    `Your private workspace is: ${cwd}`,
    "Use workspace tools for durable files, code, generated reports, logs, and test evidence.",
    "Keep file changes inside your workspace unless another explicitly granted tool allows a different location.",
    "Reply in natural language with the important workspace paths and a brief verification summary when handing off work.",
    "Do not use structured artifact reference objects as the collaboration protocol; use text, paths, evidence, and follow-up.",
  ].join("\n");
}

function formatRuntimePermissionInstructions(permissions: RuntimePermissions | undefined): string | undefined {
  const grants = activeWorkspaceGrants(permissions);
  if (grants.length === 0) return undefined;
  return [
    "Runtime workspace access for this task:",
    ...grants.flatMap((grant, index) => [
      `- grant ${index + 1}: ${grant.root}`,
      `  access: ${grant.access.join(", ")}`,
      ...(grant.reason ? [`  reason: ${grant.reason}`] : []),
      ...(grant.grantor?.name ? [`  granted_by: ${grant.grantor.name}`] : []),
      ...(grant.workItemId ? [`  work_item: ${grant.workItemId}`] : []),
    ]),
    "Read-only tools may inspect any path. Workspace grants are required for writing outside your private workspace; if a write tool returns permission_denied, write under an allowed root or ask your manager/host for a write grant.",
  ].join("\n");
}

function formatToolConcurrencyInstructions(
  tools: Array<ToolDefinition<any, any>>,
  options: Required<ToolConcurrencyOptions>,
): string | undefined {
  if (tools.length === 0 || options.mode === "sequential") return undefined;
  if (options.mode === "safe" && !tools.some(definition => definition.isConcurrencySafe)) {
    return undefined;
  }
  return [
    "You can request multiple independent tools in one assistant response; the runtime will execute calls concurrently when allowed.",
    "If a tool needs the result of an earlier tool to build its input, call them in separate assistant responses instead.",
  ].join(" ");
}

function joinPromptSections(sections: Array<string | undefined>): string | undefined {
  const joined = sections
    .map(section => section?.trim())
    .filter((section): section is string => Boolean(section))
    .join("\n\n");
  return joined || undefined;
}

function mergeToolsByName<TContext>(...groups: Array<Array<ToolDefinition<any, TContext>>>): Array<ToolDefinition<any, TContext>> {
  const merged: Array<ToolDefinition<any, TContext>> = [];
  const seen = new Set<string>();
  for (const group of groups) {
    for (const item of group) {
      if (seen.has(item.name)) continue;
      seen.add(item.name);
      merged.push(item);
    }
  }
  return merged;
}

/**
 * Entry point for implementing a custom trace sink. Validates that the
 * required `onEvent` method exists and binds `failOnError` at creation time —
 * a `ContextTracer` exposes methods only; the flag travels with the returned
 * object as internal metadata the runtime reads when a call throws.
 */
export function defineContextTracer(impl: ContextTracer & { failOnError?: boolean }): ContextTracer {
  if (typeof impl?.onEvent !== "function") {
    throw new Error("defineContextTracer requires an onEvent(event) method");
  }
  const { failOnError, ...methods } = impl;
  return bindFailOnError({ ...methods }, failOnError);
}

export function createJsonlContextTracer(options: JsonlContextTracerOptions): ContextTracer {
  if (!options.path && !options.dir) {
    throw new Error("createJsonlContextTracer requires either path or dir");
  }

  let queue = Promise.resolve();
  const tracer: ContextTracer = {
    onEvent(event) {
      const entry = options.redact ? options.redact(event) : event;
      if (!entry) return;
      const filePath = options.path ?? join(options.dir!, `${entry.session_id}.jsonl`);
      queue = queue.then(
        () => appendJsonlEntry(filePath, entry),
        () => appendJsonlEntry(filePath, entry),
      );
      return queue;
    },
    async flush() {
      await queue;
    },
    async close() {
      await queue;
    },
  };
  return bindFailOnError(tracer, options.failOnError);
}

/**
 * Chains hooks in array order: each one sees the previous one's output, so
 * redaction, truncation, and auditing can be written separately and combined.
 * Unlike the composite tracer, a failure is not swallowed — see `AgentHooks`.
 */
export function createCompositeAgentHooks<TContext = unknown>(
  hooks: Array<AgentHooks<TContext> | undefined | null>,
): AgentHooks<TContext> {
  const active = hooks.filter((hook): hook is AgentHooks<TContext> => Boolean(hook));
  return {
    async onToolResult(context) {
      let result = context.result;
      for (const hook of active) {
        const replacement = await hook.onToolResult?.({ ...context, result });
        if (replacement) result = replacement;
      }
      return result;
    },
    async onModelRequest(context) {
      let { messages, systemPrompt } = context;
      for (const hook of active) {
        const replacement = await hook.onModelRequest?.({ ...context, messages, systemPrompt });
        if (replacement?.messages) messages = replacement.messages;
        if (replacement?.systemPrompt !== undefined) systemPrompt = replacement.systemPrompt;
      }
      return { messages, ...(systemPrompt !== undefined ? { systemPrompt } : {}) };
    },
  };
}

export function createCompositeContextTracer(tracers: Array<ContextTracer | undefined | null>): ContextTracer {
  const active = tracers.filter((tracer): tracer is ContextTracer => Boolean(tracer));
  return bindFailOnError({
    async onEvent(event) {
      for (const tracer of active) {
        try {
          await tracer.onEvent(event);
        } catch (error) {
          if (shouldPropagatePortError(tracer)) throw error;
        }
      }
    },
    async flush() {
      for (const tracer of active) {
        if (!tracer.flush) continue;
        try {
          await tracer.flush();
        } catch (error) {
          if (shouldPropagatePortError(tracer)) throw error;
        }
      }
    },
    async close() {
      for (const tracer of active) {
        if (!tracer.close) continue;
        try {
          await tracer.close();
        } catch (error) {
          if (shouldPropagatePortError(tracer)) throw error;
        }
      }
    },
  }, active.some(shouldPropagatePortError));
}

type LangSmithTraceRunState = {
  root: LangSmithRunTreeLike;
  initialInputRecorded: boolean;
  modelRequests: number;
  toolUses: number;
  pendingModel?: LangSmithRunTreeLike;
  pendingTools: Map<string, LangSmithRunTreeLike>;
};

type LangSmithFlushableClient = {
  flush?: () => Promise<void>;
  awaitPendingTraceBatches?: () => Promise<void>;
};

export function createLangSmithContextTracer(options: LangSmithContextTracerOptions): ContextTracer {
  if (!options.RunTree && !options.runTree) {
    throw new Error("createLangSmithContextTracer requires either RunTree or runTree");
  }

  const makeRunTree = options.runTree ?? ((config: LangSmithRunTreeConfig) => new options.RunTree!(config));
  const runs = new Map<string, LangSmithTraceRunState>();
  let queue = Promise.resolve();

  async function handleEvent(rawEvent: ContextTraceEvent): Promise<void> {
    const event = options.redact ? options.redact(rawEvent) : rawEvent;
    if (!event) return;

    if (event.type === "run_start") {
      await startLangSmithRootRun(event, options, makeRunTree, runs);
      return;
    }

    const state = await ensureLangSmithRootRun(event, options, makeRunTree, runs);
    if (event.type === "user_message") {
      recordLangSmithUserMessage(state, event);
      return;
    }
    if (event.type === "model_request") {
      await startLangSmithModelRun(state, event, options);
      return;
    }
    if (event.type === "assistant_message") {
      await finishLangSmithModelRun(state, event);
      return;
    }
    if (event.type === "tool_use") {
      await startLangSmithToolRun(state, event, options);
      return;
    }
    if (event.type === "tool_result") {
      await finishLangSmithToolRun(state, event);
      return;
    }
    if (event.type === "result") {
      await finishLangSmithRootRun(state, event);
      return;
    }
    recordLangSmithRunEvent(state.root, event);
  }

  return bindFailOnError({
    onEvent(event) {
      queue = queue.then(
        () => handleEvent(event),
        () => handleEvent(event),
      );
      return queue;
    },
    async flush() {
      await queue;
      await flushLangSmithClients(options, runs);
    },
    async close() {
      await queue;
      await flushLangSmithClients(options, runs);
    },
  }, options.failOnError);
}

export function skill(input: SkillInput): SkillDefinition {
  if (!input.name.trim()) {
    throw new Error("Skill name is required");
  }
  if (!input.description.trim()) {
    throw new Error("Skill description is required");
  }
  if (!input.instructions.trim()) {
    throw new Error("Skill instructions are required");
  }
  return {
    name: input.name,
    description: input.description,
    instructions: input.instructions,
    ...(input.path ? { path: input.path } : {}),
  };
}

export async function loadSkill(path: string): Promise<SkillDefinition> {
  const skillPath = resolve(path, "SKILL.md");
  const raw = await readFile(skillPath, "utf8");
  const parsed = parseSkillMarkdown(raw);
  const name = parsed.frontmatter.name ?? basename(path);
  const description = parsed.frontmatter.description ?? "";
  return skill({
    name,
    description,
    instructions: parsed.body.trim(),
    path,
  });
}

export async function createMCPTools(
  client: MCPClient,
  options: MCPToolsOptions = {},
): Promise<Array<ToolDefinition<Record<string, unknown>>>> {
  const result = await client.listTools();
  return result.tools.map(mcpTool => {
    const sdkToolName = options.namePrefix
      ? `${sanitizeToolName(options.namePrefix)}_${sanitizeToolName(mcpTool.name)}`
      : sanitizeToolName(mcpTool.name);
    return {
      name: sdkToolName,
      description: mcpTool.description ?? `MCP tool ${mcpTool.name}`,
      inputSchema: mcpTool.inputSchema ?? { type: "object", additionalProperties: true },
      jsonSchema: mcpTool.inputSchema ?? { type: "object", additionalProperties: true },
      parse(input: unknown): Record<string, unknown> {
        return input && typeof input === "object" ? input as Record<string, unknown> : {};
      },
      async handler(input) {
        const output = await client.callTool({
          name: mcpTool.name,
          arguments: input,
        });
        const content = formatMCPToolResult(output);
        if (output.isError) {
          throw new Error(content);
        }
        return { content };
      },
    };
  });
}

export async function connectMCPStdioServer(
  server: StdioServerParameters,
  options: MCPStdioServerOptions = {},
): Promise<MCPStdioConnection> {
  const client = new MCPProtocolClient({
    name: options.clientName ?? "agent-lattice",
    version: options.clientVersion ?? "0.1.0",
  });
  const transport = new StdioClientTransport(server);
  await client.connect(transport);
  const wrappedClient: MCPClient = {
    listTools: () => client.listTools(),
    callTool: async input => toMCPCallToolResult(await client.callTool(input)),
  };
  return {
    client: wrappedClient,
    tools: await createMCPTools(wrappedClient, options),
    close: () => client.close(),
  };
}

export async function connectMCPStreamableHTTPServer(
  url: string | URL,
  options: MCPStreamableHTTPServerOptions = {},
): Promise<MCPStreamableHTTPConnection> {
  const client = new MCPProtocolClient({
    name: options.clientName ?? "agent-lattice",
    version: options.clientVersion ?? "0.1.0",
  });
  const transport = new StreamableHTTPClientTransport(
    typeof url === "string" ? new URL(url) : url,
    {
      ...(options.authProvider ? { authProvider: options.authProvider } : {}),
      ...(options.requestInit ? { requestInit: options.requestInit } : {}),
      ...(options.fetch ? { fetch: options.fetch } : {}),
      ...(options.reconnectionOptions ? { reconnectionOptions: options.reconnectionOptions } : {}),
      ...(options.sessionId ? { sessionId: options.sessionId } : {}),
    },
  );
  await client.connect(transport);
  const wrappedClient: MCPClient = {
    listTools: () => client.listTools(),
    callTool: async input => toMCPCallToolResult(await client.callTool(input)),
  };
  return {
    client: wrappedClient,
    tools: await createMCPTools(wrappedClient, options),
    close: () => client.close(),
    finishAuth: authorizationCode => transport.finishAuth(authorizationCode),
    terminateSession: () => transport.terminateSession(),
    get sessionId() {
      return transport.sessionId;
    },
  };
}

/**
 * Persists an Agent's history as one JSON message per line. `append()` adds a
 * line; `replace()` rewrites the whole file, which is how compaction is
 * persisted. `load()` reads every line and skips malformed ones — a truncated
 * final line from a torn write must not lose the rest of the transcript.
 * Writes are serialized through an internal queue, so callers may fire them
 * without waiting for ordering.
 */
/**
 * Entry point for implementing a custom history store. Validates that the
 * required `load`/`append`/`replace` methods exist and binds `failOnError` at
 * creation time — a `HistoryStore` exposes methods only; the flag travels
 * with the returned object as internal metadata the runtime reads when a call
 * throws.
 */
export function defineHistoryStore(impl: HistoryStore & { failOnError?: boolean }): HistoryStore {
  for (const method of ["load", "append", "replace"] as const) {
    if (typeof impl?.[method] !== "function") {
      throw new Error(`defineHistoryStore requires a ${method}() method`);
    }
  }
  const { failOnError, ...methods } = impl;
  return bindFailOnError({ ...methods }, failOnError);
}

export function createJsonlHistoryStore(options: JsonlHistoryStoreOptions): HistoryStore {
  let queue = Promise.resolve();
  const store: HistoryStore = {
    async load() {
      let raw: string;
      try {
        raw = await readFile(options.path, "utf8");
      } catch (error) {
        if ((error as { code?: string }).code === "ENOENT") return [];
        throw error;
      }
      const messages: ModelMessage[] = [];
      for (const line of raw.split("\n")) {
        if (!line.trim()) continue;
        try {
          const parsed = JSON.parse(line) as ModelMessage;
          if (parsed?.role === "user" || parsed?.role === "assistant") {
            messages.push(parsed);
          }
        } catch {
          // Malformed line: skip it, keep the rest of the transcript.
        }
      }
      return messages;
    },
    append(message) {
      queue = queue.then(
        () => appendJsonlHistoryMessage(options.path, message),
        () => appendJsonlHistoryMessage(options.path, message),
      );
      return queue;
    },
    replace(messages) {
      queue = queue.then(
        () => writeJsonlHistory(options.path, messages),
        () => writeJsonlHistory(options.path, messages),
      );
      return queue;
    },
  };
  return bindFailOnError(store, options.failOnError);
}

export function teamMember(input: TeamMemberInput): TeamMemberDefinition {
  if (!input.name.trim()) {
    throw new Error("Team member name is required");
  }
  return {
    name: sanitizeToolName(input.name),
    role: input.role,
    agent: input.agent,
    ...(input.focus ? { focus: input.focus } : {}),
    ...(input.mailboxId ? { mailboxId: input.mailboxId } : {}),
  };
}

export function createMemoryMailbox(): TeamMailbox {
  const messages: TeamMessage[] = [];
  let nextId = 0;
  const nextMessageId = () => {
    nextId++;
    if (nextId === 1) return "first";
    if (nextId === 2) return "second";
    if (nextId === 3) return "third";
    return `msg_${nextId}`;
  };

  return {
    async send(from, to, content, options = {}) {
      const id = nextMessageId();
      const inherited = options.parentMessageId
        ? messages.find(message => message.id === options.parentMessageId)
        : undefined;
      const workItemId = options.workItemId ?? inherited?.workItemId ?? id;
      const upstreamMessageId = options.upstreamMessageId ?? inherited?.upstreamMessageId ?? workItemId;
      const message: TeamMessage = {
        id,
        from,
        to,
        content,
        status: "pending",
        createdAt: Date.now(),
        threadId: options.threadId ?? inherited?.threadId ?? id,
        ...(options.parentMessageId ? { parentMessageId: options.parentMessageId } : {}),
        workItemId,
        workItemRole: options.workItemRole ?? inferTeamMessageRole(from, to, inherited),
        upstreamMessageId,
        ...(options.metadata ? { metadata: options.metadata } : {}),
      };
      messages.push(message);
      return { ...message };
    },
    async inbox(mailboxId, options = {}) {
      const status = options.status ?? "pending";
      return messages
        .filter(message => message.to === mailboxId)
        .filter(message => status === "all" || message.status === status)
        .map(message => ({ ...message }));
    },
    async get(messageId) {
      const message = messages.find(item => item.id === messageId);
      return message ? { ...message } : undefined;
    },
    async claimNext(mailboxId) {
      const message = messages
        .filter(item => item.to === mailboxId && item.status === "pending")
        .sort((left, right) => left.createdAt - right.createdAt)[0];
      if (!message) return undefined;
      message.status = "processing";
      return { ...message };
    },
    async updateStatus(messageId, status) {
      const message = messages.find(item => item.id === messageId);
      if (!message) return false;
      message.status = status;
      return true;
    },
  };
}

export function createSQLiteMailbox(options: SQLiteMailboxOptions): TeamMailbox {
  const table = normalizeSQLiteIdentifier(options.tableName ?? "team_mailbox_messages");
  initializeSQLiteMailbox(options.database, table);

  const insert = options.database.prepare(
    `INSERT INTO ${table} (
      id,
      from_mailbox,
      to_mailbox,
      content,
      status,
      created_at,
      thread_id,
      parent_message_id,
      work_item_id,
      work_item_role,
      upstream_message_id,
      metadata_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  );
  const selectById = options.database.prepare(
    `SELECT * FROM ${table} WHERE id = ? LIMIT 1`,
  );
  const selectInboxAll = options.database.prepare(
    `SELECT * FROM ${table} WHERE to_mailbox = ? ORDER BY created_at ASC`,
  );
  const selectInboxByStatus = options.database.prepare(
    `SELECT * FROM ${table} WHERE to_mailbox = ? AND status = ? ORDER BY created_at ASC`,
  );
  const update = options.database.prepare(
    `UPDATE ${table} SET status = ? WHERE id = ?`,
  );
  const claimSelect = options.database.prepare(
    `SELECT id FROM ${table} WHERE to_mailbox = ? AND status = 'pending' ORDER BY created_at ASC LIMIT 1`,
  );
  const claimUpdate = options.database.prepare(
    `UPDATE ${table} SET status = 'processing' WHERE id = ? AND status = 'pending'`,
  );

  return {
    async send(from, to, content, sendOptions = {}) {
      const id = `msg_${randomId().replace(/-/g, "").slice(0, 16)}`;
      const inherited = sendOptions.parentMessageId
        ? rowToTeamMessage(selectById.get(sendOptions.parentMessageId))
        : undefined;
      const workItemId = sendOptions.workItemId ?? inherited?.workItemId ?? id;
      const upstreamMessageId = sendOptions.upstreamMessageId ?? inherited?.upstreamMessageId ?? workItemId;
      const message: TeamMessage = {
        id,
        from,
        to,
        content,
        status: "pending",
        createdAt: Date.now(),
        threadId: sendOptions.threadId ?? inherited?.threadId ?? id,
        ...(sendOptions.parentMessageId ? { parentMessageId: sendOptions.parentMessageId } : {}),
        workItemId,
        workItemRole: sendOptions.workItemRole ?? inferTeamMessageRole(from, to, inherited),
        upstreamMessageId,
        ...(sendOptions.metadata ? { metadata: sendOptions.metadata } : {}),
      };
      insert.run(
        message.id,
        message.from,
        message.to,
        message.content,
        message.status,
        message.createdAt,
        message.threadId,
        message.parentMessageId ?? null,
        message.workItemId ?? null,
        message.workItemRole ?? null,
        message.upstreamMessageId ?? null,
        message.metadata ? JSON.stringify(message.metadata) : null,
      );
      return { ...message };
    },
    async inbox(mailboxId, inboxOptions = {}) {
      const status = inboxOptions.status ?? "pending";
      const rows = status === "all"
        ? selectInboxAll.all(mailboxId)
        : selectInboxByStatus.all(mailboxId, status);
      return rows.map(row => rowToTeamMessage(row)).filter(isTeamMessage);
    },
    async get(messageId) {
      return rowToTeamMessage(selectById.get(messageId));
    },
    async claimNext(mailboxId) {
      const row = claimSelect.get(mailboxId) as { id?: unknown } | undefined;
      if (!row || typeof row.id !== "string") return undefined;
      const result = claimUpdate.run(row.id);
      if (sqliteChanges(result) === 0) return undefined;
      return rowToTeamMessage(selectById.get(row.id));
    },
    async updateStatus(messageId, status) {
      const result = update.run(status, messageId);
      return sqliteChanges(result) > 0;
    },
  };
}

export function createTeam(options: TeamOptions): Team {
  if (!options.name.trim()) {
    throw new Error("Team name is required");
  }
  const name = sanitizeToolName(options.name);
  const mailbox = options.mailbox ?? createMemoryMailbox();
  const members = options.members.map(member => ({
    ...member,
    name: sanitizeToolName(member.name),
    mailboxId: member.mailboxId ?? teamMailboxId(name, member.name),
  }));
  const resolveMailbox = (target: string): string => resolveTeamMailbox(name, members, target);
  const leadMailboxTools = options.exposeLeadMailboxTools
    ? createTeamTools({
      mailbox,
      ownerMailboxId: "manager",
      resolveMailbox,
    })
    : [];
  const memberAgentTools = members.map(member => agentTool(
    member.name,
    member.agent,
    {
      description: formatTeamMemberDelegateDescription(member),
      targetMailboxId: member.mailboxId,
    },
  ));
  const leadTools = [...memberAgentTools, ...leadMailboxTools];
  const memberTools: Record<string, Array<ToolDefinition<any, any>>> = {};

  options.lead.addTools(leadTools);
  for (const member of members) {
    const tools = createTeamTools({
      mailbox,
      ownerMailboxId: member.mailboxId ?? teamMailboxId(name, member.name),
      resolveMailbox,
    });
    if (isToolCapableAgentLike(member.agent)) {
      member.agent.addTools(tools);
      memberTools[member.name] = tools;
    } else {
      memberTools[member.name] = [];
    }
  }

  return {
    name,
    lead: options.lead,
    members,
    mailbox,
    tools: leadTools,
    memberTools,
    runnerOptions: { ...options.runner },
    send: (from, to, content, sendOptions) => mailbox.send(from, resolveMailbox(to), content, sendOptions),
    drain: drainOptions => drainTeam({
      members,
      mailbox,
      options: drainOptions ?? {},
    }),
    query: (prompt, queryOptions) => createTeamRunner({
      root: options.lead,
      mailbox,
      source: { kind: "root", name, team: name, mailbox: "manager" },
      ...options.runner,
    }).query(prompt, queryOptions),
    prompt: (prompt, queryOptions) => createTeamRunner({
      root: options.lead,
      mailbox,
      source: { kind: "root", name, team: name, mailbox: "manager" },
      ...options.runner,
    }).prompt(prompt, queryOptions),
    interrupt: () => options.lead.interrupt(),
  };
}

export function createTeamRunner(options: TeamRunnerOptions): TeamRunner {
  const root = options.root ?? options.team?.lead;
  if (!root) {
    throw new Error("createTeamRunner requires either team or root");
  }
  const mailbox = options.mailbox ?? options.team?.mailbox ?? createMemoryMailbox();
  const source = options.source ?? {
    kind: "root" as const,
    name: options.team?.name ?? "root",
    mailbox: "manager",
  };
  const maxDelegateDepth = options.maxDelegateDepth ?? options.team?.runnerOptions?.maxDelegateDepth ?? 8;
  const maxConcurrentWorkItems = options.maxConcurrentWorkItems
    ?? options.team?.runnerOptions?.maxConcurrentWorkItems
    ?? 1;
  if (!Number.isInteger(maxConcurrentWorkItems) || maxConcurrentWorkItems < 1) {
    throw new Error("TeamRunnerOptions.maxConcurrentWorkItems must be a positive integer");
  }

  return {
    root,
    mailbox,
    interrupt: () => root.interrupt(),
    async *query(prompt: string | ContentBlock[], queryOptions: QueryOptions = {}): AsyncGenerator<TeamRunnerMessage> {
      const queue = new AsyncMessageQueue<TeamRunnerMessage>();
      let runError: unknown;
      const run = runTeamInvocation({
        agent: root,
        prompt,
        mailbox,
        source,
        emit: message => queue.push(message),
        maxDelegateDepth,
        maxConcurrentWorkItems,
        depth: 0,
        queryOptions,
        emitAgentMessage: emitRootAgentMessage,
      }).catch(error => {
        runError = error;
      }).finally(() => {
        queue.close();
      });

      while (true) {
        const next = await queue.next();
        if (next.done) break;
        yield next.value;
      }

      await run;
      if (runError) {
        throw runError;
      }
    },
    async prompt(prompt: string | ContentBlock[], queryOptions: QueryOptions = {}): Promise<SDKResultMessage> {
      let finalResult: SDKResultMessage | undefined;
      for await (const message of this.query(prompt, queryOptions)) {
        if (message.type === "result") {
          finalResult = message;
        }
      }
      if (!finalResult) {
        throw new Error("Team runner query completed without a result message");
      }
      return finalResult;
    },
  };
}

export function createAgentWorkspaceTools(options: AgentWorkspaceToolsOptions = {}): Array<ToolDefinition<any, any>> {
  const roots = normalizeAllowedDirectories(options);
  const cwd = roots.cwd;

  return [
    tool(
      "Read",
      "Read a file. Supports optional 1-based line offset and limit.",
      z.object({
        file_path: z.string(),
        offset: z.number().int().positive().optional(),
        limit: z.number().int().positive().optional(),
      }),
      async input => {
        const path = resolveToolPath(input.file_path, cwd);
        const content = await readFile(path, "utf8");
        if (input.offset === undefined && input.limit === undefined) {
          return { content };
        }
        const lines = content.split(/\r?\n/);
        const start = input.offset ? input.offset - 1 : 0;
        const end = input.limit ? start + input.limit : undefined;
        return { content: lines.slice(start, end).join("\n") };
      },
      { isConcurrencySafe: () => true },
    ),
    tool(
      "Write",
      "Write a file inside the workspace, creating parent directories as needed.",
      z.object({
        file_path: z.string(),
        content: z.string(),
      }),
      async (input, context) => {
        const path = resolveAuthorizedPath(input.file_path, roots, context, "Write", "write");
        await mkdir(dirname(path), { recursive: true });
        await writeFile(path, input.content, "utf8");
        return { content: `Wrote ${relative(cwd, path)}` };
      },
    ),
    tool(
      "Edit",
      "Replace text in an existing workspace file.",
      z.object({
        file_path: z.string(),
        old_string: z.string(),
        new_string: z.string(),
        replace_all: z.boolean().optional(),
      }),
      async (input, context) => {
        const path = resolveAuthorizedPath(input.file_path, roots, context, "Edit", "write");
        const content = await readFile(path, "utf8");
        if (!content.includes(input.old_string)) {
          throw new Error(`old_string was not found in ${input.file_path}`);
        }
        const next = input.replace_all
          ? content.split(input.old_string).join(input.new_string)
          : content.replace(input.old_string, input.new_string);
        await writeFile(path, next, "utf8");
        return { content: `Edited ${relative(cwd, path)}` };
      },
    ),
    tool(
      "LS",
      "List files and directories.",
      z.object({
        path: z.string().optional(),
      }),
      async input => {
        const path = resolveToolPath(input.path ?? ".", cwd);
        const entries = await readdir(path, { withFileTypes: true });
        const output = entries
          .map(entry => `${entry.name}${entry.isDirectory() ? "/" : ""}`)
          .sort()
          .join("\n");
        return { content: output };
      },
      { isConcurrencySafe: () => true },
    ),
    tool(
      "Glob",
      "Find files matching a glob pattern.",
      z.object({
        pattern: z.string(),
        path: z.string().optional(),
      }),
      async input => {
        const base = resolveToolPath(input.path ?? ".", cwd);
        const matcher = globToRegExp(input.pattern);
        const files = await listFiles(base);
        const output = files
          .map(file => normalizeSlash(relative(base, file)))
          .filter(file => matcher.test(file))
          .sort()
          .join("\n");
        return { content: output };
      },
      { isConcurrencySafe: () => true },
    ),
    tool(
      "Grep",
      "Search file contents with a regular expression.",
      z.object({
        pattern: z.string(),
        path: z.string().optional(),
        include: z.string().optional(),
      }),
      async input => {
        const base = resolveToolPath(input.path ?? ".", cwd);
        const matcher = input.include ? globToRegExp(input.include) : null;
        const regexp = new RegExp(input.pattern);
        const files = await listFiles(base);
        const lines: string[] = [];
        for (const file of files) {
          const rel = normalizeSlash(relative(base, file));
          if (matcher && !matcher.test(rel)) continue;
          const content = await readFile(file, "utf8").catch(() => "");
          content.split(/\r?\n/).forEach((line, index) => {
            if (regexp.test(line)) {
              lines.push(`${rel}:${index + 1}:${line}`);
            }
          });
        }
        return { content: lines.join("\n") };
      },
      { isConcurrencySafe: () => true },
    ),
    tool(
      "Bash",
      "Run a shell command in the workspace.",
      z.object({
        command: z.string(),
        timeout_ms: z.number().int().positive().optional(),
      }),
      async (input, context) => {
        authorizeShellCommand(input.command, roots, context);
        const output = await runShell(input.command, {
          cwd,
          timeoutMs: input.timeout_ms ?? options.bashTimeoutMs ?? 30_000,
        });
        return { content: output };
      },
    ),
  ];
}

export async function* query<TContext = unknown>(
  params: AgentOptions<TContext> & {
    prompt: string;
    stream?: boolean;
    outputFormat?: OutputFormat;
    thinkingConfig?: ThinkingConfig;
    reasoningEffort?: ReasoningEffort;
    signal?: AbortSignal;
    context?: TContext;
  },
): AsyncGenerator<SDKMessage> {
  const agent = createAgent(params);
  yield* agent.query(params.prompt, {
    stream: params.stream,
    outputFormat: params.outputFormat,
    thinkingConfig: params.thinkingConfig,
    reasoningEffort: params.reasoningEffort,
    signal: params.signal,
    context: params.context,
  });
}

/**
 * An Agent owns one conversation: history, workspace, and session identity.
 * Create instances with `createAgent()` or `createBareAgent()` (or spawn from
 * an `AgentSpec`) — the constructor is intentionally not exported, because the
 * factories also generate the session id and install workspace tools.
 */
class Agent<TContext = unknown> {
  private readonly options: Required<Pick<AgentOptions<TContext>, "maxTokens" | "maxTurns">> & AgentOptions<TContext>;
  private readonly modelClient: ModelClient;
  private readonly messages: ModelMessage[] = [];
  private readonly sessionId = randomId();
  private readonly toolConcurrency: Required<ToolConcurrencyOptions>;
  private running = false;
  private interruptController: AbortController | undefined;
  private historyLoaded: Promise<void> | undefined;

  constructor(options: AgentOptions<TContext>) {
    if (!options.model) {
      throw new Error("AgentOptions.model is required");
    }
    const internalOptions = options as InternalAgentOptions<TContext>;
    const publicOptions = { ...options } as InternalAgentOptions<TContext>;
    delete publicOptions[BARE_AGENT_OPTIONS];
    const configuredOptions = internalOptions[BARE_AGENT_OPTIONS]?.installWorkspace === false
      ? publicOptions
      : applyAgentWorkspaceOptions(publicOptions, this.sessionId);
    this.options = {
      maxTokens: 16384,
      maxTurns: 50,
      ...configuredOptions,
    };
    this.toolConcurrency = normalizeToolConcurrencyOptions(configuredOptions.toolConcurrency);
    this.modelClient =
      options.modelClient ??
      new AnthropicModelClient({
        apiKey: options.apiKey,
        baseURL: options.baseURL,
      });
  }

  /**
   * One Agent owns one conversation. Overlapping queries would interleave writes
   * into the shared history, so the second caller is rejected rather than served
   * a conversation containing someone else's turns.
   */
  async *query(prompt: string | ContentBlock[], options: QueryOptions<TContext> = {}): AsyncGenerator<SDKMessage> {
    if (this.running) {
      throw new ConcurrentQueryError(
        `Agent ${this.options.name ?? this.sessionId} is already running a query. ` +
          "An Agent holds one conversation: create a separate Agent per concurrent conversation.",
      );
    }
    this.running = true;
    const interruptController = new AbortController();
    this.interruptController = interruptController;
    try {
      yield* this.runQuery(prompt, options, interruptController.signal);
    } finally {
      this.running = false;
      if (this.interruptController === interruptController) {
        this.interruptController = undefined;
      }
    }
  }

  /**
   * Abort the in-flight model request and end the current query with subtype
   * "interrupted". Completed turns stay in history; follow up with a new
   * query() to continue the conversation. Unlike `QueryOptions.signal`, which
   * terminates the query as an error, an interrupt is normal control flow.
   * Returns `true` when a query was running and is now interrupted, `false`
   * when idle — a `false` tells the host it can send its next query directly
   * instead of waiting for an "interrupted" result.
   */
  interrupt(): boolean {
    if (!this.interruptController) return false;
    this.interruptController.abort();
    return true;
  }

  /**
   * The store is read once per Agent lifetime, lazily on the first query or
   * getHistory() call — the constructor cannot be async. A failed load follows
   * the same rule as a failed write: swallowed unless `failOnError` was bound
   * by the store's factory, in which case the error propagates and the Agent
   * starts empty.
   */
  private ensureHistoryLoaded(): Promise<void> {
    this.historyLoaded ??= this.loadHistory();
    return this.historyLoaded;
  }

  private async loadHistory(): Promise<void> {
    const store = this.options.historyStore;
    if (!store) return;
    try {
      const seeded = await store.load();
      // Clone so the store cannot mutate the live history through its copy.
      this.messages.push(...structuredClone(seeded));
    } catch (error) {
      if (shouldPropagatePortError(store)) throw error;
    }
  }

  private async appendStoredHistory(message: ModelMessage): Promise<void> {
    const store = this.options.historyStore;
    if (!store) return;
    try {
      await store.append(message);
    } catch (error) {
      if (shouldPropagatePortError(store)) throw error;
    }
  }

  private async replaceStoredHistory(): Promise<void> {
    const store = this.options.historyStore;
    if (!store) return;
    try {
      await store.replace([...this.messages]);
    } catch (error) {
      if (shouldPropagatePortError(store)) throw error;
    }
  }

  private async *runQuery(
    prompt: string | ContentBlock[],
    options: QueryOptions<TContext>,
    interruptSignal: AbortSignal,
  ): AsyncGenerator<SDKMessage> {
    const tracer = options.tracer ?? this.options.tracer;
    const runId = randomId();
    const totals: QueryTotals = { usage: emptyUsage() };
    const autoCompact = normalizeAutoCompact(this.options.autoCompact);
    let lastInputTokens = 0;
    let triggerInputTokens = 0;
    let overflowRecovered = false;
    const queryTraceContext = getQueryTraceContext(options);
    const source: AgentRuntimeSource = options.agentRuntime?.source ?? {
      kind: "agent",
      name: this.options.name ?? "agent",
    };
    const effectivePermissions = options.agentRuntime?.permissions ?? options.permissions;
    const systemPrompt = joinPromptSections([
      this.options.systemPrompt,
      formatToolConcurrencyInstructions(this.options.tools ?? [], this.toolConcurrency),
      formatRuntimePermissionInstructions(effectivePermissions),
    ]);
    const traceBase = {
      session_id: queryTraceContext?.sessionId ?? this.sessionId,
      run_id: runId,
      ...(queryTraceContext ? { parent_run_id: queryTraceContext.parentRunId } : {}),
      source,
    };

    await emitTraceEvent(tracer, {
      ...traceBase,
      type: "run_start",
      data: {
        model: this.options.model,
        tools: (this.options.tools ?? []).map(tool => tool.name),
        ...(queryTraceContext ? { agent_session_id: this.sessionId } : {}),
      },
    });

    const startAbort = abortErrorIfNeeded(options.signal);
    yield this.initMessage();
    if (startAbort) {
      const result = this.resultMessage("error_abort", "", 0, startAbort, totals);
      await emitTraceEvent(tracer, {
        ...traceBase,
        type: "result",
        data: traceResultData(result),
      });
      await flushTracer(tracer);
      yield result;
      return;
    }

    const inputMessage: ModelMessage = {
      role: "user",
      content: prompt,
    };
    await this.ensureHistoryLoaded();
    this.messages.push(inputMessage);
    await this.appendStoredHistory(inputMessage);
    await emitTraceEvent(tracer, {
      ...traceBase,
      type: "user_message",
      data: { message: inputMessage },
    });

    let turns = 0;
    while (true) {
      const abortError = abortErrorIfNeeded(options.signal);
      if (abortError) {
        const result = this.resultMessage("error_abort", "", turns, abortError, totals);
        await emitTraceEvent(tracer, {
          ...traceBase,
          type: "result",
          data: traceResultData(result),
        });
        await flushTracer(tracer);
        yield result;
        return;
      }

      // interrupt() during a tool batch lands here: the batch's results are
      // already in history, so the query ends cleanly before the next turn.
      if (interruptSignal.aborted) {
        const result = this.resultMessage("interrupted", "", turns, undefined, totals);
        await emitTraceEvent(tracer, {
          ...traceBase,
          type: "result",
          data: traceResultData(result),
        });
        await flushTracer(tracer);
        yield result;
        return;
      }

      if (turns >= this.options.maxTurns) {
        const error = new MaxTurnsError(`Reached maximum number of turns (${this.options.maxTurns})`);
        const result = this.resultMessage("error_max_turns", "", turns, error, totals);
        await emitTraceEvent(tracer, {
          ...traceBase,
          type: "result",
          data: traceResultData(result),
        });
        await flushTracer(tracer);
        yield result;
        return;
      }

      // Compaction happens between turns, once a response has shown how large
      // the context has become, and before the next request pays for it again.
      if (autoCompact && lastInputTokens > autoCompact.thresholdTokens) {
        const compaction = await this.compactHistory(autoCompact, options);
        if (compaction) {
          lastInputTokens = 0;
          totals.usage = addUsage(totals.usage, compaction.usage);
          await emitTraceEvent(tracer, {
            ...traceBase,
            type: "compaction",
            data: {
              compacted_messages: compaction.compacted,
              retained_messages: compaction.retained,
              trigger_input_tokens: triggerInputTokens,
              summary: compaction.summary,
            },
          });
          yield {
            type: "system",
            subtype: "compaction",
            session_id: this.sessionId,
            compacted_messages: compaction.compacted,
            retained_messages: compaction.retained,
            trigger_input_tokens: triggerInputTokens,
            usage: compaction.usage ?? emptyUsage(),
          };
        } else {
          // Nothing safe to cut; do not retry every turn.
          lastInputTokens = 0;
        }
      }

      turns++;
      let assistant: AssistantModelMessage;
      const streamEvents = new StreamEventQueue();
      let modelMessages = this.messagesForModel(prompt);
      let systemPromptForTurn = systemPrompt;
      const onModelRequest = this.options.hooks?.onModelRequest;
      if (onModelRequest) {
        // Shapes this request only: the stored conversation is untouched, so
        // trimming context here cannot destroy history.
        const replacement = await onModelRequest({
          messages: modelMessages,
          ...(systemPromptForTurn ? { systemPrompt: systemPromptForTurn } : {}),
          turn: turns,
          context: options.context,
          source,
          signal: options.signal,
        });
        if (replacement?.messages) modelMessages = replacement.messages;
        if (replacement?.systemPrompt !== undefined) systemPromptForTurn = replacement.systemPrompt;
      }
      const modelTools = this.modelTools();
      const stream = options.stream ?? true;
      const thinkingConfig = options.thinkingConfig ?? this.options.thinkingConfig;
      const reasoningEffort = options.reasoningEffort ?? this.options.reasoningEffort;
      const requestTimeoutMs = normalizeRequestTimeout(
        options.requestTimeoutMs ?? this.options.requestTimeoutMs,
        options.requestTimeoutMs !== undefined
          ? "QueryOptions.requestTimeoutMs"
          : "AgentOptions.requestTimeoutMs",
      );
      await emitTraceEvent(tracer, {
        ...traceBase,
        type: "model_request",
        data: {
          model: this.options.model,
          max_tokens: this.options.maxTokens,
          ...(systemPromptForTurn ? { systemPrompt: systemPromptForTurn } : {}),
          messages: modelMessages,
          tools: modelTools.map(tool => tool.name),
          permissions: traceRuntimePermissions(effectivePermissions),
          stream,
          ...(options.outputFormat ? { outputFormat: options.outputFormat } : {}),
          ...(thinkingConfig ? { thinkingConfig } : {}),
          ...(reasoningEffort ? { reasoningEffort } : {}),
        },
      });
      // The model call runs alongside the drain loop below so stream events reach
      // the consumer while the model is still responding. It is raced against the
      // caller's signal, the interrupt signal, and the request deadline so that
      // a model client which ignores them cannot stall the loop forever.
      const modelSignal = combineAbortSignals(options.signal, interruptSignal);
      const modelCall = settle(
        withDeadline(
          this.modelClient.createMessage({
            model: this.options.model,
            systemPrompt: systemPromptForTurn,
            maxTokens: this.options.maxTokens,
            messages: modelMessages,
            tools: modelTools,
            stream,
            outputFormat: options.outputFormat,
            thinkingConfig,
            reasoningEffort,
            timeoutMs: requestTimeoutMs,
            onStreamEvent: event => {
              if (stream) {
                streamEvents.push(event);
              }
            },
            signal: modelSignal,
          }),
          { signal: modelSignal, timeoutMs: requestTimeoutMs },
        ),
        () => streamEvents.finish(),
      );

      for await (const event of streamEvents.drain()) {
        yield {
          type: "stream_event",
          event,
          session_id: this.sessionId,
        };
      }

      const outcome = await modelCall;
      if (!outcome.ok) {
        // interrupt() fired while the model call was in flight. As with an
        // abort, the partial assistant message is dropped; unlike an abort,
        // this is normal control flow and the query ends "interrupted" so the
        // host can continue the conversation with a new query. A caller abort
        // still wins when both fired.
        if (interruptSignal.aborted && !abortErrorIfNeeded(options.signal)) {
          const result = this.resultMessage("interrupted", "", turns, undefined, totals);
          await emitTraceEvent(tracer, {
            ...traceBase,
            type: "result",
            data: traceResultData(result),
          });
          await flushTracer(tracer);
          yield result;
          return;
        }
        // An input that already exceeds the window is rejected before
        // generation, so it arrives here rather than as a stop reason.
        if (autoCompact && !overflowRecovered && isContextOverflowError(outcome.error)) {
          const recovery = await this.recoverFromOverflow(
            autoCompact, options, tracer, traceBase, totals, triggerInputTokens,
          );
          if (recovery) {
            overflowRecovered = true;
            lastInputTokens = 0;
            yield recovery;
            continue;
          }
        }
        const wrapped =
          outcome.error instanceof AbortError || outcome.error instanceof TimeoutError
            ? outcome.error
            : new APIError(errorMessage(outcome.error), { cause: outcome.error });
        const subtype = wrapped instanceof TimeoutError
          ? "error_timeout"
          : wrapped instanceof AbortError
            ? "error_abort"
            : "error";
        const result = this.resultMessage(subtype, "", turns, wrapped, totals);
        await emitTraceEvent(tracer, {
          ...traceBase,
          type: "result",
          data: traceResultData(result),
        });
        await flushTracer(tracer);
        yield result;
        return;
      }
      assistant = outcome.value;
      totals.usage = addUsage(totals.usage, assistant.usage);
      totals.stopReason = assistant.stopReason;
      lastInputTokens = assistant.usage?.input_tokens ?? 0;
      triggerInputTokens = lastInputTokens;

      // The window ran out mid-request, so this response is unusable and must
      // not enter the history. Compact and retry the same turn instead.
      // "max_tokens" is deliberately not handled here: that is an output-budget
      // failure, and compacting the input would not make the answer complete.
      if (
        autoCompact &&
        !overflowRecovered &&
        assistant.stopReason === "model_context_window_exceeded"
      ) {
        const recovery = await this.recoverFromOverflow(
          autoCompact, options, tracer, traceBase, totals, triggerInputTokens,
        );
        if (recovery) {
          overflowRecovered = true;
          lastInputTokens = 0;
          yield recovery;
          continue;
        }
      }

      this.messages.push(assistant);
      await this.appendStoredHistory(assistant);
      await emitTraceEvent(tracer, {
        ...traceBase,
        type: "assistant_message",
        data: { message: assistant },
      }, outcome.settledAt);
      yield {
        type: "assistant",
        message: assistant,
        session_id: this.sessionId,
      };

      const toolUseBlocks = assistant.content.filter(isToolUseBlock);
      if (toolUseBlocks.length === 0) {
        const result = this.resultMessage("success", extractText(assistant), turns, undefined, totals);
        await emitTraceEvent(tracer, {
          ...traceBase,
          type: "result",
          data: traceResultData(result),
        });
        await flushTracer(tracer);
        yield result;
        return;
      }

      const toolResults: ToolResultBlock[] = [];
      let firstToolError: Error | undefined;
      let batchRejection: ToolBatchPolicyRejection | undefined;

      // Emitted per call at its own start rather than for the whole batch up
      // front, so a trace shows when each tool really began and which calls
      // overlapped.
      const startToolExecution = async (block: ToolUseBlock): Promise<void> => {
        const description = this.options.tools?.find(tool => tool.name === block.name)?.description;
        await emitTraceEvent(tracer, {
          ...traceBase,
          type: "tool_use",
          data: {
            id: block.id,
            name: block.name,
            input: block.input,
            ...(description ? { description } : {}),
          },
        });
      };

      // Every result block reaches the model through here — handler success,
      // handler failure, abort, and batch rejection alike — so the hook covers
      // all of them, and runs before the trace event records what was sent.
      const finishToolExecution = async (result: ToolExecutionOutcome): Promise<ToolExecutionOutcome> => {
        const onToolResult = this.options.hooks?.onToolResult;
        if (onToolResult) {
          const block = toolUseBlocks.find(candidate => candidate.id === result.block.tool_use_id);
          const replacement = await onToolResult({
            toolName: block?.name ?? "",
            toolUseId: result.block.tool_use_id,
            input: block?.input,
            result: result.block,
            ...(result.error ? { error: result.error } : {}),
            context: options.context,
            source,
            signal: options.signal,
          });
          if (replacement) {
            result = { ...result, block: replacement };
          }
        }
        await emitTraceEvent(tracer, {
          ...traceBase,
          type: "tool_result",
          data: {
            tool_use_id: result.block.tool_use_id,
            content: result.block.content,
            is_error: result.block.is_error ?? false,
            ...(result.error ? { error: result.error } : {}),
          },
        });
        return result;
      };
      const executeTool = async (block: ToolUseBlock): Promise<ToolExecutionOutcome> => {
        await startToolExecution(block);
        return finishToolExecution(await this.runTool(
          block,
          options.signal,
          options.agentRuntime,
          options.permissions,
          options.context,
        ));
      };
      const cancelTool = async (block: ToolUseBlock): Promise<ToolExecutionOutcome> => {
        await startToolExecution(block);
        const error = new AbortError(`Tool ${block.name} was not started because the operation was aborted`);
        return finishToolExecution({
          block: {
            type: "tool_result",
            tool_use_id: block.id,
            content: error.message,
            is_error: true,
          },
          error,
        });
      };

      if (this.options.toolBatchPolicy) {
        const toolCalls: ToolBatchCall[] = toolUseBlocks.map(block => ({
          id: block.id,
          name: block.name,
          input: block.input,
          kind: this.options.tools?.find(tool => tool.name === block.name)?.kind ?? "tool",
        }));
        try {
          const decision = await this.options.toolBatchPolicy.validate({
            source: options.agentRuntime?.source,
            toolCalls,
            context: options.context,
            signal: options.signal,
          });
          if (!decision.allowed) batchRejection = decision;
        } catch (error) {
          const abortError = error instanceof AbortError
            ? error
            : abortErrorIfNeeded(options.signal);
          if (abortError) {
            // The assistant message carrying these tool_use blocks is already
            // in the history, so pair every call with an aborted tool_result
            // before leaving: a dangling tool_use would make the history sent
            // by the next query invalid.
            const abortedResults: ToolResultBlock[] = [];
            for (const block of toolUseBlocks) {
              abortedResults.push((await cancelTool(block)).block);
            }
            const abortedMessage: ModelMessage = {
              role: "user",
              content: abortedResults,
            };
            this.messages.push(abortedMessage);
            await this.appendStoredHistory(abortedMessage);
            await emitTraceEvent(tracer, {
              ...traceBase,
              type: "user_message",
              data: { message: abortedMessage },
            });
            yield {
              type: "user",
              message: abortedMessage,
              session_id: this.sessionId,
              tool_use_result: abortedResults.length === 1 ? abortedResults[0]?.content : abortedResults,
              error: abortError,
            };
            const result = this.resultMessage("error_abort", "", turns, abortError, totals);
            await emitTraceEvent(tracer, {
              ...traceBase,
              type: "result",
              data: traceResultData(result),
            });
            await flushTracer(tracer);
            yield result;
            return;
          }
          batchRejection = {
            allowed: false,
            code: "tool_batch_policy_error",
            message: "Tool batch validation failed before execution.",
            suggestedNextStep: "Retry with the tool calls split into separate steps.",
          };
        }
      }

      const batchError = batchRejection
        ? new ToolBatchRejectedError(batchRejection)
        : undefined;

      let executionResults: ToolExecutionOutcome[];
      if (batchRejection && batchError) {
        executionResults = [];
        for (const block of toolUseBlocks) {
          await startToolExecution(block);
          executionResults.push(await finishToolExecution({
            block: {
              type: "tool_result",
              tool_use_id: block.id,
              content: formatToolBatchRejection(batchRejection),
              is_error: true,
            },
            error: batchError,
          }));
        }
      } else {
        executionResults = await this.executeToolBatch(
          toolUseBlocks,
          options.signal,
          executeTool,
          cancelTool,
        );
      }

      firstToolError = executionResults.find(result => result.error)?.error;
      for (const result of executionResults) {
        toolResults.push(result.block);
      }

      const userMessage: ModelMessage = {
        role: "user",
        content: toolResults,
      };
      this.messages.push(userMessage);
      await this.appendStoredHistory(userMessage);
      await emitTraceEvent(tracer, {
        ...traceBase,
        type: "user_message",
        data: { message: userMessage },
      });
      yield {
        type: "user",
        message: userMessage,
        session_id: this.sessionId,
        tool_use_result: toolResults.length === 1 ? toolResults[0]?.content : toolResults,
        ...(firstToolError ? { error: firstToolError } : {}),
      };

      if (options.agentRuntime?.shouldPauseAfterToolBatch?.()) {
        const result = this.resultMessage(
          "success",
          "Delegated work was queued. Waiting for team runtime reports before continuing.",
          turns,
          undefined,
          totals,
        );
        await emitTraceEvent(tracer, {
          ...traceBase,
          type: "result",
          data: traceResultData(result),
        });
        await flushTracer(tracer);
        yield result;
        return;
      }

      // A tool asked to end the run: every result of the batch is already in
      // history and reported above; only the next model call is skipped.
      const endTurnOutcome = executionResults.find(result => result.endTurn);
      if (endTurnOutcome) {
        const result = this.resultMessage(
          "success",
          toolResultText(endTurnOutcome.block.content),
          turns,
          undefined,
          totals,
        );
        await emitTraceEvent(tracer, {
          ...traceBase,
          type: "result",
          data: traceResultData(result),
        });
        await flushTracer(tracer);
        yield result;
        return;
      }
    }
  }

  async prompt(prompt: string | ContentBlock[], options: QueryOptions<TContext> = {}): Promise<SDKResultMessage> {
    let finalResult: SDKResultMessage | undefined;
    for await (const message of this.query(prompt, options)) {
      if (message.type === "result") {
        finalResult = message;
      }
    }
    if (!finalResult) {
      throw new Error("Agent query completed without a result message");
    }
    return finalResult;
  }

  /**
   * The conversation history as currently held by this Agent, including any
   * messages seeded from `AgentOptions.historyStore`. Returns a deep copy, so
   * mutating the result cannot corrupt the live conversation.
   */
  async getHistory(): Promise<ModelMessage[]> {
    await this.ensureHistoryLoaded();
    return structuredClone(this.messages);
  }

  /**
   * Replace the conversation history. Idle only: throws ConcurrentQueryError
   * while a query is running. When a historyStore is configured the store is
   * replaced too, so persistence stays in sync.
   *
   * The host owns the content: the SDK does not validate the messages, so the
   * replacement must be a well-formed history — e.g. no dangling `tool_use`
   * without its matching `tool_result`.
   */
  async replaceHistory(messages: ModelMessage[]): Promise<void> {
    if (this.running) {
      throw new ConcurrentQueryError(
        `Agent ${this.options.name ?? this.sessionId} is already running a query. ` +
          "replaceHistory() is idle-only: wait for the running query to finish, or interrupt() it first.",
      );
    }
    // Mark the history as loaded so the lazy store load on the first query
    // cannot seed over the replacement.
    this.historyLoaded = Promise.resolve();
    this.messages.length = 0;
    // Clone so the caller cannot mutate the live history through its copy.
    this.messages.push(...structuredClone(messages));
    await this.replaceStoredHistory();
  }

  addTools(tools: Array<ToolDefinition<any, TContext>>): void {
    this.options.tools = [...(this.options.tools ?? []), ...tools];
  }

  private initMessage(): SDKSystemInitMessage {
    return {
      type: "system",
      subtype: "init",
      model: this.options.model,
      tools: (this.options.tools ?? []).map(tool => tool.name),
      session_id: this.sessionId,
    };
  }

  private resultMessage(
    subtype: SDKResultMessage["subtype"],
    result: string,
    numTurns: number,
    error?: Error,
    totals?: QueryTotals,
  ): SDKResultMessage {
    return {
      type: "result",
      subtype,
      // "interrupted" is normal control flow, not a failure.
      is_error: subtype !== "success" && subtype !== "interrupted",
      result,
      session_id: this.sessionId,
      num_turns: numTurns,
      ...(error ? { error } : {}),
      usage: totals?.usage ?? emptyUsage(),
      ...(totals?.stopReason ? { stop_reason: totals.stopReason } : {}),
    };
  }

  private modelTools(): ModelToolDefinition[] {
    return (this.options.tools ?? []).map(tool => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.jsonSchema,
    }));
  }

  /**
   * Replaces the summarizable head of the conversation with a model-written
   * summary. Returns undefined when there is nothing safe to compact, and lets a
   * failed summarization surface so the caller can decide: compaction is best
   * effort, and continuing with a full history is better than losing it.
   */
  private async compactHistory(
    settings: NonNullable<ReturnType<typeof normalizeAutoCompact>>,
    options: QueryOptions<TContext>,
  ): Promise<{ summary: string; compacted: number; retained: number; usage?: TokenUsage } | undefined> {
    const splitIndex = compactionSplitIndex(this.messages, settings.keepRecentMessages);
    if (splitIndex === 0) return undefined;

    const head = this.messages.slice(0, splitIndex);
    const tail = this.messages.slice(splitIndex);
    const response = await this.modelClient.createMessage({
      model: settings.model ?? this.options.model,
      systemPrompt: settings.prompt,
      maxTokens: settings.maxTokens,
      // The history to summarize, plus the instruction to do it now.
      messages: [...head, { role: "user", content: "Summarize the conversation so far." }],
      tools: [],
      stream: false,
      signal: options.signal,
    });

    const summary = extractText(response).trim();
    if (!summary) return undefined;

    this.messages.length = 0;
    this.messages.push(compactionHandoffMessage(summary), ...tail);
    // Compaction rewrites the whole history, so the store needs the full
    // replacement, not a diff.
    await this.replaceStoredHistory();
    return {
      summary,
      compacted: head.length,
      retained: tail.length,
      ...(response.usage ? { usage: response.usage } : {}),
    };
  }

  /**
   * Last-resort compaction after a turn already ran out of context, as opposed
   * to the threshold check that runs between turns. Returns the message to emit
   * so the caller can retry, or undefined when compaction cannot help and the
   * original failure should surface instead.
   */
  private async recoverFromOverflow(
    settings: NonNullable<ReturnType<typeof normalizeAutoCompact>>,
    options: QueryOptions<TContext>,
    tracer: ContextTracer | undefined,
    traceBase: Omit<ContextTraceEvent, "version" | "timestamp" | "seq" | "type" | "data">,
    totals: QueryTotals,
    triggerInputTokens: number,
  ): Promise<SDKSystemCompactionMessage | undefined> {
    let compaction: Awaited<ReturnType<typeof this.compactHistory>>;
    try {
      compaction = await this.compactHistory(settings, options);
    } catch {
      // Summarizing may fail for the same reason the turn did. Surfacing the
      // original overflow is more useful than a second, derived failure.
      return undefined;
    }
    if (!compaction) return undefined;

    totals.usage = addUsage(totals.usage, compaction.usage);
    await emitTraceEvent(tracer, {
      ...traceBase,
      type: "compaction",
      data: {
        compacted_messages: compaction.compacted,
        retained_messages: compaction.retained,
        trigger_input_tokens: triggerInputTokens,
        recovered_from_overflow: true,
        summary: compaction.summary,
      },
    });
    return {
      type: "system",
      subtype: "compaction",
      session_id: this.sessionId,
      compacted_messages: compaction.compacted,
      retained_messages: compaction.retained,
      trigger_input_tokens: triggerInputTokens,
      usage: compaction.usage ?? emptyUsage(),
    };
  }

  private messagesForModel(prompt: string | ContentBlock[]): ModelMessage[] {
    const selectedSkills = this.selectSkills(prompt);
    if (selectedSkills.length === 0) {
      return [...this.messages];
    }
    return [
      {
        role: "user",
        content: formatSkillInstructions(selectedSkills),
      },
      ...this.messages,
    ];
  }

  private selectSkills(prompt: string | ContentBlock[]): SkillDefinition[] {
    const skills = this.options.skills ?? [];
    if (skills.length === 0) return [];
    const promptText = promptTextForMatching(prompt);
    if (!promptText) return [];
    const normalizedPrompt = normalizeForMatch(promptText);
    return skills.filter(definition => {
      const haystack = normalizeForMatch(`${definition.name} ${definition.description}`);
      return haystack
        .split(/\s+/)
        .filter(token => token.length >= 3)
        .some(token => normalizedPrompt.includes(token));
    });
  }

  private async runTool(
    block: ToolUseBlock,
    signal: AbortSignal | undefined,
    agentRuntime: AgentRuntimeContext | undefined,
    permissions: RuntimePermissions | undefined,
    context: TContext | undefined,
  ): Promise<ToolExecutionOutcome> {
    const definition = (this.options.tools ?? []).find(tool => tool.name === block.name);
    if (!definition) {
      return {
        block: {
          type: "tool_result",
          tool_use_id: block.id,
          content: `Tool ${block.name} is not registered`,
          is_error: true,
        },
      };
    }

    try {
      const permission = await (this.options.permission?.({
        toolName: block.name,
        input: block.input,
        toolUseId: block.id,
      }) ?? { behavior: "allow" as const });
      if (permission.behavior === "deny") {
        return {
          block: {
            type: "tool_result",
            tool_use_id: block.id,
            content: permission.message,
            is_error: true,
          },
        };
      }

      const parsed = definition.parse(block.input);
      const output = await definition.handler(parsed, {
        signal,
        toolUseId: block.id,
        context,
        agentRuntime,
        permissions: agentRuntime?.permissions ?? permissions,
      });
      return {
        block: {
          type: "tool_result",
          tool_use_id: block.id,
          content: output.content,
        },
        ...(output.endTurn ? { endTurn: true } : {}),
      };
    } catch (error) {
      if (error instanceof ToolPermissionDeniedError) {
        return {
          block: {
            type: "tool_result",
            tool_use_id: block.id,
            content: formatPermissionDeniedToolResult(error.denial),
            is_error: true,
          },
          error,
        };
      }
      const wrapped = new ToolExecutionError(`Tool ${block.name} failed: ${errorMessage(error)}`, {
        cause: error,
      });
      return {
        block: {
          type: "tool_result",
          tool_use_id: block.id,
          content: wrapped.message,
          is_error: true,
        },
        error: wrapped,
      };
    }
  }

  private async executeToolBatch(
    blocks: ToolUseBlock[],
    signal: AbortSignal | undefined,
    execute: (block: ToolUseBlock) => Promise<ToolExecutionOutcome>,
    cancel: (block: ToolUseBlock) => Promise<ToolExecutionOutcome>,
  ): Promise<ToolExecutionOutcome[]> {
    const results: ToolExecutionOutcome[] = new Array(blocks.length);
    let index = 0;
    while (index < blocks.length) {
      const block = blocks[index]!;
      if (!this.isToolCallConcurrencySafe(block)) {
        results[index] = signal?.aborted ? await cancel(block) : await execute(block);
        index++;
        continue;
      }

      let end = index + 1;
      while (end < blocks.length && this.isToolCallConcurrencySafe(blocks[end]!)) {
        end++;
      }
      const batch = blocks.slice(index, end);
      const batchResults = await mapWithConcurrency(
        batch,
        this.toolConcurrency.maxConcurrency,
        item => signal?.aborted ? cancel(item) : execute(item),
      );
      for (let offset = 0; offset < batchResults.length; offset++) {
        results[index + offset] = batchResults[offset]!;
      }
      index = end;
    }
    return results;
  }

  private isToolCallConcurrencySafe(block: ToolUseBlock): boolean {
    if (this.toolConcurrency.mode === "all") return true;
    if (this.toolConcurrency.mode === "sequential") return false;
    const definition = (this.options.tools ?? []).find(tool => tool.name === block.name);
    if (!definition?.isConcurrencySafe) return false;
    try {
      return Boolean(definition.isConcurrencySafe(definition.parse(block.input)));
    } catch {
      return false;
    }
  }
}

// The Agent class is a type-only export: instances come from createAgent(),
// createBareAgent(), or AgentSpec.spawn(), never from a public constructor.
export type { Agent };

function normalizeToolConcurrencyOptions(
  options: ToolConcurrencyOptions | undefined,
): Required<ToolConcurrencyOptions> {
  const mode = options?.mode ?? "safe";
  if (!(["safe", "all", "sequential"] as ToolConcurrencyMode[]).includes(mode)) {
    throw new Error('AgentOptions.toolConcurrency.mode must be "safe", "all", or "sequential"');
  }
  const maxConcurrency = options?.maxConcurrency ?? 10;
  if (!Number.isInteger(maxConcurrency) || maxConcurrency < 1) {
    throw new Error("AgentOptions.toolConcurrency.maxConcurrency must be a positive integer");
  }
  return { mode, maxConcurrency };
}

async function mapWithConcurrency<TInput, TOutput>(
  inputs: TInput[],
  maxConcurrency: number,
  execute: (input: TInput) => Promise<TOutput>,
): Promise<TOutput[]> {
  const outputs: TOutput[] = new Array(inputs.length);
  let nextIndex = 0;
  const workers = Array.from(
    { length: Math.min(maxConcurrency, inputs.length) },
    async () => {
      while (true) {
        const index = nextIndex++;
        if (index >= inputs.length) return;
        outputs[index] = await execute(inputs[index]!);
      }
    },
  );
  const settled = await Promise.allSettled(workers);
  const rejected = settled.find(result => result.status === "rejected");
  if (rejected?.status === "rejected") throw rejected.reason;
  return outputs;
}

/**
 * Buffers provider stream events only until the consumer asks for the next one,
 * so `query()` can yield them while the model request is still in flight.
 */
class StreamEventQueue {
  private readonly pending: Record<string, unknown>[] = [];
  private wake: (() => void) | undefined;
  private closed = false;

  push(event: Record<string, unknown>): void {
    this.pending.push(event);
    this.notify();
  }

  finish(): void {
    this.closed = true;
    this.notify();
  }

  async *drain(): AsyncGenerator<Record<string, unknown>> {
    while (true) {
      while (this.pending.length > 0) {
        yield this.pending.shift()!;
      }
      if (this.closed) return;
      // No await between the emptiness check and this assignment, so a push
      // cannot slip through unnoticed.
      await new Promise<void>(resolve => {
        this.wake = resolve;
      });
    }
  }

  private notify(): void {
    const wake = this.wake;
    this.wake = undefined;
    wake?.();
  }
}

/**
 * Bounds a model call by the query's signal and its request deadline.
 *
 * `ModelRequest` already carries both, but honouring them is up to the client,
 * and a custom one that ignores them would otherwise hang the agent loop with no
 * error and no log line. Losing the race abandons the call rather than
 * cancelling it: the underlying work may continue, but the loop stops waiting.
 */
function withDeadline<T>(
  promise: Promise<T>,
  deadline: { signal?: AbortSignal; timeoutMs?: number },
): Promise<T> {
  const { signal, timeoutMs } = deadline;
  if (!signal && timeoutMs === undefined) return promise;

  let timer: ReturnType<typeof setTimeout> | undefined;
  let onAbort: (() => void) | undefined;
  const tripped = new Promise<never>((_, reject) => {
    if (signal) {
      if (signal.aborted) {
        reject(new AbortError("Operation was aborted"));
        return;
      }
      onAbort = () => reject(new AbortError("Operation was aborted"));
      signal.addEventListener("abort", onAbort, { once: true });
    }
    if (timeoutMs !== undefined) {
      timer = setTimeout(
        () => reject(new TimeoutError(`Model request exceeded requestTimeoutMs of ${timeoutMs}ms`)),
        timeoutMs,
      );
    }
  });
  // The loser keeps running; swallow its result so an abandoned rejection does
  // not surface as an unhandled rejection.
  promise.catch(() => {});

  return Promise.race([promise, tripped]).finally(() => {
    if (timer !== undefined) clearTimeout(timer);
    if (signal && onAbort) signal.removeEventListener("abort", onAbort);
  });
}

function normalizeRequestTimeout(timeoutMs: number | undefined, source: string): number | undefined {
  if (timeoutMs === undefined) return undefined;
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1) {
    throw new Error(`${source} must be a positive integer number of milliseconds`);
  }
  return timeoutMs;
}

type SettledOutcome<T> =
  | { ok: true; value: T; settledAt: string }
  | { ok: false; error: unknown; settledAt: string };

/**
 * Turns a promise into a never-rejecting outcome and records when it settled, so
 * callers can await it after other work without losing the real completion time.
 */
function settle<T>(promise: Promise<T>, onSettled: () => void): Promise<SettledOutcome<T>> {
  return promise.then(
    value => {
      const settledAt = new Date().toISOString();
      onSettled();
      return { ok: true as const, value, settledAt };
    },
    error => {
      const settledAt = new Date().toISOString();
      onSettled();
      return { ok: false as const, error, settledAt };
    },
  );
}

class AnthropicModelClient implements ModelClient {
  private readonly client: Anthropic;

  constructor(options: { apiKey?: string; baseURL?: string }) {
    this.client = new Anthropic({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
    });
  }

  async createMessage(request: ModelRequest): Promise<AssistantModelMessage> {
    try {
      const body = {
        model: request.model,
        ...(request.systemPrompt ? { system: request.systemPrompt } : {}),
        max_tokens: request.maxTokens,
        messages: request.messages.map(toAnthropicMessage) as never,
        tools: request.tools.map(tool => ({
          name: tool.name,
          description: tool.description,
          input_schema: tool.input_schema,
        })) as never,
        ...(request.outputFormat ? { output_config: { format: toAnthropicOutputFormat(request.outputFormat) } } : {}),
        ...(request.thinkingConfig && request.thinkingConfig.type !== "disabled"
          ? { thinking: toAnthropicThinkingConfig(request.thinkingConfig, request.maxTokens) }
          : {}),
        ...(request.reasoningEffort
          ? { reasoning_effort: request.reasoningEffort }
          : {}),
      };
      if (request.stream) {
        const stream = await this.createAnthropicMessage(
          { ...body, stream: true },
          { signal: request.signal, ...(request.timeoutMs ? { timeout: request.timeoutMs } : {}) },
        );
        const assembler = new AnthropicStreamAssembler();
        for await (const event of stream as AsyncIterable<Record<string, unknown>>) {
          request.onStreamEvent?.(event);
          assembler.add(event);
        }
        return assembler.message();
      }

      const response = await this.createAnthropicMessage(body, {
        signal: request.signal,
        ...(request.timeoutMs ? { timeout: request.timeoutMs } : {}),
      });
      if ("type" in response && response.type === "message") {
        const usage = mergeUsage(undefined, response.usage);
        return {
          role: "assistant",
          content: response.content.map(fromAnthropicBlock).filter(isContentBlock),
          ...(usage ? { usage } : {}),
          ...(typeof response.stop_reason === "string" ? { stopReason: response.stop_reason } : {}),
          ...(typeof response.id === "string" ? { providerResponseId: response.id } : {}),
          ...(typeof response.model === "string" ? { model: response.model } : {}),
        };
      }
      throw new APIError("Unexpected streaming response from Anthropic client");
    } catch (error) {
      if (abortErrorIfNeeded(request.signal)) {
        throw abortErrorIfNeeded(request.signal);
      }
      if (error instanceof AgentSDKError) throw error;
      throw new APIError(errorMessage(error), { cause: error });
    }
  }

  private createAnthropicMessage(
    body: Record<string, unknown>,
    options: { signal?: AbortSignal; timeout?: number },
  ): Promise<AnthropicMessageResponse | AsyncIterable<Record<string, unknown>>> {
    if (body.output_config) {
      return this.client.beta.messages.create(
        {
          ...body,
          betas: ["structured-outputs-2025-12-15"],
        } as never,
        options,
      ) as Promise<AnthropicMessageResponse | AsyncIterable<Record<string, unknown>>>;
    }
    return this.client.messages.create(body as never, options) as Promise<AnthropicMessageResponse | AsyncIterable<Record<string, unknown>>>;
  }
}

type AnthropicMessageResponse = {
  type?: unknown;
  id?: unknown;
  model?: unknown;
  content: unknown[];
  usage?: unknown;
  stop_reason?: unknown;
};

/**
 * Folds a provider usage object into a running total. Anthropic reports input
 * counts on message_start and output counts on message_delta, so the two have to
 * be merged rather than replaced.
 */
function mergeUsage(current: TokenUsage | undefined, reported: unknown): TokenUsage | undefined {
  if (!reported || typeof reported !== "object") return current;
  const source = reported as Record<string, unknown>;
  const field = (name: string): number | undefined =>
    typeof source[name] === "number" ? (source[name] as number) : undefined;
  const next: TokenUsage = {
    input_tokens: field("input_tokens") ?? current?.input_tokens ?? 0,
    output_tokens: field("output_tokens") ?? current?.output_tokens ?? 0,
  };
  const cacheCreation = field("cache_creation_input_tokens") ?? current?.cache_creation_input_tokens;
  const cacheRead = field("cache_read_input_tokens") ?? current?.cache_read_input_tokens;
  if (cacheCreation !== undefined) next.cache_creation_input_tokens = cacheCreation;
  if (cacheRead !== undefined) next.cache_read_input_tokens = cacheRead;
  return next;
}

type QueryTotals = { usage: TokenUsage; stopReason?: StopReason };

function normalizeAutoCompact(
  option: boolean | AutoCompactOptions | undefined,
): Required<Omit<AutoCompactOptions, "model">> & { model?: string } | undefined {
  if (!option) return undefined;
  const options = option === true ? {} : option;
  const thresholdTokens = options.thresholdTokens ?? 100_000;
  const keepRecentMessages = options.keepRecentMessages ?? 6;
  if (!Number.isInteger(thresholdTokens) || thresholdTokens < 1) {
    throw new Error("AgentOptions.autoCompact.thresholdTokens must be a positive integer");
  }
  if (!Number.isInteger(keepRecentMessages) || keepRecentMessages < 1) {
    throw new Error("AgentOptions.autoCompact.keepRecentMessages must be a positive integer");
  }
  return {
    thresholdTokens,
    keepRecentMessages,
    prompt: options.prompt ?? DEFAULT_COMPACTION_PROMPT,
    maxTokens: options.maxTokens ?? 8192,
    ...(options.model ? { model: options.model } : {}),
  };
}

/**
 * Whether a failed model call ran out of context, as opposed to failing for any
 * other reason.
 *
 * An input that already exceeds the window is rejected before generation, so it
 * arrives as an API error rather than a stop reason. Providers word this
 * differently and the SDK sees only the message, so the match is deliberately
 * narrow: a false negative surfaces the original error, while a false positive
 * would rewrite history in response to an unrelated failure.
 */
function isContextOverflowError(error: unknown): boolean {
  const message = errorMessage(error).toLowerCase();
  return (
    message.includes("prompt is too long") ||
    message.includes("context window") ||
    message.includes("request_too_large") ||
    message.includes("too many tokens")
  );
}

function isToolResultMessage(message: ModelMessage): boolean {
  return (
    message.role === "user" &&
    Array.isArray(message.content) &&
    message.content.some(block => block.type === "tool_result")
  );
}

/**
 * Picks where history can be cut so the retained tail stays valid on its own.
 *
 * A `tool_result` whose matching `tool_use` was summarized away is rejected by
 * the model API, so the cut walks backwards off any tool-result message until it
 * lands on the assistant turn that issued the call. Returns 0 when no safe cut
 * leaves anything to summarize.
 */
function compactionSplitIndex(messages: ModelMessage[], keepRecent: number): number {
  let index = Math.max(0, messages.length - keepRecent);
  while (index > 0 && isToolResultMessage(messages[index]!)) {
    index--;
  }
  return index;
}

/** The note that turns a summary into a usable first turn for the next request. */
function compactionHandoffMessage(summary: string): ModelMessage {
  return {
    role: "user",
    content:
      "Context compaction has just been performed. The conversation history above this point " +
      "was replaced by the following summary. Continue the task from here as if you had the " +
      "full history, relying on this summary for everything that came before.\n\n" +
      `<conversation_summary>\n${summary}\n</conversation_summary>`,
  };
}

function emptyUsage(): TokenUsage {
  return { input_tokens: 0, output_tokens: 0 };
}

/** Running total across the model requests in one query. */
function addUsage(total: TokenUsage, turn: TokenUsage | undefined): TokenUsage {
  if (!turn) return total;
  const next: TokenUsage = {
    input_tokens: total.input_tokens + turn.input_tokens,
    output_tokens: total.output_tokens + turn.output_tokens,
  };
  const cacheCreation =
    (total.cache_creation_input_tokens ?? 0) + (turn.cache_creation_input_tokens ?? 0);
  const cacheRead = (total.cache_read_input_tokens ?? 0) + (turn.cache_read_input_tokens ?? 0);
  if (cacheCreation > 0) next.cache_creation_input_tokens = cacheCreation;
  if (cacheRead > 0) next.cache_read_input_tokens = cacheRead;
  return next;
}

function toAnthropicOutputFormat(outputFormat: OutputFormat): BetaJSONOutputFormat {
  if (outputFormat === "json") {
    return {
      type: "json_schema",
      schema: {},
    };
  }
  return outputFormat;
}

function toAnthropicThinkingConfig(
  thinkingConfig: Exclude<ThinkingConfig, { type: "disabled" }>,
  maxTokens: number,
): Record<string, unknown> {
  if (thinkingConfig.type === "adaptive") {
    return { type: "adaptive" };
  }
  return {
    type: "enabled",
    budget_tokens: Math.min(thinkingConfig.budgetTokens, maxTokens - 1),
  };
}

class AnthropicStreamAssembler {
  private content: ContentBlock[] = [];
  private currentIndex: number | undefined;
  private jsonDeltas = new Map<number, string>();
  private usage: TokenUsage | undefined;
  private stopReason: StopReason | undefined;
  private providerResponseId: string | undefined;
  private model: string | undefined;

  add(event: Record<string, unknown>): void {
    // Usage arrives split across the stream: input counts up front, output
    // counts at the end alongside stop_reason.
    if (event.type === "message_start") {
      const message = event.message as Record<string, unknown> | undefined;
      this.usage = mergeUsage(this.usage, message?.usage);
      if (typeof message?.id === "string") this.providerResponseId = message.id;
      if (typeof message?.model === "string") this.model = message.model;
      return;
    }

    if (event.type === "message_delta") {
      this.usage = mergeUsage(this.usage, event.usage);
      const delta = event.delta as Record<string, unknown> | undefined;
      if (typeof delta?.stop_reason === "string") {
        this.stopReason = delta.stop_reason;
      }
      return;
    }

    if (event.type === "content_block_start") {
      const index = typeof event.index === "number" ? event.index : this.content.length;
      const block = fromAnthropicBlock(event.content_block);
      if (block) {
        this.content[index] = block;
        this.currentIndex = index;
      }
      return;
    }

    if (event.type !== "content_block_delta") {
      return;
    }

    const index = typeof event.index === "number" ? event.index : this.currentIndex;
    if (index === undefined) return;
    const delta = event.delta as Record<string, unknown> | undefined;
    if (!delta) return;

    if (delta.type === "text_delta" && typeof delta.text === "string") {
      const existing = this.content[index];
      if (existing?.type === "text") {
        existing.text += delta.text;
      } else {
        this.content[index] = { type: "text", text: delta.text };
      }
    }

    if (delta.type === "thinking_delta" && typeof delta.thinking === "string") {
      const existing = this.content[index];
      if (existing?.type === "thinking") {
        existing.thinking += delta.thinking;
      } else {
        this.content[index] = { type: "thinking", thinking: delta.thinking, signature: "" };
      }
    }

    if (delta.type === "signature_delta" && typeof delta.signature === "string") {
      const existing = this.content[index];
      if (existing?.type === "thinking") {
        existing.signature += delta.signature;
      }
    }

    if (delta.type === "input_json_delta" && typeof delta.partial_json === "string") {
      const next = (this.jsonDeltas.get(index) ?? "") + delta.partial_json;
      this.jsonDeltas.set(index, next);
      const existing = this.content[index];
      if (existing?.type === "tool_use") {
        try {
          existing.input = JSON.parse(next) as Record<string, unknown>;
        } catch {
          existing.input = {};
        }
      }
    }
  }

  message(): AssistantModelMessage {
    return {
      role: "assistant",
      content: this.content.filter(isContentBlock),
      ...(this.usage ? { usage: this.usage } : {}),
      ...(this.stopReason ? { stopReason: this.stopReason } : {}),
      ...(this.providerResponseId ? { providerResponseId: this.providerResponseId } : {}),
      ...(this.model ? { model: this.model } : {}),
    };
  }
}

function toAnthropicMessage(message: ModelMessage) {
  return {
    role: message.role,
    content:
      typeof message.content === "string"
        ? message.content
        : message.content.map(block => {
            if (block.type === "tool_result") {
              return {
                type: "tool_result" as const,
                tool_use_id: block.tool_use_id,
                content: block.content,
                ...(block.is_error ? { is_error: true } : {}),
              };
            }
            return block;
          }),
  };
}

function fromAnthropicBlock(block: unknown): ContentBlock | undefined {
  if (!block || typeof block !== "object") return undefined;
  const value = block as Record<string, unknown>;
  if (value.type === "text" && typeof value.text === "string") {
    return { type: "text", text: value.text };
  }
  if (
    value.type === "thinking" &&
    typeof value.thinking === "string" &&
    typeof value.signature === "string"
  ) {
    return { type: "thinking", thinking: value.thinking, signature: value.signature };
  }
  if (value.type === "redacted_thinking" && typeof value.data === "string") {
    return { type: "redacted_thinking", data: value.data };
  }
  if (
    value.type === "tool_use" &&
    typeof value.id === "string" &&
    typeof value.name === "string" &&
    value.input &&
    typeof value.input === "object"
  ) {
    return {
      type: "tool_use",
      id: value.id,
      name: value.name,
      input: value.input as Record<string, unknown>,
    };
  }
  return undefined;
}

type InferInput<TSchema> = TSchema extends { parse(input: unknown): infer TInput }
  ? TInput
  : Record<string, unknown>;

function parseWithSchema(schema: unknown, input: unknown): unknown {
  if (schema && typeof schema === "object" && "parse" in schema && typeof schema.parse === "function") {
    return schema.parse(input);
  }
  return input;
}

function isToolCapableAgentLike(agent: AgentLike<any>): agent is ToolCapableAgentLike<any> {
  return "addTools" in agent && typeof agent.addTools === "function";
}

function schemaToJSONSchema(schema: unknown): Record<string, unknown> {
  if (schema && typeof schema === "object" && "parse" in schema) {
    return toJSONSchema(schema as never) as Record<string, unknown>;
  }
  if (schema && typeof schema === "object") {
    return schema as Record<string, unknown>;
  }
  return { type: "object", additionalProperties: true };
}

function parseSkillMarkdown(raw: string): { frontmatter: Record<string, string>; body: string } {
  const normalized = raw.replace(/^\uFEFF/, "");
  if (!normalized.startsWith("---\n") && !normalized.startsWith("---\r\n")) {
    return { frontmatter: {}, body: normalized };
  }

  const match = normalized.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match) {
    return { frontmatter: {}, body: normalized };
  }

  return {
    frontmatter: parseSimpleFrontmatter(match[1] ?? ""),
    body: normalized.slice(match[0].length),
  };
}

function parseSimpleFrontmatter(input: string): Record<string, string> {
  const output: Record<string, string> = {};
  for (const line of input.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const separator = trimmed.indexOf(":");
    if (separator === -1) continue;
    const key = trimmed.slice(0, separator).trim();
    const value = trimmed.slice(separator + 1).trim();
    if (!key) continue;
    output[key] = stripQuotes(value);
  }
  return output;
}

function stripQuotes(value: string): string {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

function normalizeForMatch(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function promptTextForMatching(prompt: string | ContentBlock[]): string {
  if (typeof prompt === "string") return prompt;
  return prompt
    .filter((block): block is TextBlock => block.type === "text")
    .map(block => block.text)
    .join("\n");
}

function formatSkillInstructions(skills: SkillDefinition[]): string {
  return [
    "The following skills are relevant to the user's request. Follow their instructions when helpful.",
    "",
    ...skills.map(definition => [
      `<skill name="${escapeXmlAttribute(definition.name)}">`,
      `Description: ${definition.description}`,
      "",
      definition.instructions,
      "</skill>",
    ].join("\n")),
  ].join("\n\n");
}

function escapeXmlAttribute(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function sanitizeToolName(name: string): string {
  const sanitized = name.replace(/[^a-zA-Z0-9_-]/g, "_").replace(/_+/g, "_");
  return sanitized || "mcp_tool";
}

function sanitizeWorkspaceSegment(name: string): string {
  const sanitized = name.replace(/[^a-zA-Z0-9._-]/g, "_").replace(/_+/g, "_");
  return sanitized || "agent";
}

class AsyncMessageQueue<T> {
  private readonly values: T[] = [];
  private readonly resolvers: Array<(value: IteratorResult<T>) => void> = [];
  private closed = false;

  push(value: T): void {
    if (this.closed) return;
    const resolver = this.resolvers.shift();
    if (resolver) {
      resolver({ value, done: false });
      return;
    }
    this.values.push(value);
  }

  close(): void {
    this.closed = true;
    while (this.resolvers.length > 0) {
      this.resolvers.shift()?.({ value: undefined, done: true });
    }
  }

  next(): Promise<IteratorResult<T>> {
    const value = this.values.shift();
    if (value) {
      return Promise.resolve({ value, done: false });
    }
    if (this.closed) {
      return Promise.resolve({ value: undefined, done: true });
    }
    return new Promise(resolve => {
      this.resolvers.push(resolve);
    });
  }

  drainAvailable(): T[] {
    return this.values.splice(0);
  }
}

type AgentLikeMessageEmitter = (
  message: AgentLikeEvent,
  context: {
    source: AgentRuntimeSource;
    hasAcceptedHandoffs: boolean;
    emit(message: TeamRunnerMessage): void;
  },
) => void;

async function runTeamInvocation(input: {
  agent: AgentLike<any>;
  prompt: string | ContentBlock[];
  mailbox: TeamMailbox;
  source: AgentRuntimeSource;
  emit(message: TeamRunnerMessage): void;
  maxDelegateDepth: number;
  maxConcurrentWorkItems: number;
  depth: number;
  queryOptions: QueryOptions;
  emitAgentMessage: AgentLikeMessageEmitter;
}): Promise<SDKResultMessage> {
  const tracer = input.queryOptions.tracer;
  const inheritedTraceContext = getQueryTraceContext(input.queryOptions);
  const sessionId = inheritedTraceContext?.sessionId ?? randomId();
  const runId = randomId();
  const traceBase = {
    session_id: sessionId,
    run_id: runId,
    ...(inheritedTraceContext ? { parent_run_id: inheritedTraceContext.parentRunId } : {}),
    source: input.source,
  };

  await emitTraceEvent(tracer, {
    ...traceBase,
    type: "run_start",
    data: {
      runtime: "team",
      team: input.source.team ?? input.source.name,
      max_delegate_depth: input.maxDelegateDepth,
      max_concurrent_work_items: input.maxConcurrentWorkItems,
    },
  });
  await emitTraceEvent(tracer, {
    ...traceBase,
    type: "user_message",
    data: {
      message: {
        role: "user",
        content: input.prompt,
      },
    },
  });

  const childQueryOptions = withQueryTraceContext(input.queryOptions, {
    sessionId,
    parentRunId: runId,
  });

  try {
    const result = await runAgentLikeToFinal({
      ...input,
      queryOptions: childQueryOptions,
    });
    await emitTraceEvent(tracer, {
      ...traceBase,
      type: "result",
      data: traceResultData(result),
    });
    await flushTracer(tracer);
    return result;
  } catch (error) {
    const normalized = error instanceof Error ? error : new Error(String(error));
    await emitTraceEvent(tracer, {
      ...traceBase,
      type: "error",
      data: { error: normalized },
    });
    await emitTraceEvent(tracer, {
      ...traceBase,
      type: "result",
      data: {
        subtype: normalized instanceof AbortError ? "error_abort" : "error",
        is_error: true,
        result: "",
        num_turns: 0,
        error: normalized,
      },
    });
    await flushTracer(tracer);
    throw error;
  }
}

async function runAgentLikeToFinal(input: {
  agent: AgentLike<any>;
  prompt: string | ContentBlock[];
  mailbox: TeamMailbox;
  source: AgentRuntimeSource;
  emit(message: TeamRunnerMessage): void;
  maxDelegateDepth: number;
  maxConcurrentWorkItems: number;
  depth: number;
  queryOptions: QueryOptions;
  emitAgentMessage: AgentLikeMessageEmitter;
}): Promise<SDKResultMessage> {
  let nextPrompt = input.prompt;
  let handoffRounds = 0;

  while (true) {
    const abortError = abortErrorIfNeeded(input.queryOptions.signal);
    if (abortError) throw abortError;

    const acceptedHandoffs: AcceptedDelegateWork[] = [];
    const agentRuntime = createTeamRunnerRuntime({
      mailbox: input.mailbox,
      source: input.source,
      emit: input.emit,
      maxDelegateDepth: input.maxDelegateDepth,
      maxConcurrentWorkItems: input.maxConcurrentWorkItems,
      depth: input.depth,
      queryOptions: input.queryOptions,
      acceptedHandoffs,
    });
    let finalResult: SDKResultMessage | undefined;

    const agentQueryOptions = withInheritedQueryTraceContext(input.queryOptions, {
      stream: input.queryOptions.stream,
      outputFormat: input.queryOptions.outputFormat,
      signal: input.queryOptions.signal,
      context: input.queryOptions.context,
      tracer: input.queryOptions.tracer,
      agentRuntime,
      permissions: agentRuntime.permissions,
    });
    for await (const agentMessage of input.agent.query(nextPrompt, agentQueryOptions)) {
      if (agentMessage.type === "result") {
        finalResult = agentMessage;
      }
      input.emitAgentMessage(agentMessage, {
        source: input.source,
        hasAcceptedHandoffs: acceptedHandoffs.length > 0,
        emit: input.emit,
      });
    }

    if (!finalResult) {
      throw new Error("AgentLike query completed without a result message");
    }
    if (finalResult.is_error) {
      await cancelAcceptedDelegateWork(acceptedHandoffs, input.mailbox, input.emit);
      return finalResult;
    }
    if (acceptedHandoffs.length === 0) {
      return finalResult;
    }

    handoffRounds++;
    if (handoffRounds > 50) {
      await cancelAcceptedDelegateWork(acceptedHandoffs, input.mailbox, input.emit);
      throw new Error("Team runner exceeded maximum handoff follow-up rounds (50)");
    }

    const delegateResults = await executeAcceptedDelegateWork({
      handoffs: acceptedHandoffs,
      mailbox: input.mailbox,
      emit: input.emit,
      maxDelegateDepth: input.maxDelegateDepth,
      maxConcurrentWorkItems: input.maxConcurrentWorkItems,
      queryOptions: input.queryOptions,
    });
    const replies = delegateResults.flatMap(result => result.reply ? [result.reply] : []);

    if (replies.length === 0) {
      return finalResult;
    }

    nextPrompt = renderAcceptedHandoffReplyPrompt(input.source, replies);
  }
}

async function executeAcceptedDelegateWork(input: {
  handoffs: AcceptedDelegateWork[];
  mailbox: TeamMailbox;
  emit(message: TeamRunnerMessage): void;
  maxDelegateDepth: number;
  maxConcurrentWorkItems: number;
  queryOptions: QueryOptions;
}): Promise<AgentRuntimeDelegateResult[]> {
  const pending = new Set(input.handoffs.map((_, index) => index));
  const activeTargets = new Set<string>();
  const inFlight = new Map<number, Promise<{
    index: number;
    ok: true;
    result: AgentRuntimeDelegateResult;
  } | {
    index: number;
    ok: false;
    error: unknown;
  }>>();
  const results: Array<AgentRuntimeDelegateResult | undefined> = new Array(input.handoffs.length);
  const runtimeAbort = new AbortController();
  const signal = combineAbortSignals(input.queryOptions.signal, runtimeAbort.signal);
  let hasFatalError = false;
  let fatalError: unknown;

  const start = (index: number): void => {
    const handoff = input.handoffs[index]!;
    pending.delete(index);
    activeTargets.add(handoff.targetMailbox);
    const execution = executeDelegateWork({
      ...handoff,
      mailbox: input.mailbox,
      emit: input.emit,
      maxDelegateDepth: input.maxDelegateDepth,
      maxConcurrentWorkItems: input.maxConcurrentWorkItems,
      queryOptions: { ...input.queryOptions, signal },
    }).then(
      result => ({ index, ok: true as const, result }),
      error => ({ index, ok: false as const, error }),
    );
    inFlight.set(index, execution);
  };

  while (pending.size > 0 || inFlight.size > 0) {
    const abortError = abortErrorIfNeeded(signal);
    if (abortError) {
      hasFatalError = true;
      fatalError = abortError;
      break;
    }

    while (inFlight.size < input.maxConcurrentWorkItems) {
      const nextIndex = [...pending].find(index =>
        !activeTargets.has(input.handoffs[index]!.targetMailbox)
      );
      if (nextIndex === undefined) break;
      start(nextIndex);
    }

    if (inFlight.size === 0) break;
    const settled = await Promise.race(inFlight.values());
    const handoff = input.handoffs[settled.index]!;
    inFlight.delete(settled.index);
    activeTargets.delete(handoff.targetMailbox);
    if (settled.ok) {
      results[settled.index] = settled.result;
    } else {
      hasFatalError = true;
      fatalError = settled.error;
      break;
    }
  }

  if (hasFatalError) {
    runtimeAbort.abort(fatalError);
    const remaining = await Promise.all(inFlight.values());
    for (const settled of remaining) {
      if (settled.ok) results[settled.index] = settled.result;
    }
    await cancelAcceptedDelegateWork(input.handoffs, input.mailbox, input.emit);
    throw fatalError;
  }

  return results.map((result, index) => {
    if (!result) throw new Error(`Accepted handoff ${index} completed without a result`);
    return result;
  });
}

async function executeDelegateWork(input: AcceptedDelegateWork & {
  mailbox: TeamMailbox;
  emit(message: TeamRunnerMessage): void;
  maxDelegateDepth: number;
  maxConcurrentWorkItems: number;
  queryOptions: QueryOptions;
}): Promise<AgentRuntimeDelegateResult> {
  const claimed = await input.mailbox.claimNext(input.targetMailbox);
  const message = claimed ?? input.request;
  input.emit({
    type: "team_message",
    subtype: "claimed",
    source: input.callerSource,
    mailbox: input.targetMailbox,
    message,
  });

  input.emit({
    type: "team_agent",
    subtype: "started",
    source: input.targetSource,
  });

  const childPermissions: RuntimePermissions = { workspaceGrants: input.workspaceGrants };
  let finalResult: SDKResultMessage;
  try {
    finalResult = await runAgentLikeToFinal({
      agent: input.delegateInput.agent,
      prompt: renderDelegateTaskPrompt(input.delegateInput, message),
      mailbox: input.mailbox,
      source: input.targetSource,
      emit: input.emit,
      maxDelegateDepth: input.maxDelegateDepth,
      maxConcurrentWorkItems: input.maxConcurrentWorkItems,
      depth: input.depth + 1,
      queryOptions: {
        ...input.queryOptions,
        permissions: childPermissions,
      },
      emitAgentMessage: emitChildAgentMessage,
    });
  } catch (error) {
    if (error instanceof AbortError || abortErrorIfNeeded(input.queryOptions.signal)) {
      await cancelDelegateWork(input, message);
      throw error;
    }
    return failDelegateWork(input, message, error);
  } finally {
    input.emit({
      type: "team_agent",
      subtype: "finished",
      source: input.targetSource,
    });
  }

  if (finalResult.is_error) {
    if (finalResult.subtype === "error_abort" || finalResult.error instanceof AbortError) {
      await cancelDelegateWork(input, message);
      throw finalResult.error ?? new AbortError(finalResult.result || "Delegated work aborted");
    }
    return failDelegateWork(
      input,
      message,
      finalResult.error ?? new Error(finalResult.result || `Delegate ${input.delegateInput.name} failed`),
      finalResult,
    );
  }

  await input.mailbox.updateStatus(message.id, "done");
  const done = await input.mailbox.get(message.id);
  if (done) {
    input.emit({
      type: "team_message",
      subtype: "done",
      source: input.targetSource,
      mailbox: input.targetMailbox,
      message: done,
    });
  }
  const reply = await input.mailbox.send(input.targetMailbox, input.callerMailbox, finalResult.result, {
    threadId: message.threadId,
    parentMessageId: message.id,
    workItemId: message.workItemId,
    upstreamMessageId: message.upstreamMessageId ?? message.id,
    workItemRole: "upstream_report",
  });
  input.emit({
    type: "team_message",
    subtype: "replied",
    source: input.targetSource,
    mailbox: input.callerMailbox,
    message: reply,
  });

  return {
    status: "completed",
    content: finalResult.result,
    request: message,
    reply,
    result: finalResult,
    workspaceGrants: input.workspaceGrants,
  };
}

async function failDelegateWork(
  input: AcceptedDelegateWork & {
    mailbox: TeamMailbox;
    emit(message: TeamRunnerMessage): void;
  },
  message: TeamMessage,
  error: unknown,
  result?: SDKResultMessage,
): Promise<AgentRuntimeDelegateResult> {
  const failure = normalizeAgentRuntimeFailure(error);
  await input.mailbox.updateStatus(message.id, "failed");
  const failed = await input.mailbox.get(message.id) ?? { ...message, status: "failed" as const };
  input.emit({
    type: "team_message",
    subtype: "failed",
    source: input.targetSource,
    mailbox: input.targetMailbox,
    message: failed,
    error: failure,
  });

  const content = [
    `Delegated work assigned to ${input.delegateInput.name} failed.`,
    `Error code: ${failure.code}`,
    `Reason: ${failure.message}`,
    "Decide whether to retry, revise the task, continue with partial results, or finish without this work item.",
  ].join("\n");
  const reply = await input.mailbox.send(input.targetMailbox, input.callerMailbox, content, {
    threadId: message.threadId,
    parentMessageId: message.id,
    workItemId: message.workItemId,
    upstreamMessageId: message.upstreamMessageId ?? message.id,
    workItemRole: "upstream_report",
    metadata: { status: "failed", error: failure },
  });
  input.emit({
    type: "team_message",
    subtype: "replied",
    source: input.targetSource,
    mailbox: input.callerMailbox,
    message: reply,
    error: failure,
  });

  return {
    status: "failed",
    content,
    request: failed,
    reply,
    result,
    error: failure,
    workspaceGrants: input.workspaceGrants,
  };
}

async function cancelDelegateWork(
  input: AcceptedDelegateWork & {
    mailbox: TeamMailbox;
    emit(message: TeamRunnerMessage): void;
  },
  message: TeamMessage,
): Promise<void> {
  await input.mailbox.updateStatus(message.id, "cancelled");
  const cancelled = await input.mailbox.get(message.id) ?? { ...message, status: "cancelled" as const };
  input.emit({
    type: "team_message",
    subtype: "cancelled",
    source: input.targetSource,
    mailbox: input.targetMailbox,
    message: cancelled,
  });
}

async function cancelAcceptedDelegateWork(
  handoffs: AcceptedDelegateWork[],
  mailbox: TeamMailbox,
  emit: (message: TeamRunnerMessage) => void,
): Promise<void> {
  for (const handoff of handoffs) {
    const current = await mailbox.get(handoff.request.id);
    if (!current || !["pending", "processing"].includes(current.status)) continue;
    await cancelDelegateWork({ ...handoff, mailbox, emit }, current);
  }
}

function normalizeAgentRuntimeFailure(error: unknown): AgentRuntimeFailure {
  const normalized = error instanceof Error ? error : new Error(String(error));
  const code = normalized instanceof MaxTurnsError
    ? "max_turns_exceeded"
    : normalized instanceof APIError
      ? "api_error"
      : normalized instanceof ToolPermissionDeniedError
        ? "permission_denied"
        : normalized instanceof ToolExecutionError
          ? "tool_execution_error"
          : "agent_error";
  return {
    code,
    message: normalized.message,
    name: normalized.name || "Error",
  };
}

function emitRootAgentMessage(message: AgentLikeEvent, context: {
  source: AgentRuntimeSource;
  hasAcceptedHandoffs: boolean;
  emit(message: TeamRunnerMessage): void;
}): void {
  if (isNestedTeamRunnerMessage(message)) {
    context.emit(message);
    return;
  }
  if (message.type === "result" && !message.is_error && context.hasAcceptedHandoffs) {
    context.emit({
      type: "agent_message",
      source: context.source,
      message,
    });
    return;
  }
  context.emit(message);
}

function emitChildAgentMessage(message: AgentLikeEvent, context: {
  source: AgentRuntimeSource;
  hasAcceptedHandoffs: boolean;
  emit(message: TeamRunnerMessage): void;
}): void {
  if (isNestedTeamRunnerMessage(message)) {
    context.emit(message);
    return;
  }
  if (message.type === "stream_event") {
    context.emit({
      type: "stream_event",
      source: context.source,
      event: message.event,
      session_id: message.session_id,
    });
    return;
  }
  context.emit({
    type: "agent_message",
    source: context.source,
    message,
  });
}

function createTeamRunnerRuntime(input: {
  mailbox: TeamMailbox;
  source: AgentRuntimeSource;
  emit(message: TeamRunnerMessage): void;
  maxDelegateDepth: number;
  maxConcurrentWorkItems: number;
  depth: number;
  queryOptions: QueryOptions;
  acceptedHandoffs: AcceptedDelegateWork[];
}): AgentRuntimeContext {
  const permissions: RuntimePermissions = normalizeRuntimePermissions(input.queryOptions.permissions);
  return {
    source: input.source,
    permissions,
    emit: input.emit,
    shouldPauseAfterToolBatch: () => input.acceptedHandoffs.length > 0,
    async delegate(delegateInput) {
      if (input.depth >= input.maxDelegateDepth) {
        throw new Error(`Reached maximum delegate depth (${input.maxDelegateDepth})`);
      }

      const targetMailbox = delegateInput.targetMailboxId ?? sanitizeToolName(delegateInput.name);
      const callerMailbox = input.source.mailbox ?? input.source.name ?? "manager";
      const targetSource: AgentRuntimeSource = {
        kind: "team_member",
        name: delegateInput.name,
        ...(input.source.team ? { team: input.source.team } : {}),
        member: delegateInput.name,
        mailbox: targetMailbox,
      };
      const workspaceGrants = authorizeDelegatedWorkspaceGrants({
        requested: delegateInput.workspaceGrants ?? [],
        parentPermissions: permissions,
        grantor: input.source,
        grantee: targetSource,
      });
      const request = await input.mailbox.send(callerMailbox, targetMailbox, delegateInput.task, {
        workItemRole: "delegation",
        ...(workspaceGrants.length > 0 ? { metadata: { workspaceGrants } } : {}),
      });
      input.emit({
        type: "team_message",
        subtype: "sent",
        source: input.source,
        mailbox: targetMailbox,
        message: request,
      });

      const delegateWork: AcceptedDelegateWork = {
        delegateInput,
        request,
        callerMailbox,
        targetMailbox,
        callerSource: input.source,
        targetSource,
        workspaceGrants,
        depth: input.depth,
      };

      if (delegateInput.wait === "accepted") {
        input.acceptedHandoffs.push(delegateWork);
        return {
          status: "accepted",
          content: formatDelegateAccepted(request, workspaceGrants),
          request,
          workspaceGrants,
        };
      }

      return executeDelegateWork({
        ...delegateWork,
        mailbox: input.mailbox,
        emit: input.emit,
        maxDelegateDepth: input.maxDelegateDepth,
        maxConcurrentWorkItems: input.maxConcurrentWorkItems,
        queryOptions: input.queryOptions,
      });
    },
  };
}

function formatTeamMemberDelegateDescription(member: TeamMemberDefinition): string {
  const details = [
    `Delegate work to team member ${member.name}.`,
    `Role: ${member.role}.`,
    ...(member.focus ? [`Focus: ${member.focus}.`] : []),
    "Pass a clear task. The member will return a final result.",
    "By default, ask for durable deliverables to be written in the member's own private workspace and reported in natural language with important workspace paths and verification notes.",
    "If you instruct the member to write into any shared or manager-owned path, include workspaceGrants with access=[\"write\"] for that exact shared root. Do not name a write destination outside the member's workspace unless you also grant it.",
  ];
  return details.join(" ");
}

function formatAgentToolDescription(description: string, freshSessionPerCall: boolean): string {
  return [
    description,
    "",
    "This tool calls another AgentLike. Choose the action mode explicitly:",
    '- mode="ask": ask the AgentLike and wait for its final answer in this tool call.',
    '- mode="handoff": queue work and receive an acceptance receipt immediately; requires a team/runtime context. The receipt means queued, not completed. After the current tool batch, the team runtime runs the member and resumes you with a completed or failed report. Do not call tools that depend on the delegated result before that report arrives.',
    '- mode="observe": request observable long-running work; currently unsupported and will return a clear error.',
    freshSessionPerCall
      ? "Statefulness: every call spawns a fresh session with no memory of previous calls. Make each task self-contained."
      : "Statefulness: this target is a long-lived session that retains conversation history across calls. Earlier tasks may influence its answers.",
    "Provide a clear task, expected output, and acceptance criteria when useful.",
    "If the target needs to write in a shared workspace, include workspaceGrants with root, access, and reason. The runtime will only grant write access that the caller is already allowed to delegate, and the result will say which grants were accepted or why they were denied. Read-only tools do not require workspace grants.",
    "Choose one workspace strategy explicitly: either ask the target to write deliverables in its own private workspace and report paths, or provide workspaceGrants for every shared/manager-owned root you ask it to write under.",
  ].join("\n");
}

function formatAgentToolTask(input: AgentToolInput): string {
  const lines = [
    input.task,
  ];
  if (input.expectedOutput) {
    lines.push("", "Expected output:", input.expectedOutput);
  }
  if (input.acceptanceCriteria?.length) {
    lines.push("", "Acceptance criteria:");
    for (const criterion of input.acceptanceCriteria) {
      lines.push(`- ${criterion}`);
    }
  }
  return lines.join("\n");
}

function isNestedTeamRunnerMessage(message: AgentLikeEvent): message is Exclude<TeamRunnerMessage, SDKMessage> {
  if (message.type === "team_message" || message.type === "team_agent" || message.type === "agent_message") {
    return true;
  }
  return message.type === "stream_event" && "source" in message;
}

function renderDelegateTaskPrompt(input: AgentRuntimeDelegateInput, message: TeamMessage): string {
  const workspaceGrants = workspaceGrantsFromMessage(message);
  return [
    "You are handling delegated work sent through an agent mailbox.",
    "",
    `delegate: ${input.name}`,
    ...(input.description ? [`description: ${input.description}`] : []),
    `message_id: ${message.id}`,
    `from: ${message.from}`,
    `to: ${message.to}`,
    `thread: ${message.threadId}`,
    "",
    "Return the final result as assistant text. If you have AgentLike tools, you may use them to ask or hand off work to other AgentLike workers.",
    "Write durable deliverables in your own workspace. When handing off results, mention the important workspace paths and a brief verification summary in natural language.",
    "Do not use structured artifact reference objects as the collaboration protocol.",
    ...(workspaceGrants.length > 0
      ? ["", formatWorkspaceGrantsForPrompt(workspaceGrants)]
      : []),
    "",
    "--- delegated task ---",
    message.content,
  ].join("\n");
}

function renderAcceptedHandoffReplyPrompt(source: AgentRuntimeSource, replies: TeamMessage[]): string {
  return [
    "Team runtime update:",
    "Previously accepted delegated work has reported back through the mailbox. Use these replies to continue the original task.",
    "If more work is needed, you may call your AgentLike tools again. When ready, return the final delivery to your caller.",
    "Do not answer with only an assignment, acceptance receipt, or in-progress status.",
    "",
    `mailbox: ${source.mailbox ?? source.name ?? "manager"}`,
    "",
    ...replies.map(reply => [
      "--- mailbox reply ---",
      `id: ${reply.id}`,
      `from: ${reply.from}`,
      `to: ${reply.to}`,
      `status: ${reply.status}`,
      `thread: ${reply.threadId}`,
      `work_item: ${reply.workItemId ?? ""}`,
      `role: ${reply.workItemRole ?? ""}`,
      `reply_to: ${reply.parentMessageId ?? ""}`,
      "",
      reply.content,
    ].join("\n")),
  ].join("\n");
}

function formatDelegateAccepted(message: TeamMessage, workspaceGrants: WorkspaceGrant[] = []): string {
  return JSON.stringify({
    status: "accepted",
    phase: "queued",
    completion_pending: true,
    message_id: message.id,
    work_item_id: message.workItemId,
    thread_id: message.threadId,
    to: message.to,
    ...(workspaceGrants.length > 0
      ? { workspaceGrants: workspaceGrants.map(formatWorkspaceGrantForReceipt) }
      : {}),
  }, null, 2);
}

function formatToolBatchRejection(rejection: ToolBatchPolicyRejection): string {
  return JSON.stringify({
    status: "rejected",
    code: rejection.code,
    message: rejection.message,
    ...(rejection.conflictingToolCallIds
      ? { conflicting_tool_call_ids: rejection.conflictingToolCallIds }
      : {}),
    ...(rejection.suggestedNextStep
      ? { suggested_next_step: rejection.suggestedNextStep }
      : {}),
  }, null, 2);
}

function workspaceGrantsFromMessage(message: TeamMessage): WorkspaceGrant[] {
  const value = message.metadata?.workspaceGrants;
  if (!Array.isArray(value)) return [];
  return normalizeWorkspaceGrantInputs(value.filter(isWorkspaceGrantInput));
}

function isWorkspaceGrantInput(value: unknown): value is WorkspaceGrantInput {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return typeof record.root === "string";
}

function formatWorkspaceGrantsForPrompt(grants: WorkspaceGrant[]): string {
  return [
    "Workspace access granted for this delegated task:",
    ...grants.flatMap((grant, index) => [
      `- grant ${index + 1}: ${grant.root}`,
      `  access: ${grant.access.join(", ")}`,
      ...(grant.reason ? [`  reason: ${grant.reason}`] : []),
      "  Use this shared workspace only for the delegated task. Read-only tools may inspect any path; write tools must stay under an allowed root or granted shared workspace.",
    ]),
  ].join("\n");
}

function formatWorkspaceGrantForReceipt(grant: WorkspaceGrant): Record<string, unknown> {
  return {
    root: grant.root,
    access: grant.access,
    ...(grant.reason ? { reason: grant.reason } : {}),
    ...(grant.expiresAt ? { expiresAt: grant.expiresAt } : {}),
  };
}

function createTeamTools(options: {
  mailbox: TeamMailbox;
  ownerMailboxId: string;
  resolveMailbox(target: string): string;
}): Array<ToolDefinition<any, any>> {
  const owner = options.ownerMailboxId;
  return [
    tool(
      "team_send",
      "Send an asynchronous team message to another member mailbox.",
      z.object({
        to: z.string(),
        content: z.string(),
      }),
      async input => {
        const target = options.resolveMailbox(input.to);
        const message = await options.mailbox.send(owner, target, input.content);
        return { content: formatTeamMessageReceipt("sent", message) };
      },
    ),
    tool(
      "team_inbox",
      "List pending messages in your team mailbox.",
      z.object({
        status: z.enum(["pending", "processing", "done", "failed", "cancelled", "read", "all"]).optional(),
      }),
      async input => {
        const messages = await options.mailbox.inbox(owner, { status: input.status ?? "pending" });
        return { content: formatTeamInbox(messages) };
      },
    ),
    tool(
      "team_read",
      "Read and claim a team mailbox message by id.",
      z.object({
        message_id: z.string(),
      }),
      async input => {
        const message = await options.mailbox.get(input.message_id);
        if (!message) throw new Error(`Team message ${input.message_id} was not found`);
        if (message.to !== owner) {
          throw new Error(`Team message ${input.message_id} does not belong to ${owner}`);
        }
        if (message.status === "pending") {
          await options.mailbox.updateStatus(message.id, "processing");
        }
        return { content: formatTeamMessage(message) };
      },
    ),
    tool(
      "team_reply",
      "Send a final reply to a team mailbox message and mark the original message done.",
      z.object({
        message_id: z.string(),
        content: z.string(),
      }),
      async input => {
        const message = await options.mailbox.get(input.message_id);
        if (!message) throw new Error(`Team message ${input.message_id} was not found`);
        if (message.to !== owner) {
          throw new Error(`Team message ${input.message_id} does not belong to ${owner}`);
        }
        if (message.status === "done" || message.status === "cancelled") {
          return { content: `Team message ${message.id} is already ${message.status}` };
        }
        await options.mailbox.updateStatus(message.id, "done");
        const reply = await options.mailbox.send(owner, message.from, input.content, {
          threadId: message.threadId,
          parentMessageId: message.id,
          workItemId: message.workItemId,
          upstreamMessageId: message.upstreamMessageId,
          workItemRole: message.from === "manager" ? "upstream_report" : "downstream_reply",
        });
        return { content: formatTeamMessageReceipt("replied", reply) };
      },
    ),
    tool(
      "team_followup",
      "Send a progress update for a team mailbox message without closing it.",
      z.object({
        message_id: z.string(),
        content: z.string(),
      }),
      async input => {
        const message = await options.mailbox.get(input.message_id);
        if (!message) throw new Error(`Team message ${input.message_id} was not found`);
        if (message.to !== owner) {
          throw new Error(`Team message ${input.message_id} does not belong to ${owner}`);
        }
        const followup = await options.mailbox.send(owner, message.from, input.content, {
          threadId: message.threadId,
          parentMessageId: message.id,
          workItemId: message.workItemId,
          upstreamMessageId: message.upstreamMessageId,
          workItemRole: "followup",
        });
        return { content: formatTeamMessageReceipt("followup_sent", followup) };
      },
    ),
    tool(
      "team_status",
      "Summarize your team mailbox status.",
      z.object({}),
      async () => {
        const all = await options.mailbox.inbox(owner, { status: "all" });
        const counts = all.reduce<Record<string, number>>((acc, message) => {
          acc[message.status] = (acc[message.status] ?? 0) + 1;
          return acc;
        }, {});
        return { content: JSON.stringify({ mailbox: owner, counts }, null, 2) };
      },
    ),
  ];
}

async function drainTeam(input: {
  members: TeamMemberDefinition[];
  mailbox: TeamMailbox;
  options: TeamDrainOptions;
}): Promise<TeamDrainResult> {
  const maxRounds = input.options.maxRounds ?? 10;
  const maxMessages = input.options.maxMessages ?? 50;
  let processed = 0;
  let failed = 0;
  let rounds = 0;

  while (rounds < maxRounds && processed < maxMessages) {
    const abortError = abortErrorIfNeeded(input.options.signal);
    if (abortError) throw abortError;
    rounds++;
    let claimedThisRound = 0;

    for (const member of input.members) {
      if (processed >= maxMessages) break;
      const mailboxId = member.mailboxId ?? member.name;
      const message = await input.mailbox.claimNext(mailboxId);
      if (!message) continue;
      claimedThisRound++;
      processed++;

      try {
        const mustUseTeamTools = isToolCapableAgentLike(member.agent);
        const messagePermissions: RuntimePermissions = {
          workspaceGrants: workspaceGrantsFromMessage(message),
        };
        const result = await member.agent.prompt(renderTeamTaskPrompt(member, message), {
          signal: input.options.signal,
          permissions: messagePermissions,
        });
        const current = await input.mailbox.get(message.id);
        if (result.is_error) {
          failed++;
          await failUnclosedTeamMessage(input.mailbox, message, mailboxId, result.result || result.error?.message);
        } else if (current?.status === "processing") {
          if (mustUseTeamTools) {
            failed++;
            await failUnclosedTeamMessage(input.mailbox, message, mailboxId, result.result || result.error?.message);
          } else {
            await completeTeamMessageFromResult(input.mailbox, message, mailboxId, result.result);
          }
        }
      } catch (error) {
        failed++;
        await failUnclosedTeamMessage(input.mailbox, message, mailboxId, errorMessage(error));
      }
    }

    if (claimedThisRound === 0) break;
  }

  return { processed, failed, rounds };
}

function renderTeamTaskPrompt(member: TeamMemberDefinition, message: TeamMessage): string {
  const canUseTeamTools = isToolCapableAgentLike(member.agent);
  const workspaceGrants = workspaceGrantsFromMessage(message);
  const closeLoopInstructions = canUseTeamTools
    ? [
        "You must close the loop through team tools:",
        `- Call team_reply(message_id="${message.id}", content="...") when you have the final result.`,
        `- Call team_followup(message_id="${message.id}", content="...") if you need to report progress or ask a question.`,
        "- Do not finish with only plain assistant text.",
      ]
    : [
        "Return your final result as the assistant response.",
        "The parent team will record that result as your upstream reply.",
      ];

  return [
    "You are processing a team mailbox message assigned specifically to you.",
    "",
    `member: ${member.name}`,
    `role: ${member.role}`,
    ...(member.focus ? [`focus: ${member.focus}`] : []),
    "",
    `message_id: ${message.id}`,
    `from: ${message.from}`,
    `to: ${message.to}`,
    `thread: ${message.threadId}`,
    `work_item: ${message.workItemId ?? ""}`,
    `work_item_role: ${message.workItemRole ?? ""}`,
    "",
    ...closeLoopInstructions,
    "Write durable deliverables in your own workspace. In final replies, mention the important workspace paths and a brief verification summary in natural language.",
    "Do not use structured artifact reference objects as the collaboration protocol.",
    ...(workspaceGrants.length > 0
      ? ["", formatWorkspaceGrantsForPrompt(workspaceGrants)]
      : []),
    "",
    "--- message content ---",
    message.content,
  ].join("\n");
}

async function completeTeamMessageFromResult(
  mailbox: TeamMailbox,
  message: TeamMessage,
  ownerMailboxId: string,
  content: string,
): Promise<void> {
  const current = await mailbox.get(message.id);
  if (!current || current.status !== "processing") return;
  await mailbox.updateStatus(message.id, "done");
  await mailbox.send(ownerMailboxId, message.from, content, {
    threadId: message.threadId,
    parentMessageId: message.id,
    workItemId: message.workItemId,
    upstreamMessageId: message.upstreamMessageId,
    workItemRole: "upstream_report",
  });
}

async function failUnclosedTeamMessage(
  mailbox: TeamMailbox,
  message: TeamMessage,
  ownerMailboxId: string,
  reason?: string,
): Promise<void> {
  const current = await mailbox.get(message.id);
  if (!current || current.status !== "processing") return;
  await mailbox.updateStatus(message.id, "failed");
  await mailbox.send(
    ownerMailboxId,
    message.from,
    `Team member ${ownerMailboxId} ended without team_reply or team_followup for message ${message.id}.${reason ? ` Reason: ${reason}` : ""}`,
    {
      threadId: message.threadId,
      parentMessageId: message.id,
      workItemId: message.workItemId,
      upstreamMessageId: message.upstreamMessageId,
      workItemRole: "followup",
    },
  );
}

function teamMailboxId(teamName: string, memberName: string): string {
  return `${teamName}::${sanitizeToolName(memberName)}`;
}

function resolveTeamMailbox(
  teamName: string,
  members: TeamMemberDefinition[],
  target: string,
): string {
  if (target === "manager" || target.includes("::")) return target;
  const normalized = sanitizeToolName(target);
  const member = members.find(item => item.name === normalized || item.mailboxId === target);
  return member?.mailboxId ?? teamMailboxId(teamName, normalized);
}

function inferTeamMessageRole(
  from: string,
  to: string,
  inherited?: TeamMessage,
): TeamMessageRole {
  if (from === "manager") return "upstream_request";
  if (to === "manager") return "upstream_report";
  if (inherited?.workItemRole === "executor_result") return "downstream_reply";
  if (inherited?.workItemId) return "executor_result";
  return "delegation";
}

function formatTeamInbox(messages: TeamMessage[]): string {
  if (messages.length === 0) return "No team mailbox messages.";
  return messages.map(formatTeamMessage).join("\n\n");
}

function formatTeamMessage(message: TeamMessage): string {
  return [
    `id: ${message.id}`,
    `from: ${message.from}`,
    `to: ${message.to}`,
    `status: ${message.status}`,
    `thread: ${message.threadId}`,
    `work_item: ${message.workItemId ?? ""}`,
    `role: ${message.workItemRole ?? ""}`,
    "",
    message.content,
  ].join("\n");
}

function formatTeamMessageReceipt(action: string, message: TeamMessage): string {
  return [
    `team_message_${action}`,
    `id: ${message.id}`,
    `to: ${message.to}`,
    `thread: ${message.threadId}`,
    `work_item: ${message.workItemId ?? ""}`,
  ].join("\n");
}

function initializeSQLiteMailbox(database: SQLiteDatabaseLike, table: string): void {
  const sql = `CREATE TABLE IF NOT EXISTS ${table} (
    id TEXT PRIMARY KEY,
    from_mailbox TEXT NOT NULL,
    to_mailbox TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    thread_id TEXT NOT NULL,
    parent_message_id TEXT,
    work_item_id TEXT,
    work_item_role TEXT,
    upstream_message_id TEXT,
    metadata_json TEXT
  )`;
  if (database.exec) {
    database.exec(sql);
  } else {
    database.prepare(sql).run();
  }
  database.prepare(`CREATE INDEX IF NOT EXISTS ${table}_to_status_created_idx ON ${table} (to_mailbox, status, created_at)`).run();
}

function normalizeSQLiteIdentifier(identifier: string): string {
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(identifier)) {
    throw new Error(`Invalid SQLite table name: ${identifier}`);
  }
  return identifier;
}

function rowToTeamMessage(row: unknown): TeamMessage | undefined {
  if (!row || typeof row !== "object") return undefined;
  const value = row as Record<string, unknown>;
  if (
    typeof value.id !== "string" ||
    typeof value.from_mailbox !== "string" ||
    typeof value.to_mailbox !== "string" ||
    typeof value.content !== "string" ||
    typeof value.status !== "string" ||
    typeof value.created_at !== "number" ||
    typeof value.thread_id !== "string"
  ) {
    return undefined;
  }
  return {
    id: value.id,
    from: value.from_mailbox,
    to: value.to_mailbox,
    content: value.content,
    status: value.status as TeamMessageStatus,
    createdAt: value.created_at,
    threadId: value.thread_id,
    ...(typeof value.parent_message_id === "string" ? { parentMessageId: value.parent_message_id } : {}),
    ...(typeof value.work_item_id === "string" ? { workItemId: value.work_item_id } : {}),
    ...(typeof value.work_item_role === "string" ? { workItemRole: value.work_item_role as TeamMessageRole } : {}),
    ...(typeof value.upstream_message_id === "string" ? { upstreamMessageId: value.upstream_message_id } : {}),
    ...(typeof value.metadata_json === "string" ? { metadata: parseMetadataJSON(value.metadata_json) } : {}),
  };
}

function isTeamMessage(message: TeamMessage | undefined): message is TeamMessage {
  return message !== undefined;
}

function parseMetadataJSON(value: string): Record<string, unknown> | undefined {
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined;
  } catch {
    return undefined;
  }
}

function sqliteChanges(result: unknown): number {
  if (result && typeof result === "object" && "changes" in result && typeof result.changes === "number") {
    return result.changes;
  }
  return 0;
}

function formatMCPToolResult(result: MCPCallToolResult): string {
  const blocks = result.content ?? [];
  const renderedBlocks = blocks.map(block => {
    if (block.type === "text" && typeof block.text === "string") {
      return block.text;
    }
    return JSON.stringify(block);
  });
  if (result.structuredContent !== undefined) {
    renderedBlocks.push(JSON.stringify(result.structuredContent));
  }
  return renderedBlocks.join("\n");
}

function toMCPCallToolResult(value: unknown): MCPCallToolResult {
  if (!value || typeof value !== "object") {
    return { content: [{ type: "text", text: String(value) }] };
  }
  const result = value as Record<string, unknown>;
  return {
    content: Array.isArray(result.content) ? result.content as MCPContentBlock[] : [],
    ...(result.structuredContent !== undefined ? { structuredContent: result.structuredContent } : {}),
    ...(typeof result.isError === "boolean" ? { isError: result.isError } : {}),
  };
}

function isToolUseBlock(block: ContentBlock): block is ToolUseBlock {
  return block.type === "tool_use";
}

function isContentBlock(block: ContentBlock | undefined): block is ContentBlock {
  return block !== undefined;
}

function extractText(message: AssistantModelMessage): string {
  return message.content
    .filter((block): block is TextBlock => block.type === "text")
    .map(block => block.text)
    .join("");
}

function toolResultText(content: string | ContentBlock[]): string {
  if (typeof content === "string") return content;
  return content
    .filter((block): block is TextBlock => block.type === "text")
    .map(block => block.text)
    .join("");
}

function abortErrorIfNeeded(signal: AbortSignal | undefined): AbortError | undefined {
  return signal?.aborted ? new AbortError("Operation aborted") : undefined;
}

function getQueryTraceContext(options: QueryOptions): QueryTraceContext | undefined {
  return (options as InternalQueryOptions)[QUERY_TRACE_CONTEXT];
}

function withQueryTraceContext<TContext>(
  options: QueryOptions<TContext>,
  traceContext: QueryTraceContext,
): QueryOptions<TContext> {
  return {
    ...options,
    [QUERY_TRACE_CONTEXT]: traceContext,
  } as InternalQueryOptions<TContext>;
}

function withInheritedQueryTraceContext<TContext>(
  source: QueryOptions,
  target: QueryOptions<TContext>,
): QueryOptions<TContext> {
  const traceContext = getQueryTraceContext(source);
  return traceContext ? withQueryTraceContext(target, traceContext) : target;
}

function combineAbortSignals(...signals: Array<AbortSignal | undefined>): AbortSignal {
  const active = signals.filter((signal): signal is AbortSignal => signal !== undefined);
  if (active.length === 1) return active[0]!;
  return AbortSignal.any(active);
}

async function startLangSmithRootRun(
  event: ContextTraceEvent,
  options: LangSmithContextTracerOptions,
  makeRunTree: (config: LangSmithRunTreeConfig) => LangSmithRunTreeLike,
  runs: Map<string, LangSmithTraceRunState>,
): Promise<LangSmithTraceRunState> {
  const existing = runs.get(event.run_id);
  if (existing) return existing;

  const parent = event.parent_run_id ? runs.get(event.parent_run_id)?.root : undefined;
  const config: LangSmithRunTreeConfig = {
    name: options.name ?? langSmithSourceName(event.source),
    run_type: "chain",
    ...(isUuidLike(event.run_id) ? { id: event.run_id } : {}),
    ...(options.projectName ? { project_name: options.projectName } : {}),
    ...(parent ? { parent_run: parent } : {}),
    start_time: event.timestamp,
    inputs: {},
    metadata: langSmithMetadata(event, options, {
      model: event.data.model,
      tools: event.data.tools,
      agent_session_id: event.data.agent_session_id,
      runtime: event.data.runtime,
      team: event.data.team,
    }),
    tags: langSmithTags(event, options),
    ...langSmithConnectionConfig(options),
  };
  const root = parent ? parent.createChild(config) : makeRunTree(config);
  const state: LangSmithTraceRunState = {
    root,
    initialInputRecorded: false,
    modelRequests: 0,
    toolUses: 0,
    pendingTools: new Map(),
  };
  runs.set(event.run_id, state);
  await root.postRun?.(true);
  return state;
}

async function ensureLangSmithRootRun(
  event: ContextTraceEvent,
  options: LangSmithContextTracerOptions,
  makeRunTree: (config: LangSmithRunTreeConfig) => LangSmithRunTreeLike,
  runs: Map<string, LangSmithTraceRunState>,
): Promise<LangSmithTraceRunState> {
  return runs.get(event.run_id) ?? startLangSmithRootRun(
    {
      ...event,
      type: "run_start",
      data: {},
    },
    options,
    makeRunTree,
    runs,
  );
}

function recordLangSmithUserMessage(
  state: LangSmithTraceRunState,
  event: ContextTraceEvent,
): void {
  const message = event.data.message;
  if (!state.initialInputRecorded) {
    state.root.inputs = { message: jsonSafeValue(message) };
    state.initialInputRecorded = true;
    return;
  }
  recordLangSmithRunEvent(state.root, event);
}

async function startLangSmithModelRun(
  state: LangSmithTraceRunState,
  event: ContextTraceEvent,
  options: LangSmithContextTracerOptions,
): Promise<void> {
  state.modelRequests++;
  const model = typeof event.data.model === "string" ? event.data.model : "model";
  const run = state.root.createChild({
    name: `${model} turn ${state.modelRequests}`,
    run_type: "llm",
    start_time: event.timestamp,
    inputs: langSmithModelInputs(event.data),
    metadata: langSmithMetadata(event, options, {
      sdk_event_type: "model_request",
      model: event.data.model,
      max_tokens: event.data.max_tokens,
      stream: event.data.stream,
    }),
    tags: langSmithTags(event, options, ["run:llm"]),
  });
  state.pendingModel = run;
  await run.postRun?.(true);
}

async function finishLangSmithModelRun(
  state: LangSmithTraceRunState,
  event: ContextTraceEvent,
): Promise<void> {
  const run = state.pendingModel;
  if (!run) {
    recordLangSmithRunEvent(state.root, event);
    return;
  }
  state.pendingModel = undefined;
  await run.end?.(
    { message: jsonSafeValue(event.data.message) },
    undefined,
    Date.parse(event.timestamp),
    {
      sdk_end_event_type: "assistant_message",
      sdk_end_seq: event.seq,
    },
  );
  await run.patchRun?.({ excludeInputs: false });
}

async function startLangSmithToolRun(
  state: LangSmithTraceRunState,
  event: ContextTraceEvent,
  options: LangSmithContextTracerOptions,
): Promise<void> {
  state.toolUses++;
  const toolUseId = typeof event.data.id === "string"
    ? event.data.id
    : `tool_${state.toolUses}`;
  const toolName = typeof event.data.name === "string"
    ? event.data.name
    : "tool";
  const run = state.root.createChild({
    name: toolName,
    run_type: "tool",
    start_time: event.timestamp,
    inputs: {
      input: jsonSafeValue(event.data.input),
    },
    metadata: langSmithMetadata(event, options, {
      sdk_event_type: "tool_use",
      tool_use_id: toolUseId,
      tool_name: toolName,
      ...(typeof event.data.description === "string"
        ? { tool_description: event.data.description }
        : {}),
    }),
    tags: langSmithTags(event, options, ["run:tool", `tool:${toolName}`]),
  });
  state.pendingTools.set(toolUseId, run);
  await run.postRun?.(true);
}

async function finishLangSmithToolRun(
  state: LangSmithTraceRunState,
  event: ContextTraceEvent,
): Promise<void> {
  const toolUseId = typeof event.data.tool_use_id === "string"
    ? event.data.tool_use_id
    : undefined;
  const run = toolUseId ? state.pendingTools.get(toolUseId) : undefined;
  if (!run) {
    recordLangSmithRunEvent(state.root, event);
    return;
  }
  state.pendingTools.delete(toolUseId!);
  const isError = event.data.is_error === true;
  await run.end?.(
    {
      content: jsonSafeValue(event.data.content),
      is_error: isError,
    },
    isError ? langSmithErrorMessage(event.data) : undefined,
    Date.parse(event.timestamp),
    {
      sdk_end_event_type: "tool_result",
      sdk_end_seq: event.seq,
    },
  );
  await run.patchRun?.({ excludeInputs: false });
}

async function finishLangSmithRootRun(
  state: LangSmithTraceRunState,
  event: ContextTraceEvent,
): Promise<void> {
  const isError = event.data.is_error === true;
  await state.root.end?.(
    jsonSafeRecord(event.data),
    isError ? langSmithErrorMessage(event.data) : undefined,
    Date.parse(event.timestamp),
    {
      sdk_end_event_type: "result",
      sdk_end_seq: event.seq,
    },
  );
  await state.root.patchRun?.({ excludeInputs: false });
}

function recordLangSmithRunEvent(
  run: LangSmithRunTreeLike,
  event: ContextTraceEvent,
): void {
  run.addEvent?.({
    name: event.type,
    time: event.timestamp,
    kwargs: {
      sdk_trace_version: event.version,
      sdk_trace_seq: event.seq,
      sdk_session_id: event.session_id,
      sdk_run_id: event.run_id,
      sdk_parent_run_id: event.parent_run_id,
      source: jsonSafeValue(event.source),
      data: jsonSafeValue(event.data),
    },
  });
}

async function flushLangSmithClients(
  options: LangSmithContextTracerOptions,
  runs: Map<string, LangSmithTraceRunState>,
): Promise<void> {
  const clients = new Set<LangSmithFlushableClient>();
  addLangSmithClient(clients, options.client);
  for (const replica of [langSmithWriteReplica(options)]) {
    if (replica && !Array.isArray(replica)) {
      addLangSmithClient(clients, replica.client);
    }
  }
  for (const state of runs.values()) {
    collectLangSmithRunClients(state.root, clients);
    if (state.pendingModel) {
      collectLangSmithRunClients(state.pendingModel, clients);
    }
    for (const run of state.pendingTools.values()) {
      collectLangSmithRunClients(run, clients);
    }
  }

  for (const client of clients) {
    await client.flush?.();
    await client.awaitPendingTraceBatches?.();
  }
}

function collectLangSmithRunClients(
  run: LangSmithRunTreeLike,
  clients: Set<LangSmithFlushableClient>,
): void {
  addLangSmithClient(clients, run.client);
  for (const replica of run.replicas ?? []) {
    if (Array.isArray(replica)) continue;
    addLangSmithClient(clients, replica.client);
  }
  for (const child of run.child_runs ?? []) {
    collectLangSmithRunClients(child, clients);
  }
}

function addLangSmithClient(
  clients: Set<LangSmithFlushableClient>,
  value: unknown,
): void {
  if (!value || typeof value !== "object") return;
  const candidate = value as LangSmithFlushableClient;
  if (typeof candidate.flush === "function" || typeof candidate.awaitPendingTraceBatches === "function") {
    clients.add(candidate);
  }
}

function langSmithSourceName(source: AgentRuntimeSource): string {
  return source.name ?? source.member ?? source.team ?? source.kind;
}

function langSmithConnectionConfig(options: LangSmithContextTracerOptions): Pick<LangSmithRunTreeConfig, "client" | "replicas"> {
  const replica = langSmithWriteReplica(options);
  return {
    ...(options.client ? { client: options.client } : {}),
    ...(replica ? { replicas: [replica] } : {}),
  };
}

function langSmithWriteReplica(options: LangSmithContextTracerOptions): LangSmithWriteReplicaConfig | undefined {
  if (!options.apiUrl && !options.apiKey && !options.workspaceId) return undefined;
  return {
    ...(options.projectName ? { projectName: options.projectName } : {}),
    ...(options.apiUrl ? { apiUrl: options.apiUrl } : {}),
    ...(options.apiKey ? { apiKey: options.apiKey } : {}),
    ...(options.workspaceId ? { workspaceId: options.workspaceId } : {}),
    ...(options.client ? { client: options.client } : {}),
  };
}

function langSmithMetadata(
  event: ContextTraceEvent,
  options: LangSmithContextTracerOptions,
  extra: LangSmithKVMap = {},
): LangSmithKVMap {
  return {
    ...(options.metadata ?? {}),
    sdk_trace_version: event.version,
    sdk_trace_seq: event.seq,
    sdk_session_id: event.session_id,
    sdk_run_id: event.run_id,
    sdk_parent_run_id: event.parent_run_id,
    sdk_source_kind: event.source.kind,
    sdk_source_name: event.source.name,
    sdk_source_team: event.source.team,
    sdk_source_member: event.source.member,
    sdk_source_mailbox: event.source.mailbox,
    ...extra,
  };
}

function langSmithTags(
  event: ContextTraceEvent,
  options: LangSmithContextTracerOptions,
  extra: string[] = [],
): string[] {
  const tags = new Set([
    "agent-lattice",
    "context-trace",
    `source:${event.source.kind}`,
    ...extra,
    ...(options.tags ?? []),
  ]);
  if (event.source.team) tags.add(`team:${event.source.team}`);
  if (event.source.member) tags.add(`member:${event.source.member}`);
  if (event.source.mailbox) tags.add(`mailbox:${event.source.mailbox}`);
  return [...tags];
}

function jsonSafeRecord(value: unknown): LangSmithKVMap {
  const safe = jsonSafeValue(value);
  if (safe && typeof safe === "object" && !Array.isArray(safe)) {
    return safe as LangSmithKVMap;
  }
  return { value: safe };
}

function langSmithModelInputs(data: Record<string, unknown>): LangSmithKVMap {
  const inputs = jsonSafeRecord(data);
  const systemPrompt = typeof inputs.systemPrompt === "string" && inputs.systemPrompt.trim()
    ? inputs.systemPrompt
    : undefined;
  if (!systemPrompt) return inputs;

  const messages = Array.isArray(inputs.messages) ? inputs.messages : [];
  const firstMessage = messages[0];
  if (isRecord(firstMessage) && firstMessage.role === "system") return inputs;

  inputs.messages = [
    { role: "system", content: systemPrompt },
    ...messages,
  ];
  return inputs;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function jsonSafeValue(value: unknown): unknown {
  if (value === undefined) return undefined;
  try {
    return JSON.parse(JSON.stringify(value, (_key, inner) => {
      if (inner instanceof Error) {
        return {
          name: inner.name,
          message: inner.message,
        };
      }
      return inner;
    }));
  } catch {
    return String(value);
  }
}

function langSmithErrorMessage(data: Record<string, unknown>): string {
  const error = data.error;
  if (error instanceof Error) return error.message;
  if (error && typeof error === "object" && "message" in error) {
    return String((error as { message?: unknown }).message);
  }
  if (typeof data.result === "string" && data.result) return data.result;
  if (typeof data.content === "string" && data.content) return data.content;
  return "LangSmith traced SDK event marked as error";
}

function isUuidLike(value: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

const traceSequences = new WeakMap<ContextTracer, number>();

async function emitTraceEvent(
  tracer: ContextTracer | undefined,
  input: Omit<ContextTraceEvent, "version" | "timestamp" | "seq">,
  // When the traced work finished earlier than this call, pass that moment so the
  // trace records the work's duration instead of including later SDK bookkeeping.
  occurredAt?: string,
): Promise<void> {
  if (!tracer) return;
  const nextSeq = (traceSequences.get(tracer) ?? 0) + 1;
  traceSequences.set(tracer, nextSeq);
  const event: ContextTraceEvent = {
    version: 1,
    timestamp: occurredAt ?? new Date().toISOString(),
    seq: nextSeq,
    ...input,
  };
  try {
    await tracer.onEvent(event);
  } catch (error) {
    if (shouldPropagatePortError(tracer)) throw error;
  }
}

async function flushTracer(tracer: ContextTracer | undefined): Promise<void> {
  if (!tracer?.flush) return;
  try {
    await tracer.flush();
  } catch (error) {
    if (shouldPropagatePortError(tracer)) throw error;
  }
}

async function appendJsonlEntry(filePath: string, entry: ContextTraceEvent): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  await appendFile(filePath, `${stringifyTraceEntry(entry)}\n`, "utf8");
}

async function appendJsonlHistoryMessage(filePath: string, message: ModelMessage): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  await appendFile(filePath, `${JSON.stringify(message)}\n`, "utf8");
}

async function writeJsonlHistory(filePath: string, messages: ModelMessage[]): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  const body = messages.map(message => JSON.stringify(message)).join("\n");
  await writeFile(filePath, body ? `${body}\n` : "", "utf8");
}

function stringifyTraceEntry(entry: ContextTraceEvent): string {
  return JSON.stringify(entry, (_key, value) => {
    if (value instanceof Error) {
      return {
        name: value.name,
        message: value.message,
      };
    }
    return value;
  });
}

function traceResultData(result: SDKResultMessage): Record<string, unknown> {
  return {
    subtype: result.subtype,
    is_error: result.is_error,
    result: result.result,
    num_turns: result.num_turns,
    ...(result.error ? { error: result.error } : {}),
  };
}

function traceRuntimePermissions(permissions: RuntimePermissions | undefined): Record<string, unknown> {
  return {
    workspaceGrants: activeWorkspaceGrants(permissions).map(grant => ({
      root: grant.root,
      access: grant.access,
      ...(grant.reason ? { reason: grant.reason } : {}),
      ...(grant.workItemId ? { workItemId: grant.workItemId } : {}),
    })),
  };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function randomId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

type AllowedRoots = {
  cwd: string;
  directories: string[];
};

const WORKSPACE_ACCESS_VALUES: WorkspaceAccess[] = ["read", "write", "execute"];
const DEFAULT_WORKSPACE_GRANT_ACCESS: WorkspaceAccess[] = ["write"];

type WorkspaceAuthorizationContext = {
  permissions?: RuntimePermissions;
};

function normalizeAllowedDirectories(options: AgentWorkspaceToolsOptions): AllowedRoots {
  const cwd = resolve(options.cwd ?? process.cwd());
  const directories = (options.allowedDirectories && options.allowedDirectories.length > 0
    ? options.allowedDirectories
    : [cwd]
  ).map(directory => resolve(cwd, directory));
  return { cwd, directories };
}

function resolveAuthorizedPath(
  inputPath: string,
  roots: AllowedRoots,
  context: WorkspaceAuthorizationContext | undefined,
  toolName: string,
  operation: WorkspaceAccess,
): string {
  const path = isAbsolute(inputPath)
    ? resolve(inputPath)
    : resolve(roots.cwd, inputPath);
  const allowedRoots = allowedWorkspaceRootsForOperation(roots.directories, context?.permissions, operation);
  if (!allowedRoots.some(root => pathInRoot(path, root))) {
    throw new ToolPermissionDeniedError(createPathPermissionDenial({
      toolName,
      operation,
      requestedPath: path,
      allowedRoots,
    }));
  }
  return path;
}

function resolveToolPath(inputPath: string, cwd: string): string {
  return isAbsolute(inputPath)
    ? resolve(inputPath)
    : resolve(cwd, inputPath);
}

function allowedWorkspaceRootsForOperation(
  baseRoots: string[],
  permissions: RuntimePermissions | undefined,
  operation: WorkspaceAccess,
): string[] {
  const roots = [
    ...baseRoots,
    ...activeWorkspaceGrants(permissions)
      .filter(grant => grant.access.includes(operation))
      .map(grant => grant.root),
  ];
  return uniqueNormalizedPaths(roots);
}

function activeWorkspaceGrants(permissions: RuntimePermissions | undefined): WorkspaceGrant[] {
  const now = Date.now();
  return normalizeRuntimePermissions(permissions).workspaceGrants
    .filter(grant => grant.expiresAt === undefined || grant.expiresAt > now);
}

function normalizeRuntimePermissions(permissions: RuntimePermissions | undefined): { workspaceGrants: WorkspaceGrant[] } {
  return {
    workspaceGrants: normalizeWorkspaceGrantInputs(permissions?.workspaceGrants ?? []),
  };
}

function normalizeWorkspaceGrantInputs(grants: WorkspaceGrantInput[]): WorkspaceGrant[] {
  return grants.map(grant => normalizeWorkspaceGrantInput(grant));
}

function normalizeWorkspaceGrantInput(
  grant: WorkspaceGrantInput,
  metadata: {
    grantor?: AgentRuntimeSource;
    grantee?: AgentRuntimeSource;
    workItemId?: string;
  } = {},
): WorkspaceGrant {
  return {
    ...grant,
    ...metadata,
    kind: "workspace",
    root: resolve(grant.root),
    access: normalizeWorkspaceAccess(grant.access),
  };
}

function normalizeWorkspaceAccess(access: WorkspaceAccess[] | undefined): WorkspaceAccess[] {
  const requested = access && access.length > 0 ? access : DEFAULT_WORKSPACE_GRANT_ACCESS;
  return WORKSPACE_ACCESS_VALUES.filter(item => requested.includes(item));
}

function uniqueNormalizedPaths(paths: string[]): string[] {
  return [...new Set(paths.map(path => resolve(path)))];
}

function pathInRoot(path: string, root: string): boolean {
  const normalizedPath = resolve(path);
  const normalizedRoot = resolve(root);
  return normalizedPath === normalizedRoot || normalizedPath.startsWith(normalizedRoot + sep);
}

function createPathPermissionDenial(input: {
  toolName: string;
  operation: WorkspaceAccess;
  requestedPath: string;
  allowedRoots: string[];
}): PermissionDenial {
  return {
    status: "permission_denied",
    tool: input.toolName,
    operation: input.operation,
    requestedPath: input.requestedPath,
    reasonCode: "outside_allowed_roots",
    reason: `Path ${input.requestedPath} is outside allowed ${input.operation} roots.`,
    allowedRoots: input.allowedRoots,
    suggestedNextStep: suggestedNextStepForDenial("outside_allowed_roots", input.operation, input.allowedRoots.length > 0),
  };
}

function createGrantPermissionDenial(input: {
  toolName: string;
  operation: WorkspaceAccess;
  requestedPath: string;
  allowedRoots: string[];
}): PermissionDenial {
  return {
    status: "permission_denied",
    tool: input.toolName,
    operation: input.operation,
    requestedPath: input.requestedPath,
    reasonCode: "grant_not_authorized",
    reason: `Cannot grant ${input.operation} access to ${input.requestedPath}; the caller does not have delegable access to that path.`,
    allowedRoots: input.allowedRoots,
    suggestedNextStep: suggestedNextStepForDenial("grant_not_authorized", input.operation, input.allowedRoots.length > 0),
  };
}

function authorizeDelegatedWorkspaceGrants(input: {
  requested: WorkspaceGrantInput[];
  parentPermissions: RuntimePermissions;
  grantor: AgentRuntimeSource;
  grantee: AgentRuntimeSource;
}): WorkspaceGrant[] {
  return input.requested.map(requested => {
    const normalized = normalizeWorkspaceGrantInput(requested, {
      grantor: input.grantor,
      grantee: input.grantee,
    });
    for (const operation of normalized.access) {
      if (operation !== "write") continue;
      const allowedRoots = allowedWorkspaceRootsForOperation([], input.parentPermissions, operation);
      if (!allowedRoots.some(root => pathInRoot(normalized.root, root))) {
        throw new ToolPermissionDeniedError(createGrantPermissionDenial({
          toolName: input.grantee.name ?? input.grantee.member ?? "agentTool",
          operation,
          requestedPath: normalized.root,
          allowedRoots,
        }));
      }
    }
    return normalized;
  });
}

function suggestedNextStepForDenial(
  reasonCode: PermissionDenialReasonCode,
  operation: WorkspaceAccess,
  hasAllowedRoots: boolean,
): string {
  if (reasonCode === "grant_not_authorized") {
    return "Ask the host or your manager for a shared workspace grant before delegating this access.";
  }
  if (reasonCode === "outside_allowed_roots" && hasAllowedRoots) {
    return `Use an allowed ${operation} root, or ask your manager to grant access to the required shared workspace.`;
  }
  return `Ask your manager to grant ${operation} access to the required shared workspace.`;
}

function formatPermissionDeniedToolResult(denial: PermissionDenial): string {
  const operationRootsKey = `allowed${capitalize(denial.operation)}Roots`;
  return JSON.stringify({
    status: denial.status,
    tool: denial.tool,
    ...(denial.requestedPath ? { requestedPath: denial.requestedPath } : {}),
    reason: denial.reason,
    [operationRootsKey]: denial.allowedRoots,
    suggestedNextStep: denial.suggestedNextStep,
  }, null, 2);
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

async function listFiles(root: string): Promise<string[]> {
  const rootStat = await stat(root);
  if (!rootStat.isDirectory()) {
    return [root];
  }
  const files: string[] = [];
  const entries = await readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name === "node_modules" || entry.name === ".git") continue;
    const fullPath = join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await listFiles(fullPath)));
    } else if (entry.isFile()) {
      files.push(fullPath);
    }
  }
  return files;
}

function normalizeSlash(path: string): string {
  return path.split(sep).join("/");
}

function globToRegExp(pattern: string): RegExp {
  let source = "";
  for (let i = 0; i < pattern.length; i++) {
    const char = pattern[i];
    const next = pattern[i + 1];
    if (char === "*" && next === "*") {
      source += ".*";
      i++;
      if (pattern[i + 1] === "/") {
        source += "(?:/)?";
        i++;
      }
      continue;
    }
    if (char === "*") {
      source += "[^/]*";
      continue;
    }
    if (char === "?") {
      source += "[^/]";
      continue;
    }
    source += escapeRegExp(char);
  }
  return new RegExp(`^${source}$`);
}

function escapeRegExp(char: string): string {
  return /[\\^$+?.()|[\]{}]/.test(char) ? `\\${char}` : char;
}

type ShellPathCheck = {
  path: string;
  operation: WorkspaceAccess;
};

function authorizeShellCommand(
  command: string,
  roots: AllowedRoots,
  context: WorkspaceAuthorizationContext | undefined,
): void {
  for (const check of extractShellPathChecks(command)) {
    if (check.operation !== "write") continue;
    if (isShellDiscardTarget(check.path, roots.cwd)) continue;
    resolveAuthorizedPath(check.path, roots, context, "Bash", check.operation);
  }
}

function extractShellPathChecks(command: string): ShellPathCheck[] {
  const checks: ShellPathCheck[] = [];
  for (const redirectedPath of extractShellRedirectPaths(command)) {
    checks.push({ path: redirectedPath, operation: "write" });
  }

  for (const segment of command.split(/&&|\|\||;/)) {
    const words = shellWords(segment);
    if (words.length === 0) continue;
    const commandName = basename(words[0] ?? "");
    const args = commandArgsWithoutRedirections(words.slice(1));
    if (["mkdir", "touch", "rm", "rmdir"].includes(commandName)) {
      for (const path of nonFlagArgs(args)) {
        checks.push({ path, operation: "write" });
      }
      continue;
    }
    if (commandName === "cp") {
      const paths = nonFlagArgs(args);
      if (paths.at(-1)) {
        checks.push({ path: paths.at(-1)!, operation: "write" });
      }
      continue;
    }
    if (commandName === "mv") {
      for (const path of nonFlagArgs(args)) {
        checks.push({ path, operation: "write" });
      }
    }
  }

  return checks.filter(check => looksLikePath(check.path));
}

function isShellDiscardTarget(inputPath: string, cwd: string): boolean {
  const stripped = normalizeShellRedirectTarget(inputPath);
  const path = isAbsolute(stripped) ? resolve(stripped) : resolve(cwd, stripped);
  return path === "/dev/null";
}

function extractShellRedirectPaths(command: string): string[] {
  const paths: string[] = [];
  const redirectPattern = /(?:^|\s)(?:>>|&>|\d?>)\s*("[^"]+"|'[^']+'|[^\s]+)/g;
  let match: RegExpExecArray | null;
  while ((match = redirectPattern.exec(command)) !== null) {
    const path = normalizeShellRedirectTarget(match[1] ?? "");
    if (path) paths.push(path);
  }
  return paths;
}

function normalizeShellRedirectTarget(raw: string): string {
  const trimmed = raw.trim();
  if ((trimmed.startsWith("\"") && trimmed.endsWith("\"")) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
    return stripShellQuotes(trimmed);
  }
  return stripShellQuotes(trimmed).replace(/[;&|]+$/g, "");
}

function shellWords(segment: string): string[] {
  const words: string[] = [];
  const pattern = /"([^"]*)"|'([^']*)'|[^\s]+/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(segment.trim())) !== null) {
    words.push(stripShellQuotes(match[0]));
  }
  return words;
}

function nonFlagArgs(args: string[]): string[] {
  const result: string[] = [];
  let afterDoubleDash = false;
  for (const arg of args) {
    if (afterDoubleDash) {
      result.push(arg);
      continue;
    }
    if (arg === "--") {
      afterDoubleDash = true;
      continue;
    }
    if (!arg.startsWith("-")) {
      result.push(arg);
    }
  }
  return result;
}

function commandArgsWithoutRedirections(args: string[]): string[] {
  const result: string[] = [];
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (!arg) continue;
    if (/^(?:\d?>|>>|&>)$/.test(arg)) {
      index++;
      continue;
    }
    if (/^(?:\d?>|>>|&>)/.test(arg)) {
      continue;
    }
    result.push(arg);
  }
  return result;
}

function firstNonFlagArg(args: string[]): string | undefined {
  return nonFlagArgs(args)[0];
}

function stripShellQuotes(value: string): string {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

function looksLikePath(value: string): boolean {
  if (!value) return false;
  if (value.includes("$")) return false;
  return (
    value === "." ||
    value === ".." ||
    value.startsWith("/") ||
    value.startsWith("./") ||
    value.startsWith("../") ||
    value.includes("/")
  );
}

function runShell(
  command: string,
  options: { cwd: string; timeoutMs: number },
): Promise<string> {
  const shell = process.platform === "win32" ? "cmd.exe" : "/bin/sh";
  const args = process.platform === "win32" ? ["/d", "/s", "/c", command] : ["-c", command];
  return new Promise((resolvePromise, reject) => {
    execFile(
      shell,
      args,
      {
        cwd: options.cwd,
        timeout: options.timeoutMs,
        maxBuffer: 1024 * 1024,
        env: process.env,
      },
      (error, stdout, stderr) => {
        const output = [stdout, stderr].filter(Boolean).join("");
        if (error) {
          reject(new Error(output || error.message));
          return;
        }
        resolvePromise(output);
      },
    );
  });
}
