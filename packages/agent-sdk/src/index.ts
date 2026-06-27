import Anthropic from "@anthropic-ai/sdk";
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
import { execFile } from "node:child_process";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
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

export type TextBlock = { type: "text"; text: string };
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
export type ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock;

export type AgentLikeEvent = SDKMessage | TeamRunnerMessage;

export type AgentLike = {
  query(prompt: string | ContentBlock[], options?: QueryOptions): AsyncGenerator<AgentLikeEvent>;
  prompt(prompt: string | ContentBlock[], options?: QueryOptions): Promise<SDKResultMessage>;
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
  agent: AgentLike;
  task: string;
  wait?: DelegateWaitMode;
  targetMailboxId?: string;
};

export type AgentRuntimeDelegateResult = {
  status: "completed" | "accepted";
  content: string;
  request: TeamMessage;
  reply?: TeamMessage;
  result?: SDKResultMessage;
};

export type AgentRuntimeContext = {
  source: AgentRuntimeSource;
  delegate(input: AgentRuntimeDelegateInput): Promise<AgentRuntimeDelegateResult>;
  emit(message: TeamRunnerMessage): void;
};

type ToolCapableAgentLike = AgentLike & {
  addTools(tools: Array<ToolDefinition<any>>): void;
};

export type ModelMessage = {
  role: "user" | "assistant";
  content: string | ContentBlock[];
};

export type AssistantModelMessage = {
  role: "assistant";
  content: ContentBlock[];
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
  onStreamEvent?: (event: Record<string, unknown>) => void;
  signal?: AbortSignal;
};

export interface ModelClient {
  createMessage(request: ModelRequest): Promise<AssistantModelMessage>;
}

export type ToolResult = {
  content: string | ContentBlock[];
};

export type ToolHandler<TInput = unknown> = (
  input: TInput,
  context: { signal?: AbortSignal; toolUseId: string; runtime?: AgentRuntimeContext },
) => Promise<ToolResult> | ToolResult;

export type ToolDefinition<TInput = unknown> = {
  name: string;
  description: string;
  inputSchema: unknown;
  jsonSchema: Record<string, unknown>;
  parse(input: unknown): TInput;
  handler: ToolHandler<TInput>;
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
  tools: Array<ToolDefinition<any>>;
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
  agent: AgentLike;
  mailboxId?: string;
  workspace?: string;
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

export type TeamOptions = {
  name: string;
  lead: Agent;
  members: TeamMemberDefinition[];
  mailbox?: TeamMailbox;
  exposeLeadMailboxTools?: boolean;
};

export type Team = {
  name: string;
  lead: Agent;
  members: TeamMemberDefinition[];
  mailbox: TeamMailbox;
  tools: Array<ToolDefinition<any>>;
  memberTools: Record<string, Array<ToolDefinition<any>>>;
  send(from: string, to: string, content: string, options?: TeamSendOptions): Promise<TeamMessage>;
  drain(options?: TeamDrainOptions): Promise<TeamDrainResult>;
  query(prompt: string | ContentBlock[], options?: QueryOptions): AsyncGenerator<TeamRunnerMessage>;
  prompt(prompt: string | ContentBlock[], options?: QueryOptions): Promise<SDKResultMessage>;
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

export type TeamRunnerOptions = {
  team?: Team;
  root?: AgentLike;
  mailbox?: TeamMailbox;
  source?: AgentRuntimeSource;
  maxDelegateDepth?: number;
};

export type TeamRunner = {
  root: AgentLike;
  mailbox: TeamMailbox;
  query(prompt: string | ContentBlock[], options?: QueryOptions): AsyncGenerator<TeamRunnerMessage>;
  prompt(prompt: string | ContentBlock[], options?: QueryOptions): Promise<SDKResultMessage>;
};

export type PermissionRequest = {
  toolName: string;
  input: Record<string, unknown>;
  toolUseId: string;
};

export type PermissionDecision =
  | { behavior: "allow" }
  | { behavior: "deny"; message: string };

export type AgentOptions = {
  apiKey?: string;
  baseURL?: string;
  model: string;
  systemPrompt?: string;
  maxTokens?: number;
  maxTurns?: number;
  tools?: Array<ToolDefinition<any>>;
  skills?: SkillDefinition[];
  permission?: (request: PermissionRequest) => Promise<PermissionDecision> | PermissionDecision;
  modelClient?: ModelClient;
};

export type ClaudeCodeToolsOptions = {
  cwd?: string;
  allowedDirectories?: string[];
  bashTimeoutMs?: number;
};

export type QueryOptions = {
  stream?: boolean;
  signal?: AbortSignal;
  runtime?: AgentRuntimeContext;
};

export type SDKSystemInitMessage = {
  type: "system";
  subtype: "init";
  model: string;
  tools: string[];
  session_id: string;
};

export type SDKAssistantMessage = {
  type: "assistant";
  message: AssistantModelMessage;
  session_id: string;
};

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
  subtype: "success" | "error" | "error_max_turns" | "error_abort";
  is_error: boolean;
  result: string;
  session_id: string;
  num_turns: number;
  error?: Error;
};

export type SDKMessage =
  | SDKSystemInitMessage
  | SDKStreamEventMessage
  | SDKAssistantMessage
  | SDKUserMessage
  | SDKResultMessage;

export type TeamRunnerSource = AgentRuntimeSource;

export type TeamRunnerTeamMessage = {
  type: "team_message";
  subtype: "sent" | "claimed" | "replied" | "followup" | "done" | "failed";
  source: TeamRunnerSource;
  mailbox: string;
  message: TeamMessage;
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

export function tool<TSchema>(
  name: string,
  description: string,
  inputSchema: TSchema,
  handler: ToolHandler<InferInput<TSchema>>,
): ToolDefinition<InferInput<TSchema>> {
  return {
    name,
    description,
    inputSchema,
    jsonSchema: schemaToJSONSchema(inputSchema),
    parse(input: unknown): InferInput<TSchema> {
      return parseWithSchema(inputSchema, input) as InferInput<TSchema>;
    },
    handler,
  };
}

export type DelegateToolOptions = {
  wait?: DelegateWaitMode;
  targetMailboxId?: string;
};

export type AgentToolMode = "ask" | "handoff" | "observe";

export const agentToolInputSchema = z.object({
  mode: z.enum(["ask", "handoff", "observe"]),
  task: z.string(),
  expectedOutput: z.string().optional(),
  acceptanceCriteria: z.array(z.string()).optional(),
});

export type AgentToolInput = z.infer<typeof agentToolInputSchema>;

export type AgentToolOptions = {
  description: string;
  targetMailboxId?: string;
};

export function agentTool(
  name: string,
  agent: AgentLike,
  options: AgentToolOptions,
): ToolDefinition<AgentToolInput> {
  const toolName = sanitizeToolName(name);
  return tool(
    toolName,
    formatAgentToolDescription(options.description),
    agentToolInputSchema,
    async (input, context) => {
      const task = formatAgentToolTask(input);
      if (input.mode === "observe") {
        throw new Error(`agentTool("${toolName}") mode=observe is not supported. Available modes: ask, handoff.`);
      }

      if (input.mode === "handoff") {
        if (!context.runtime) {
          throw new Error(`agentTool("${toolName}") mode=handoff requires an AgentRuntime. Available modes without AgentRuntime: ask.`);
        }
        const result = await context.runtime.delegate({
          name: toolName,
          description: options.description,
          agent,
          task,
          wait: "accepted",
          targetMailboxId: options.targetMailboxId,
        });
        return { content: result.content };
      }

      if (context.runtime) {
        const result = await context.runtime.delegate({
          name: toolName,
          description: options.description,
          agent,
          task,
          wait: "result",
          targetMailboxId: options.targetMailboxId,
        });
        return { content: result.content };
      }

      const result = await agent.prompt(task, { signal: context.signal });
      if (result.is_error) {
        throw result.error ?? new Error(result.result || `agentTool("${toolName}") target returned an error`);
      }
      return { content: result.result };
    },
  );
}

export function delegateTool(
  name: string,
  description: string,
  agent: AgentLike,
  options: DelegateToolOptions = {},
): ToolDefinition<{ task: string }> {
  const toolName = sanitizeToolName(name);
  return tool(
    toolName,
    description,
    z.object({
      task: z.string(),
    }),
    async (input, context) => {
      if (!context.runtime) {
        throw new Error(`delegateTool("${toolName}") requires an AgentRuntime. Use createTeamRunner().query() or createTeamRunner().prompt().`);
      }
      const result = await context.runtime.delegate({
        name: toolName,
        description,
        agent,
        task: input.task,
        wait: options.wait,
        targetMailboxId: options.targetMailboxId,
      });
      return { content: result.content };
    },
  );
}

export function createAgent(options: AgentOptions): Agent {
  return new Agent(options);
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
    name: options.clientName ?? "claude-team-agent-sdk",
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
    name: options.clientName ?? "claude-team-agent-sdk",
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
    ...(input.workspace ? { workspace: input.workspace } : {}),
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
  const memberTools: Record<string, Array<ToolDefinition<any>>> = {};

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
    }).query(prompt, queryOptions),
    prompt: (prompt, queryOptions) => createTeamRunner({
      root: options.lead,
      mailbox,
      source: { kind: "root", name, team: name, mailbox: "manager" },
    }).prompt(prompt, queryOptions),
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
  const maxDelegateDepth = options.maxDelegateDepth ?? 8;

  return {
    root,
    mailbox,
    async *query(prompt: string | ContentBlock[], queryOptions: QueryOptions = {}): AsyncGenerator<TeamRunnerMessage> {
      const queue = new AsyncMessageQueue<TeamRunnerMessage>();
      const runtime = createTeamRunnerRuntime({
        mailbox,
        source,
        emit: message => queue.push(message),
        maxDelegateDepth,
        depth: 0,
        queryOptions,
      });
      const iterator = root.query(prompt, {
        ...queryOptions,
        runtime,
      });

      let rootDone = false;
      let rootNext = iterator.next().then(value => ({ kind: "root" as const, value }));
      let queueNext = queue.next().then(value => ({ kind: "queue" as const, value }));

      while (!rootDone) {
        const next = await Promise.race([queueNext, rootNext]);
        if (next.kind === "queue") {
          if (!next.value.done) {
            yield next.value.value;
          }
          queueNext = queue.next().then(value => ({ kind: "queue" as const, value }));
          continue;
        }

        if (next.value.done) {
          rootDone = true;
          queue.close();
          continue;
        }

        for (const queued of queue.drainAvailable()) {
          yield queued;
        }
        yield next.value.value;
        rootNext = iterator.next().then(value => ({ kind: "root" as const, value }));
      }

      while (true) {
        const next = await queue.next();
        if (next.done) break;
        yield next.value;
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

export function createClaudeCodeTools(options: ClaudeCodeToolsOptions = {}): Array<ToolDefinition<any>> {
  const roots = normalizeAllowedDirectories(options);
  const cwd = roots.cwd;

  return [
    tool(
      "Read",
      "Read a file from the workspace. Supports optional 1-based line offset and limit.",
      z.object({
        file_path: z.string(),
        offset: z.number().int().positive().optional(),
        limit: z.number().int().positive().optional(),
      }),
      async input => {
        const path = resolveAllowedPath(input.file_path, roots);
        const content = await readFile(path, "utf8");
        if (input.offset === undefined && input.limit === undefined) {
          return { content };
        }
        const lines = content.split(/\r?\n/);
        const start = input.offset ? input.offset - 1 : 0;
        const end = input.limit ? start + input.limit : undefined;
        return { content: lines.slice(start, end).join("\n") };
      },
    ),
    tool(
      "Write",
      "Write a file inside the workspace, creating parent directories as needed.",
      z.object({
        file_path: z.string(),
        content: z.string(),
      }),
      async input => {
        const path = resolveAllowedPath(input.file_path, roots);
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
      async input => {
        const path = resolveAllowedPath(input.file_path, roots);
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
      "List files and directories in a workspace directory.",
      z.object({
        path: z.string().optional(),
      }),
      async input => {
        const path = resolveAllowedPath(input.path ?? ".", roots);
        const entries = await readdir(path, { withFileTypes: true });
        const output = entries
          .map(entry => `${entry.name}${entry.isDirectory() ? "/" : ""}`)
          .sort()
          .join("\n");
        return { content: output };
      },
    ),
    tool(
      "Glob",
      "Find workspace files matching a glob pattern.",
      z.object({
        pattern: z.string(),
        path: z.string().optional(),
      }),
      async input => {
        const base = resolveAllowedPath(input.path ?? ".", roots);
        const matcher = globToRegExp(input.pattern);
        const files = await listFiles(base, roots);
        const output = files
          .map(file => normalizeSlash(relative(base, file)))
          .filter(file => matcher.test(file))
          .sort()
          .join("\n");
        return { content: output };
      },
    ),
    tool(
      "Grep",
      "Search workspace file contents with a regular expression.",
      z.object({
        pattern: z.string(),
        path: z.string().optional(),
        include: z.string().optional(),
      }),
      async input => {
        const base = resolveAllowedPath(input.path ?? ".", roots);
        const matcher = input.include ? globToRegExp(input.include) : null;
        const regexp = new RegExp(input.pattern);
        const files = await listFiles(base, roots);
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
    ),
    tool(
      "Bash",
      "Run a shell command in the workspace.",
      z.object({
        command: z.string(),
        timeout_ms: z.number().int().positive().optional(),
      }),
      async input => {
        const output = await runShell(input.command, {
          cwd,
          timeoutMs: input.timeout_ms ?? options.bashTimeoutMs ?? 30_000,
        });
        return { content: output };
      },
    ),
  ];
}

export async function* query(params: AgentOptions & { prompt: string; stream?: boolean; signal?: AbortSignal }): AsyncGenerator<SDKMessage> {
  const agent = createAgent(params);
  yield* agent.query(params.prompt, { stream: params.stream, signal: params.signal });
}

export class Agent {
  private readonly options: Required<Pick<AgentOptions, "maxTokens" | "maxTurns">> & AgentOptions;
  private readonly modelClient: ModelClient;
  private readonly messages: ModelMessage[] = [];
  private readonly sessionId = randomId();

  constructor(options: AgentOptions) {
    if (!options.model) {
      throw new Error("AgentOptions.model is required");
    }
    this.options = {
      maxTokens: 4096,
      maxTurns: 50,
      ...options,
    };
    this.modelClient =
      options.modelClient ??
      new AnthropicModelClient({
        apiKey: options.apiKey,
        baseURL: options.baseURL,
      });
  }

  async *query(prompt: string | ContentBlock[], options: QueryOptions = {}): AsyncGenerator<SDKMessage> {
    const startAbort = abortErrorIfNeeded(options.signal);
    yield this.initMessage();
    if (startAbort) {
      yield this.resultMessage("error_abort", "", 0, startAbort);
      return;
    }

    this.messages.push({
      role: "user",
      content: prompt,
    });

    let turns = 0;
    while (true) {
      const abortError = abortErrorIfNeeded(options.signal);
      if (abortError) {
        yield this.resultMessage("error_abort", "", turns, abortError);
        return;
      }

      if (turns >= this.options.maxTurns) {
        const error = new MaxTurnsError(`Reached maximum number of turns (${this.options.maxTurns})`);
        yield this.resultMessage("error_max_turns", "", turns, error);
        return;
      }

      turns++;
      let assistant: AssistantModelMessage;
      const streamEvents: Record<string, unknown>[] = [];
      try {
        assistant = await this.modelClient.createMessage({
          model: this.options.model,
          systemPrompt: this.options.systemPrompt,
          maxTokens: this.options.maxTokens,
          messages: this.messagesForModel(prompt),
          tools: this.modelTools(),
          stream: options.stream ?? true,
          onStreamEvent: event => {
            if (options.stream ?? true) {
              streamEvents.push(event);
            }
          },
          signal: options.signal,
        });
      } catch (error) {
        const wrapped =
          error instanceof AbortError
            ? error
            : new APIError(errorMessage(error), { cause: error });
        const subtype = wrapped instanceof AbortError ? "error_abort" : "error";
        yield this.resultMessage(subtype, "", turns, wrapped);
        return;
      }

      for (const event of streamEvents) {
        yield {
          type: "stream_event",
          event,
          session_id: this.sessionId,
        };
      }

      this.messages.push(assistant);
      yield {
        type: "assistant",
        message: assistant,
        session_id: this.sessionId,
      };

      const toolUseBlocks = assistant.content.filter(isToolUseBlock);
      if (toolUseBlocks.length === 0) {
        yield this.resultMessage("success", extractText(assistant), turns);
        return;
      }

      const toolResults: ToolResultBlock[] = [];
      let firstToolError: Error | undefined;
      for (const block of toolUseBlocks) {
        const result = await this.runTool(block, options.signal, options.runtime);
        if (result.error && !firstToolError) {
          firstToolError = result.error;
        }
        toolResults.push(result.block);
      }

      const userMessage: ModelMessage = {
        role: "user",
        content: toolResults,
      };
      this.messages.push(userMessage);
      yield {
        type: "user",
        message: userMessage,
        session_id: this.sessionId,
        tool_use_result: toolResults.length === 1 ? toolResults[0]?.content : toolResults,
        ...(firstToolError ? { error: firstToolError } : {}),
      };
    }
  }

  async prompt(prompt: string | ContentBlock[], options: QueryOptions = {}): Promise<SDKResultMessage> {
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

  addTools(tools: Array<ToolDefinition<any>>): void {
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
  ): SDKResultMessage {
    return {
      type: "result",
      subtype,
      is_error: subtype !== "success",
      result,
      session_id: this.sessionId,
      num_turns: numTurns,
      ...(error ? { error } : {}),
    };
  }

  private modelTools(): ModelToolDefinition[] {
    return (this.options.tools ?? []).map(tool => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.jsonSchema,
    }));
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
    if (typeof prompt !== "string") return [];
    const normalizedPrompt = normalizeForMatch(prompt);
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
    runtime: AgentRuntimeContext | undefined,
  ): Promise<{ block: ToolResultBlock; error?: Error }> {
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

    try {
      const parsed = definition.parse(block.input);
      const output = await definition.handler(parsed, {
        signal,
        toolUseId: block.id,
        runtime,
      });
      return {
        block: {
          type: "tool_result",
          tool_use_id: block.id,
          content: output.content,
        },
      };
    } catch (error) {
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
      };
      if (request.stream) {
        const stream = await this.client.messages.create(
          { ...body, stream: true },
          { signal: request.signal },
        );
        const assembler = new AnthropicStreamAssembler();
        for await (const event of stream as AsyncIterable<Record<string, unknown>>) {
          request.onStreamEvent?.(event);
          assembler.add(event);
        }
        return assembler.message();
      }

      const response = await this.client.messages.create(body, { signal: request.signal });
      if ("type" in response && response.type === "message") {
        return {
          role: "assistant",
          content: response.content.map(fromAnthropicBlock).filter(isContentBlock),
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
}

class AnthropicStreamAssembler {
  private content: ContentBlock[] = [];
  private currentIndex: number | undefined;
  private jsonDeltas = new Map<number, string>();

  add(event: Record<string, unknown>): void {
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
            if (block.type === "text") return block;
            if (block.type === "tool_use") return block;
            return {
              type: "tool_result" as const,
              tool_use_id: block.tool_use_id,
              content: block.content,
              ...(block.is_error ? { is_error: true } : {}),
            };
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

function isToolCapableAgentLike(agent: AgentLike): agent is ToolCapableAgentLike {
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

function createTeamRunnerRuntime(input: {
  mailbox: TeamMailbox;
  source: AgentRuntimeSource;
  emit(message: TeamRunnerMessage): void;
  maxDelegateDepth: number;
  depth: number;
  queryOptions: QueryOptions;
}): AgentRuntimeContext {
  return {
    source: input.source,
    emit: input.emit,
    async delegate(delegateInput) {
      if (input.depth >= input.maxDelegateDepth) {
        throw new Error(`Reached maximum delegate depth (${input.maxDelegateDepth})`);
      }

      const targetMailbox = delegateInput.targetMailboxId ?? sanitizeToolName(delegateInput.name);
      const callerMailbox = input.source.mailbox ?? input.source.name ?? "manager";
      const request = await input.mailbox.send(callerMailbox, targetMailbox, delegateInput.task, {
        workItemRole: "delegation",
      });
      input.emit({
        type: "team_message",
        subtype: "sent",
        source: input.source,
        mailbox: targetMailbox,
        message: request,
      });

      if (delegateInput.wait === "accepted") {
        return {
          status: "accepted",
          content: formatDelegateAccepted(request),
          request,
        };
      }

      const claimed = await input.mailbox.claimNext(targetMailbox);
      const message = claimed ?? request;
      input.emit({
        type: "team_message",
        subtype: "claimed",
        source: input.source,
        mailbox: targetMailbox,
        message,
      });

      const targetSource: AgentRuntimeSource = {
        kind: "team_member",
        name: delegateInput.name,
        member: delegateInput.name,
        mailbox: targetMailbox,
      };
      input.emit({
        type: "team_agent",
        subtype: "started",
        source: targetSource,
      });

      const childRuntime = createTeamRunnerRuntime({
        ...input,
        source: targetSource,
        depth: input.depth + 1,
      });
      let finalResult: SDKResultMessage | undefined;
      for await (const agentMessage of delegateInput.agent.query(renderDelegateTaskPrompt(delegateInput, message), {
        stream: input.queryOptions.stream,
        signal: input.queryOptions.signal,
        runtime: childRuntime,
      })) {
        if (isNestedTeamRunnerMessage(agentMessage)) {
          input.emit(agentMessage);
        } else if (agentMessage.type === "stream_event") {
          input.emit({
            type: "stream_event",
            source: targetSource,
            event: agentMessage.event,
            session_id: agentMessage.session_id,
          });
        } else {
          input.emit({
            type: "agent_message",
            source: targetSource,
            message: agentMessage,
          });
        }
        if (agentMessage.type === "result") {
          finalResult = agentMessage;
        }
      }

      input.emit({
        type: "team_agent",
        subtype: "finished",
        source: targetSource,
      });

      if (!finalResult) {
        await input.mailbox.updateStatus(message.id, "failed");
        const failed = await input.mailbox.get(message.id);
        if (failed) {
          input.emit({
            type: "team_message",
            subtype: "failed",
            source: targetSource,
            mailbox: targetMailbox,
            message: failed,
          });
        }
        throw new Error(`Delegate ${delegateInput.name} completed without a result`);
      }

      if (finalResult.is_error) {
        await input.mailbox.updateStatus(message.id, "failed");
        const failed = await input.mailbox.get(message.id);
        if (failed) {
          input.emit({
            type: "team_message",
            subtype: "failed",
            source: targetSource,
            mailbox: targetMailbox,
            message: failed,
          });
        }
        throw finalResult.error ?? new Error(finalResult.result || `Delegate ${delegateInput.name} failed`);
      }

      await input.mailbox.updateStatus(message.id, "done");
      const done = await input.mailbox.get(message.id);
      if (done) {
        input.emit({
          type: "team_message",
          subtype: "done",
          source: targetSource,
          mailbox: targetMailbox,
          message: done,
        });
      }
      const reply = await input.mailbox.send(targetMailbox, callerMailbox, finalResult.result, {
        threadId: message.threadId,
        parentMessageId: message.id,
        workItemId: message.workItemId,
        upstreamMessageId: message.upstreamMessageId ?? message.id,
        workItemRole: "upstream_report",
      });
      input.emit({
        type: "team_message",
        subtype: "replied",
        source: targetSource,
        mailbox: callerMailbox,
        message: reply,
      });

      return {
        status: "completed",
        content: finalResult.result,
        request: message,
        reply,
        result: finalResult,
      };
    },
  };
}

function formatTeamMemberDelegateDescription(member: TeamMemberDefinition): string {
  const details = [
    `Delegate work to team member ${member.name}.`,
    `Role: ${member.role}.`,
    ...(member.focus ? [`Focus: ${member.focus}.`] : []),
    "Pass a clear task. The member will return a final result.",
  ];
  return details.join(" ");
}

function formatAgentToolDescription(description: string): string {
  return [
    description,
    "",
    "This tool calls another AgentLike. Choose the action mode explicitly:",
    '- mode="ask": ask the AgentLike and wait for its final answer in this tool call.',
    '- mode="handoff": assign work and receive an acceptance receipt immediately; requires a team/runtime context.',
    '- mode="observe": request observable long-running work; currently unsupported and will return a clear error.',
    "Provide a clear task, expected output, and acceptance criteria when useful.",
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
    "",
    "--- delegated task ---",
    message.content,
  ].join("\n");
}

function formatDelegateAccepted(message: TeamMessage): string {
  return JSON.stringify({
    status: "accepted",
    message_id: message.id,
    thread_id: message.threadId,
    to: message.to,
  }, null, 2);
}

function createTeamTools(options: {
  mailbox: TeamMailbox;
  ownerMailboxId: string;
  resolveMailbox(target: string): string;
}): Array<ToolDefinition<any>> {
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
        const result = await member.agent.prompt(renderTeamTaskPrompt(member, message), {
          signal: input.options.signal,
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

function abortErrorIfNeeded(signal: AbortSignal | undefined): AbortError | undefined {
  return signal?.aborted ? new AbortError("Operation aborted") : undefined;
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

function normalizeAllowedDirectories(options: ClaudeCodeToolsOptions): AllowedRoots {
  const cwd = resolve(options.cwd ?? process.cwd());
  const directories = (options.allowedDirectories && options.allowedDirectories.length > 0
    ? options.allowedDirectories
    : [cwd]
  ).map(directory => resolve(cwd, directory));
  return { cwd, directories };
}

function resolveAllowedPath(inputPath: string, roots: AllowedRoots): string {
  const path = isAbsolute(inputPath)
    ? resolve(inputPath)
    : resolve(roots.cwd, inputPath);
  if (!roots.directories.some(root => path === root || path.startsWith(root + sep))) {
    throw new Error(`Path ${inputPath} is outside allowed directories`);
  }
  return path;
}

async function listFiles(root: string, roots: AllowedRoots): Promise<string[]> {
  const rootStat = await stat(root);
  if (!rootStat.isDirectory()) {
    return [root];
  }
  const files: string[] = [];
  const entries = await readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name === "node_modules" || entry.name === ".git") continue;
    const fullPath = resolveAllowedPath(join(root, entry.name), roots);
    if (entry.isDirectory()) {
      files.push(...(await listFiles(fullPath, roots)));
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
