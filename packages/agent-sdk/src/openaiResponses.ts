import OpenAI from "openai";
import type {
  Response,
  ResponseCreateParamsNonStreaming,
  ResponseInput,
  ResponseReasoningItem,
} from "openai/resources/responses/responses";
import {
  APIError,
  type AssistantModelMessage,
  type ContentBlock,
  type ModelClient,
  type ModelMessage,
  type TextBlock,
  type ToolUseBlock,
} from "./index.js";
import { describeJsonParseError, toolInputFailures } from "./toolInput.js";

/** Replayable reasoning content, rather than response metadata. Persist with role/content. */
export type OpenAIReasoningBlock = {
  type: "openai_reasoning";
  item: ResponseReasoningItem;
};

export type OpenAIResponsesClientOptions = {
  /** Defaults to OPENAI_API_KEY, as in the official OpenAI client. */
  apiKey?: string;
  /** Defaults to https://api.openai.com/v1. Must expose POST /responses. */
  baseURL?: string;
  /** Optional transport override, for example for tests or a host proxy. */
  fetch?: typeof globalThis.fetch;
  /** Provider default when omitted. Agent/query low/high override it; max maps to xhigh. */
  reasoningEffort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh";
};

/** Creates a stateless Responses API adapter for AgentOptions.modelClient. */
export function createOpenAIResponsesClient(options: OpenAIResponsesClientOptions = {}): ModelClient {
  const client = new OpenAI({
    apiKey: options.apiKey,
    baseURL: options.baseURL ?? "https://api.openai.com/v1",
    fetch: options.fetch,
    maxRetries: 0,
  });
  const isDeepSeek = new URL(client.baseURL).hostname === "api.deepseek.com";
  return {
    async createMessage(request) {
      try {
        if (request.thinkingConfig && request.thinkingConfig.type !== "disabled") {
          throw new APIError("Responses does not support thinkingConfig budgets or adaptive thinking; use reasoningEffort");
        }
        // Submission recovery forces a tool and disables thinking. DeepSeek rejects
        // forced tools while thinking is on and defaults thinking to on, so omission
        // is insufficient there. Other endpoints may use non-reasoning models.
        const effort = request.thinkingConfig?.type === "disabled"
          ? (isDeepSeek || options.reasoningEffort === "none" ? "none" : undefined)
          : request.reasoningEffort === "max" ? "xhigh"
          : request.reasoningEffort ?? options.reasoningEffort;
        const body: ResponseCreateParamsNonStreaming = {
          model: request.model,
          input: toInput(request.messages),
          ...(request.systemPrompt !== undefined ? { instructions: request.systemPrompt } : {}),
          max_output_tokens: request.maxTokens,
          store: false,
          include: ["reasoning.encrypted_content"],
          tools: request.tools.map(tool => ({
            type: "function",
            name: tool.name,
            description: tool.description,
            parameters: tool.input_schema,
            strict: false,
          })),
          ...(request.toolChoice ? { tool_choice: { type: "function", name: request.toolChoice.name } as const } : {}),
          ...(effort ? { reasoning: { effort } } : {}),
          ...(request.outputFormat ? { text: { format: request.outputFormat === "json"
            ? { type: "json_object" as const }
            : { type: "json_schema" as const, name: "agent_output", schema: request.outputFormat.schema, strict: true } } } : {}),
          stream: false,
        };
        const requestOptions = {
          signal: request.signal,
          ...(request.timeoutMs !== undefined ? { timeout: request.timeoutMs } : {}),
        };
        if (!request.stream) return fromResponse(await client.responses.create(body, requestOptions));

        const stream = await client.responses.create({ ...body, stream: true }, requestOptions);
        let completed: Response | undefined;
        try {
          for await (const event of stream) {
            request.onStreamEvent?.(event as unknown as Record<string, unknown>);
            if (event.type === "error") throw new APIError(event.message);
            if (event.type === "response.failed") {
              throw new APIError(event.response.error?.message ?? "Responses request failed");
            }
            if (event.type === "response.completed" || event.type === "response.incomplete") {
              completed = event.response;
            }
          }
        } finally {
          // Also closes the HTTP body if a callback or stream event throws.
          stream.controller.abort();
        }
        if (!completed) throw new APIError("Responses stream ended without a terminal response");
        return fromResponse(completed);
      } catch (error) {
        if (request.signal?.aborted) throw request.signal.reason ?? error;
        if (error instanceof APIError) throw error;
        if (error instanceof OpenAI.APIError && error.code === "context_length_exceeded") {
          // Reuse the loop's existing overflow recovery classification.
          throw new APIError(`Responses context window exceeded: ${error.message}`, { cause: error });
        }
        throw new APIError(error instanceof Error ? error.message : String(error), { cause: error });
      }
    },
  };
}

function toInput(messages: ModelMessage[]): ResponseInput {
  const input: ResponseInput = [];
  for (const message of messages) {
    if (typeof message.content === "string") {
      input.push({ role: message.role, content: message.content });
      continue;
    }
    for (const block of message.content) {
      switch (block.type) {
        case "text":
          input.push({
            role: message.role,
            content: block.text,
            ...(message.role === "assistant" && block.phase !== undefined ? { phase: block.phase } : {}),
          });
          break;
        case "image":
          if (message.role !== "user") throw new APIError("Responses image blocks require the user role");
          input.push({ role: "user", content: [imageInput(block)] });
          break;
        case "tool_use":
          if (message.role !== "assistant") throw new APIError("Responses tool_use blocks require the assistant role");
          input.push({ type: "function_call", call_id: block.id, name: block.name, arguments: JSON.stringify(block.input) });
          break;
        case "tool_result":
          if (message.role !== "user") throw new APIError("Responses tool_result blocks require the user role");
          input.push({
            type: "function_call_output", call_id: block.tool_use_id,
            output: typeof block.content === "string" ? block.content : block.content.map(part => {
              if (part.type === "text") return { type: "input_text" as const, text: part.text };
              if (part.type === "image") return imageInput(part);
              throw new APIError(`Unsupported Responses tool result block: ${part.type}`);
            }),
          });
          break;
        case "openai_reasoning":
          if (message.role !== "assistant") throw new APIError("Responses reasoning blocks require the assistant role");
          input.push(block.item);
          break;
        default:
          throw new APIError(`Unsupported Responses content block: ${block.type}`);
      }
    }
  }
  return input;
}

function imageInput(block: Extract<ContentBlock, { type: "image" }>) {
  const source = block.source;
  if (source.type !== "url" && source.type !== "base64") throw new APIError("Unsupported Responses image source");
  return {
    type: "input_image" as const,
    image_url: source.type === "url" ? source.url : `data:${source.media_type};base64,${source.data}`,
    detail: "auto" as const,
  };
}

function fromResponse(response: Response): AssistantModelMessage {
  if (response.error || response.status === "failed" || response.status === "cancelled") {
    throw new APIError(response.error?.message ?? `Responses request ${response.status}`);
  }
  if (response.status !== "completed" && response.status !== "incomplete") {
    throw new APIError(`Unexpected Responses status: ${response.status}`);
  }
  const content: ContentBlock[] = [];
  let refused = false;
  for (const item of response.output) {
    switch (item.type) {
      case "reasoning":
        content.push({ type: "openai_reasoning", item });
        break;
      case "message":
        for (const part of item.content) {
          if (part.type === "refusal") refused = true;
          const block: TextBlock = {
            type: "text", text: part.type === "refusal" ? part.refusal : part.text,
            ...(item.phase !== undefined ? { phase: item.phase } : {}),
          };
          content.push(block);
        }
        break;
      case "function_call": {
        const block: ToolUseBlock = { type: "tool_use", id: item.call_id, name: item.name, input: {} };
        try {
          const parsed: unknown = JSON.parse(item.arguments);
          if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            toolInputFailures.set(block, { kind: "object" });
          } else block.input = parsed as Record<string, unknown>;
        } catch (error) {
          toolInputFailures.set(block, { kind: "json", ...describeJsonParseError(error) });
        }
        content.push(block);
        break;
      }
      default:
        throw new APIError(`Unsupported Responses output item: ${item.type}`);
    }
  }
  const cached = response.usage?.input_tokens_details?.cached_tokens ?? 0;
  const incomplete = response.incomplete_details?.reason;
  return {
    role: "assistant", content,
    providerResponseId: response.id,
    model: response.model,
    stopReason: incomplete === "max_output_tokens" ? "max_tokens"
      : incomplete === "content_filter" || refused ? "refusal"
      : response.status === "incomplete" ? "incomplete"
      : content.some(block => block.type === "tool_use") ? "tool_use" : "end_turn",
    ...(response.usage ? { usage: {
      // Match the SDK's mutually exclusive Anthropic usage buckets.
      input_tokens: response.usage.input_tokens - cached,
      output_tokens: response.usage.output_tokens,
      ...(cached ? { cache_read_input_tokens: cached } : {}),
    } } : {}),
  };
}
