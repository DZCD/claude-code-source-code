---
name: claude-sdk-real-test-harness
description: Use when analyzing, debugging, or validating claude-team-agent-sdk behavior against the external real integration test project, especially live DeepSeek runs, multi-agent mailbox/team runtime logs, stream:false traces, tool-use/tool-result behavior, maxTurns regressions, or release checks before npm publish.
---

# Claude SDK Real Test Harness

This project has a separate real-behavior test harness at:

```text
/Users/duzicong/code/duzicong/claude-code-sdk-test
```

Treat that project as the live integration surface for `claude-team-agent-sdk`. It is used to verify SDK behavior with real providers, currently including DeepSeek Anthropic-compatible calls such as:

```ts
createAgent({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/anthropic",
  model: "deepseek-v4-flash",
});
```

Never store or repeat API keys in this skill, docs, tests, or commits.

## When To Use

Use this harness when SDK work touches:

- live model streaming or non-streaming behavior
- tool calls and tool results
- mailbox/team routing
- nested `createTeam()` behavior
- `teamMember()` and AgentLike composition
- `team.query()` / `team.prompt()` defaults
- `maxTurns`, abort, or long-running team tests
- npm release validation with real provider behavior

## Important Artifacts

Primary log path:

```text
/Users/duzicong/code/duzicong/claude-code-sdk-test/logs/multi-agent-stream.jsonl
```

Backup logs usually follow:

```text
/Users/duzicong/code/duzicong/claude-code-sdk-test/logs/multi-agent-stream.before-*.jsonl
```

Prefer `rg` over opening the whole log:

```bash
rg -n '"type":"result"|"subtype":"error|team_reply failed|Tool .* failed|team_message|team_agent|tool_use' logs/multi-agent-stream.jsonl
```

## Reading Results

For team tests, do not trust Vitest pass/fail alone. Always inspect or assert the final SDK result:

```ts
expect(finalResult).toBeDefined();
expect(finalResult.is_error).toBe(false);
expect(finalResult.subtype).toBe("success");
```

Use `{ stream: false }` when the test should record workflow context without token-level noise:

```ts
for await (const message of team.query(prompt, { stream: false })) {
  if (message.type === "stream_event") continue;
  await saveStreamMessage(logFile, message);
}
```

This should preserve useful events such as system init, assistant tool use, tool results, `team_message`, `team_agent`, and final `result`, while avoiding per-token stream deltas.

## Known Interpretation Notes

- `num_turns` is counted per agent loop/model request, not wall-clock seconds or mailbox message count.
- A tool failure inside the log can be recoverable when the SDK converts it into an error `tool_result` and the model continues to a final success.
- `Tool team_reply failed: Team message first does not belong to manager` means an agent attempted to reply to a mailbox message through a manager-scoped `team_reply` tool whose ownership check did not match the message recipient. Treat it as a routing/tool-scope signal, not automatically as a whole-run failure.
- If a run ends with `subtype: "error_max_turns"` or `is_error: true`, the test may still pass unless the test explicitly asserts the final result. Strengthen the test before claiming the SDK behavior is good.

## Release Check Habit

Before publishing a new SDK version, prefer this sequence:

1. Build and unit-test `packages/agent-sdk`.
2. Pack or publish the candidate version.
3. Update the harness dependency to that exact version.
4. Run the live multi-agent/team test with `{ stream: false }`.
5. Inspect the final result, `num_turns`, recoverable tool failures, and mailbox event flow.

