import { describe, expect, test } from 'bun:test';
import { z } from 'zod/v4';
import { createBareAgent, tool } from '../index.js';
function stream(parts: string[], name = 'submit_output') {
  const events = [
    { type: 'message_start', message: { id: 'test', type: 'message', role: 'assistant', model: 'test', content: [], usage: { input_tokens: 1, output_tokens: 0 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'call-test', name, input: {} } },
    ...parts.map(partial_json => ({ type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json } })),
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 10 } },
    { type: 'message_stop' },
  ];
  return new Response(events.map(e => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join(''), { headers: { 'content-type': 'text/event-stream' } });
}
describe('tool JSON feedback', () => {
  test('stream - malformed JSON then correction - returns actionable error and accepts corrected output', async () => {
    let requests = 0; let feedback: any;
    const server = Bun.serve({ port: 0, async fetch(request) {
      const body = await request.json() as any;
      if (++requests === 1) return stream(['{"result: {"reason": "private-content"}}']);
      feedback = body.messages.at(-1).content[0];
      return stream(['{"result":', ' {"reason":"fixed"}}']);
    } });
    try {
      const agent = createBareAgent({ apiKey: 'test', baseURL: server.url.origin, model: 'test', outputSchema: z.object({ result: z.object({ reason: z.string() }) }), maxTurns: 3 });
      const result = await agent.prompt('go');
      expect(feedback.is_error).toBe(true); expect(feedback.tool_use_id).toBe('call-test');
      expect(feedback.content).toContain('not valid JSON'); expect(feedback.content).toContain('Required top-level fields: "result"');
      expect(feedback.content).toContain('double-quoted'); expect(feedback.content).not.toContain('private-content');
      expect(result.is_error).toBe(false); expect(result.structuredResult).toEqual({ result: { reason: 'fixed' } }); expect(requests).toBe(2);
    } finally { server.stop(true); }
  });
  test.each(['{"x":', '[]', 'null', '"text"'])('stream - invalid arguments %s - never executes a permissive handler', async raw => {
    let calls = 0; const server = Bun.serve({ port: 0, fetch: () => stream([raw], 'write') });
    try {
      const agent = createBareAgent({ apiKey: 'test', baseURL: server.url.origin, model: 'test', maxTurns: 1, tools: [tool('write', 'test', z.any(), () => { calls++; return { content: 'bad' }; })] });
      const messages = []; for await (const message of agent.query('go')) messages.push(message);
      expect(calls).toBe(0); const response = messages.find((m: any) => m.type === 'user') as any;
      expect(response.message.content[0].is_error).toBe(true); expect(response.message.content[0].content).toContain(raw.startsWith('{') ? 'not valid JSON' : 'must be a JSON object');
    } finally { server.stop(true); }
  });
  test('stream - incomplete chunks become valid - executes normally', async () => {
    let calls = 0; const server = Bun.serve({ port: 0, fetch: () => stream(['{"value":', '"ok"', '}'], 'write') });
    try {
      const agent = createBareAgent({ apiKey: 'test', baseURL: server.url.origin, model: 'test', maxTurns: 1, tools: [tool('write', 'test', z.object({ value: z.string() }), input => { calls++; expect(input.value).toBe('ok'); return { content: 'ok', endTurn: true }; })] });
      expect((await agent.prompt('go')).is_error).toBe(false); expect(calls).toBe(1);
    } finally { server.stop(true); }
  });
  test('stream - valid empty object for no-argument tool - remains valid', async () => {
    const server = Bun.serve({ port: 0, fetch: () => stream(['{}'], 'read') });
    try {
      const agent = createBareAgent({ apiKey: 'test', baseURL: server.url.origin, model: 'test', maxTurns: 1, tools: [tool('read', 'test', z.object({}), () => ({ content: 'ok', endTurn: true }))] });
      expect((await agent.prompt('go')).is_error).toBe(false);
    } finally { server.stop(true); }
  });
});
