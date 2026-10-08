import { test, expect, mock } from 'claude-code/testing'

// The test's `on` sits beneath the plugin: its `tool.check` hook stands for
// core's verdict, its `http.fetch` hook for the host's endpoint.

const reply = (v: unknown, status = 200) => ({
  value: {
    status,
    ok: status < 400,
    headers: { "content-type": "application/json" },
    text: JSON.stringify(v),
  },
})

test('a core deny stands without asking the endpoint', async ($, on) => {
  mock.env(on, { YAH_GATE_SOCK: '/tmp/gate.sock' })
  let fetched = 0
  on('tool.check', () => ({ decision: 'deny', reason: 'core said no' }))
  on('http.fetch', () => (fetched++, reply({ decision: 'allow' })))
  const v = await $.tool.check({ tool: 'Read', input: { file_path: '/x' } })
  expect(v.decision).toBe('deny')
  expect(v.reason).toBe('core said no')
  expect(fetched).toBe(0)
})

test('a core allow is forwarded as core_decision=allow and the host can escalate it', async ($, on) => {
  mock.env(on, { YAH_GATE_SOCK: '/tmp/gate.sock' })
  const bodies: any[] = []
  on('tool.check', () => ({ decision: 'allow' }))
  on('http.fetch', (_$, e) => {
    bodies.push(JSON.parse(String(e.init?.body)))
    return reply({ decision: 'deny', reason: 'camp boundary: declined' })
  })
  const v = await $.tool.check({ tool: 'Read', input: { file_path: '/etc/hosts' } })
  expect(v.decision).toBe('deny')
  expect(v.reason).toBe('camp boundary: declined')
  expect(bodies.length).toBe(1)
  expect(bodies[0].core_decision).toBe('allow')
  expect(bodies[0].tool).toBe('Read')
})

test('a core allow the host also allows stays an allow', async ($, on) => {
  mock.env(on, { YAH_GATE_SOCK: '/tmp/gate.sock' })
  on('tool.check', () => ({ decision: 'allow' }))
  on('http.fetch', () => reply({ decision: 'allow' }))
  const v = await $.tool.check({ tool: 'Read', input: { file_path: '/x' } })
  expect(v.decision).toBe('allow')
})

test('a core ask is put to the endpoint over the socket', async ($, on) => {
  mock.env(on, { YAH_GATE_SOCK: '/tmp/gate.sock' })
  const seen: { url: string; socketPath?: string; body: any }[] = []
  on('tool.check', () => ({ decision: 'ask' }))
  on('http.fetch', (_$, e) => {
    seen.push({ url: e.url, socketPath: e.init?.socketPath, body: JSON.parse(String(e.init?.body)) })
    return reply({ decision: 'allow', reason: 'approved' })
  })
  const v = await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })
  expect(v.decision).toBe('allow')
  expect(seen.length).toBe(1)
  expect(seen[0].socketPath).toBe('/tmp/gate.sock')
  expect(seen[0].body.tool).toBe('Bash')
  expect(seen[0].body.input).toEqual({ command: 'ls' })
  expect(seen[0].body.core_decision).toBe('ask')
})

test('an endpoint deny reaches the model with its reason', async ($, on) => {
  mock.env(on, { YAH_GATE_SOCK: '/tmp/gate.sock' })
  on('tool.check', () => ({ decision: 'ask' }))
  on('http.fetch', () => reply({ decision: 'deny', reason: 'user declined' }))
  const v = await $.tool.check({ tool: 'Bash', input: { command: 'rm -rf x' } })
  expect(v.decision).toBe('deny')
  expect(v.reason).toBe('user declined')
})

test('pending is re-asked with the same tool_use_id until decided', async ($, on) => {
  mock.env(on, { YAH_GATE_SOCK: '/tmp/gate.sock' })
  const ids: string[] = []
  on('tool.check', () => ({ decision: 'ask' }))
  on('http.fetch', (_$, e) => {
    ids.push(JSON.parse(String(e.init?.body)).tool_use_id)
    return reply(ids.length < 3 ? { decision: 'pending' } : { decision: 'allow' })
  })
  const v = await $.tool.check({ tool: 'Write', input: { file_path: '/x', content: '' } })
  expect(v.decision).toBe('allow')
  expect(ids.length).toBe(3)
  expect(new Set(ids).size).toBe(1)
})

test('no socket fails closed', async ($, on) => {
  mock.env(on, {})
  on('tool.check', () => ({ decision: 'ask' }))
  const v = await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })
  expect(v.decision).toBe('deny')
})

test('an endpoint error fails closed', async ($, on) => {
  mock.env(on, { YAH_GATE_SOCK: '/tmp/gate.sock' })
  on('tool.check', () => ({ decision: 'ask' }))
  on('http.fetch', () => reply({ error: 'boom' }, 500))
  const v = await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })
  expect(v.decision).toBe('deny')
})

test('a throwing endpoint fails closed through the catch', async ($, on) => {
  mock.env(on, { YAH_GATE_SOCK: '/tmp/gate.sock' })
  on('tool.check', () => ({ decision: 'ask' }))
  on('http.fetch', () => {
    throw new Error('ECONNREFUSED')
  })
  const v = await $.tool.check({ tool: 'Bash', input: { command: 'ls' } })
  expect(v.decision).toBe('deny')
})
