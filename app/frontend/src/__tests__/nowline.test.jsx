// NowLine + verbs: the live status line always answers "what is it
// doing?" in plain language, with the elapsed clock and the run's
// counters — and a governor halt lands on the timeline as a visible
// step rather than an unexplained ending.

import React from 'react';
import { it, expect, afterEach, describe } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import NowLine from '../components/NowLine.jsx';
import { nowVerb, verbFor } from '../lib/verbs.js';
import { reduceTurn } from '../components/Chat.jsx';

afterEach(cleanup);

const base = () => ({
  role: 'ai', content: '', timeline: [], activity: [], reasoning: [],
  streaming: true, startedAt: Date.now() - 24000, gotFirstEvent: true,
});

describe('verbs', () => {
  it('speaks plain language for known tools', () => {
    expect(verbFor('search_knowledge')).toBe('Searching enterprise knowledge');
    expect(verbFor('create_file')).toBe('Writing a file');
  });

  it('names the server for MCP tools and humanises unknowns', () => {
    expect(verbFor('mcp__github__create_issue')).toBe('Using github: create issue');
    expect(verbFor('some_new_tool')).toBe('Some new tool');
  });

  it('the now-verb tracks the in-flight tool, then the swarm, then writing', () => {
    const t = base();
    t.timeline = [{ kind: 'tool', tool: 'read_inbox', result: null }];
    expect(nowVerb(t)).toBe('Reading the inbox…');

    t.timeline = [{ kind: 'tool', tool: 'read_inbox', result: { ok: true } }];
    t.swarm = { order: ['a', 'b'], agents: {
      a: { status: 'running' }, b: { status: 'done' },
    } };
    expect(nowVerb(t)).toBe('1 sub-agent working…');

    t.swarm = null;
    t.content = 'partial answer';
    expect(nowVerb(t)).toBe('Writing…');
  });
});

describe('NowLine', () => {
  it('shows the verb, the elapsed clock and the tool count', () => {
    const t = base();
    t.timeline = [
      { kind: 'tool', tool: 'search_knowledge', result: { ok: true } },
      { kind: 'tool', tool: 'create_file', result: null },
    ];
    render(<NowLine turn={t} />);
    const line = screen.getByTestId('now-line');
    expect(line).toHaveTextContent('Writing a file…');
    expect(line).toHaveTextContent('24s');
    expect(line).toHaveTextContent('2 tools');
  });

  it('surfaces the governor budget when the governor has spoken', () => {
    const t = base();
    t.governor = {
      action: 'warn', pathology: 'repetition', reason: 'retrying identical call',
      state: { llm_turns: 2, tool_calls: 5, budget: { max_llm_turns: 12 } },
    };
    render(<NowLine turn={t} />);
    const line = screen.getByTestId('now-line');
    expect(line).toHaveTextContent('pass 3/12');
    expect(line).toHaveTextContent('repetition');
  });

  it('renders nothing once the turn settles', () => {
    const t = { ...base(), streaming: false };
    const { container } = render(<NowLine turn={t} />);
    expect(container.firstChild).toBeNull();
  });
});

describe('governor events in the reducer', () => {
  it('a halt becomes a visible timeline step', () => {
    const t = base();
    const out = reduceTurn(t, {
      type: 'governor', action: 'halt', pathology: 'stagnation',
      reason: 'the last 6 tool calls produced nothing',
      state: { tool_calls: 6 },
    });
    expect(out.governor.pathology).toBe('stagnation');
    const step = out.timeline[out.timeline.length - 1];
    expect(step.kind).toBe('law_block');
    expect(step.detail).toMatch(/produced nothing/);
  });

  it('a warning updates the meter without adding a step', () => {
    const t = base();
    const out = reduceTurn(t, {
      type: 'governor', action: 'warn', pathology: 'scope_drift',
      reason: 'off-goal capability', state: { tool_calls: 3 },
    });
    expect(out.governor.pathology).toBe('scope_drift');
    expect(out.timeline).toHaveLength(0);
  });
});
