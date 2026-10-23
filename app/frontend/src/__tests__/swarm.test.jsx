// The swarm lanes: named, ordered, honest about failure.
//
// The reducer is the correctness core — events from ten concurrent
// workers interleave arbitrarily on one stream, and the lanes must not
// jump, merge, or lose a failure.

import React from 'react';
import { it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import { reduceTurn } from '../components/Chat.jsx';
import SwarmPanel from '../components/Swarm.jsx';

afterEach(cleanup);

const turn = () => ({ content: '', gotFirstEvent: false });
const feed = (t, events) => events.reduce(reduceTurn, t);

it('lanes appear in first-seen order and do not jump as events interleave', () => {
  const t = feed(turn(), [
    { type: 'subagent', source: 'pricing', phase: 'spawned', text: 'find pricing' },
    { type: 'subagent', source: 'contracts', phase: 'spawned', text: 'find contracts' },
    { type: 'subagent', source: 'pricing', phase: 'tool', body: { tool: 'search' } },
    { type: 'subagent', source: 'contracts', phase: 'done', text: 'all found' },
  ]);

  expect(t.swarm.order).toEqual(['pricing', 'contracts']);
  expect(t.swarm.agents.pricing.status).toBe('running');
  expect(t.swarm.agents.contracts.status).toBe('done');
});

it('a worker’s streamed text accumulates in its own lane only', () => {
  const t = feed(turn(), [
    { type: 'subagent', source: 'a', phase: 'spawned', text: 'x' },
    { type: 'subagent', source: 'b', phase: 'spawned', text: 'y' },
    { type: 'subagent', source: 'a', phase: 'text', body: { type: 'text', text: 'one ' } },
    { type: 'subagent', source: 'b', phase: 'text', body: { type: 'text', text: 'other' } },
    { type: 'subagent', source: 'a', phase: 'text', body: { type: 'text', text: 'two' } },
  ]);

  expect(t.swarm.agents.a.stream).toBe('one two');
  expect(t.swarm.agents.b.stream).toBe('other');
  // And the parent turn's own content is untouched by worker text.
  expect(t.content).toBe('');
});

it('a failure is recorded with its reason', () => {
  const t = feed(turn(), [
    { type: 'subagent', source: 'boom', phase: 'spawned', text: 'x' },
    { type: 'subagent', source: 'boom', phase: 'failed', text: 'model exploded' },
  ]);

  expect(t.swarm.agents.boom.status).toBe('failed');
  expect(t.swarm.agents.boom.error).toBe('model exploded');
});

it('lane events are capped so a runaway worker cannot grow state unbounded', () => {
  const events = [{ type: 'subagent', source: 'w', phase: 'spawned', text: 'x' }];
  for (let i = 0; i < 300; i++) {
    events.push({ type: 'subagent', source: 'w', phase: 'tool', body: { tool: `t${i}` } });
  }
  const t = feed(turn(), events);

  expect(t.swarm.agents.w.events.length).toBeLessThanOrEqual(200);
});

it('renders a lane per worker with name, status, and tools', () => {
  const t = feed(turn(), [
    { type: 'subagent', source: 'pricing', phase: 'spawned', text: 'find pricing' },
    { type: 'subagent', source: 'pricing', phase: 'reasoning',
      body: { type: 'reasoning', text: 'checking the rate card' } },
    { type: 'subagent', source: 'pricing', phase: 'tool', body: { tool: 'search_enterprise' } },
    { type: 'subagent', source: 'pricing', phase: 'done', text: 'rates found' },
  ]);
  render(<SwarmPanel swarm={t.swarm} />);

  expect(screen.getByText('pricing')).toBeInTheDocument();
  expect(screen.getByText('done')).toBeInTheDocument();
  // Tool chips speak plain language; the exact tool name survives on
  // the tooltip for anyone who needs the engineering identifier.
  expect(screen.getByTitle('search_enterprise')).toBeInTheDocument();
  expect(screen.getByText('Search enterprise')).toBeInTheDocument();
  // The worker's thinking is present (progressive disclosure handles
  // visibility; presence is what we can assert).
  expect(screen.getByText('checking the rate card')).toBeInTheDocument();
});

it('a failed lane is open by default — failures are not behind a click', () => {
  const t = feed(turn(), [
    { type: 'subagent', source: 'boom', phase: 'spawned', text: 'x' },
    { type: 'subagent', source: 'boom', phase: 'failed', text: 'timed out' },
  ]);
  const { container } = render(<SwarmPanel swarm={t.swarm} />);

  const lane = container.querySelector('.sw-lane.is-failed');
  expect(lane).toBeTruthy();
  expect(lane.hasAttribute('open')).toBe(true);
  expect(screen.getByText('timed out')).toBeInTheDocument();
});

it('a constitutional block inside a worker lane is shown as such', () => {
  const t = feed(turn(), [
    { type: 'subagent', source: 'w', phase: 'spawned', text: 'x' },
    { type: 'subagent', source: 'w', phase: 'law_block',
      body: { type: 'law_block', article: 'A9', detail: 'no shell' } },
  ]);
  render(<SwarmPanel swarm={t.swarm} />);

  expect(screen.getByText(/Blocked by the constitution/)).toBeInTheDocument();
});
