// Presence — the lifelike ink-ring avatar. The states derive from the
// same turn object the NowLine reads, so avatar and status line can
// never disagree.

import React from 'react';
import { it, expect, describe, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import Presence, { presenceOf } from '../components/Presence.jsx';

afterEach(cleanup);

const base = {
  role: 'ai', content: '', timeline: [], streaming: true,
  gotFirstEvent: true, startedAt: Date.now(),
};

describe('presenceOf', () => {
  it('is idle when nothing streams', () => {
    expect(presenceOf(null)).toBe('idle');
    expect(presenceOf({ ...base, streaming: false, content: 'done' })).toBe('idle');
  });

  it('thinks before any output, works while a tool is in flight', () => {
    expect(presenceOf(base)).toBe('thinking');
    const working = {
      ...base,
      timeline: [{ kind: 'tool', tool: 'view', result: null }],
    };
    expect(presenceOf(working)).toBe('working');
    // The tool finished and text is flowing: speaking.
    const speaking = {
      ...base,
      content: 'Here is what I found…',
      timeline: [{ kind: 'tool', tool: 'view', result: 'ok' }],
    };
    expect(presenceOf(speaking)).toBe('speaking');
  });

  it('holds an amber breath while a question waits on a person', () => {
    const waiting = {
      ...base,
      question: { question_id: 'q1', status: 'open' },
      timeline: [{ kind: 'tool', tool: 'ask_user', result: null }],
    };
    expect(presenceOf(waiting)).toBe('waiting');
    const answered = {
      ...waiting,
      question: { question_id: 'q1', status: 'answered' },
      timeline: [{ kind: 'tool', tool: 'ask_user', result: 'yes' }],
    };
    expect(presenceOf(answered)).toBe('thinking');
  });
});

describe('Presence', () => {
  it('renders the ring with the state as a class', () => {
    const { container } = render(<Presence turn={base} />);
    const el = container.querySelector('.presence');
    expect(el).toHaveClass('is-thinking');
    expect(el.getAttribute('data-state')).toBe('thinking');
    expect(container.querySelector('.pr-ink')).not.toBeNull();
    expect(container.querySelector('.pr-core')).not.toBeNull();
  });

  it('re-renders into the waiting state', () => {
    const { container, rerender } = render(<Presence turn={base} />);
    rerender(<Presence turn={{ ...base, question: { status: 'open' } }} />);
    expect(container.querySelector('.presence')).toHaveClass('is-waiting');
  });
});
