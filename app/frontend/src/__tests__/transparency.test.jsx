// Transparency > Actions must show WHICH tool ran and reflect the real
// outcome (a denied action must not render as 'ok').

import React from 'react';
import { it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import { Actions, Memories } from '../components/Transparency.jsx';

afterEach(cleanup);

it('Forget passes the memory primary key (id), not source_ref/content', () => {
  const seen = [];
  const data = { memories: [
    { id: 'vm-1', kind: 'preference', content: 'Likes bullets', source_ref: 'chat:s1' },
  ] };
  render(<Memories data={data} onForget={(x) => seen.push(x)} />);
  screen.getByText('Forget').click();
  expect(seen).toEqual(['vm-1']);
});

it('shows tool name and the real outcome', () => {
  const data = {
    actions: [
      { ts: '2026-07-21 09:00', action_type: 'tool_call', tool_name: 'send_email',
        detail: 'Blocked', outcome: 'denied' },
      { ts: '2026-07-21 08:00', action_type: 'tool_call', tool_name: 'browser_navigate',
        detail: 'Opened status page', outcome: 'ok' },
    ],
  };
  const { container } = render(<Actions data={data} />);
  // Tool name is shown (not the generic action_type).
  expect(screen.getByText('send_email')).toBeInTheDocument();
  expect(screen.getByText('browser_navigate')).toBeInTheDocument();
  // The denied action carries the denied class + badge; the ok one does not.
  const denied = container.querySelector('li.tx-action.is-denied');
  expect(denied).toBeTruthy();
  expect(denied.querySelector('.tx-outcome.is-denied')).toBeTruthy();
  expect(container.querySelectorAll('li.tx-action').length).toBe(2);
  // Only the denied one gets a badge — a badge on every row would make
  // the exceptional case invisible again.
  expect(container.querySelectorAll('.tx-outcome').length).toBe(1);
});

it('renders empty state with no actions', () => {
  render(<Actions data={{ actions: [] }} />);
  expect(screen.getByText(/No actions logged yet/)).toBeInTheDocument();
});
