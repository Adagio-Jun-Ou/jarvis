// Board: the task-core surfaces — email proposals (the human gate on
// origination), dependency badges (enforcement made visible), and the
// acceptance checklist (the crew's bar, with the agent's verdicts).

import React from 'react';
import { it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import Board from '../components/Board.jsx';
import { clearCache } from '../lib/swr.js';

afterEach(() => { cleanup(); vi.restoreAllMocks(); clearCache(); });

const session = { email: 'dana@alex.dev', authHeaders: () => ({ Authorization: 'Bearer tok' }) };

const emptyBoard = { columns: ['backlog', 'todo', 'in_progress', 'review', 'done'], tasks: [] };

function stubFetch(routes) {
  const calls = [];
  const fetchFn = vi.fn((url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    for (const [match, payload] of routes) {
      if (u.includes(match)) {
        return Promise.resolve({ ok: true, json: async () => (typeof payload === 'function' ? payload() : payload) });
      }
    }
    return Promise.resolve({ ok: true, json: async () => ({}) });
  });
  vi.stubGlobal('fetch', fetchFn);
  return calls;
}

it('shows email proposals and accepting one posts the decision', async () => {
  const calls = stubFetch([
    ['/api/board/proposals', {
      proposals: [{
        proposal_id: 'prop-1',
        title: 'send the Q3 variance analysis',
        description: 'From an email (em-1) — cfo@alex.dev asked',
      }],
    }],
    ['/api/board/tasks', emptyBoard],
  ]);

  render(<Board session={session} />);

  await screen.findByText(/send the Q3 variance analysis/);
  expect(screen.getByText(/Found in your email/)).toBeInTheDocument();

  fireEvent.click(screen.getByText('Add to board'));
  await waitFor(() => {
    const c = calls.find((x) => x.url.includes('/api/board/proposals/prop-1'));
    expect(c).toBeTruthy();
    expect(c.opts.method).toBe('POST');
    expect(JSON.parse(c.opts.body).action).toBe('accept');
  });
});

it('badges a blocked, email-originated card', async () => {
  stubFetch([
    ['/api/board/proposals', { proposals: [] }],
    ['/api/board/tasks', {
      ...emptyBoard,
      tasks: [{
        task_id: 't-1', title: 'Reconcile the ledger', status: 'todo',
        priority: 2, crew: [], source: 'email',
        blocked_by: [{ task_id: 't-0', title: 'Gather statements', status: 'in_progress', open: true }],
        waiting_on: 1,
      }],
    }],
  ]);

  render(<Board session={session} />);

  await screen.findByText('Reconcile the ledger');
  expect(screen.getByText(/waiting on 1/)).toBeInTheDocument();
  expect(screen.getByText(/email/)).toBeInTheDocument();
});

it('flags an overdue task loudly and leaves done tasks quiet', async () => {
  stubFetch([
    ['/api/board/proposals', { proposals: [] }],
    ['/api/board/tasks', {
      ...emptyBoard,
      tasks: [
        { task_id: 't-1', title: 'File the report', status: 'todo', priority: 2,
          crew: [], blocked_by: [], waiting_on: 0, due_at: '2020-01-05' },
        { task_id: 't-2', title: 'Old finished thing', status: 'done', priority: 3,
          crew: [], blocked_by: [], waiting_on: 0, due_at: '2020-01-05' },
      ],
    }],
  ]);

  render(<Board session={session} />);

  await screen.findByText('File the report');
  expect(screen.getByText(/overdue/)).toBeInTheDocument();
  // The done task's stale date is history, not pressure.
  expect(document.querySelectorAll('.bd-due').length).toBe(1);
});

it('renders the acceptance checklist with met and unmet verdicts', async () => {
  stubFetch([
    ['/api/board/proposals', { proposals: [] }],
    ['/api/board/tasks/t-1', {
      task_id: 't-1', title: 'Close the quarter', status: 'in_progress',
      priority: 2, crew: [], messages: [], plan: [],
      acceptance: [
        { crit_no: 1, text: 'report is sent', status: 'met', note: 'sent 09:12' },
        { crit_no: 2, text: 'totals reconcile', status: 'unmet', note: 'SAP differs by 402.10' },
      ],
      blocked_by: [], past_work: [], run: null,
    }],
    ['/api/board/tasks', {
      ...emptyBoard,
      tasks: [{ task_id: 't-1', title: 'Close the quarter', status: 'in_progress', priority: 2, crew: [], blocked_by: [], waiting_on: 0 }],
    }],
  ]);

  render(<Board session={session} />);
  fireEvent.click(await screen.findByText('Close the quarter'));

  await screen.findByText(/1\/2 met/);
  expect(screen.getByText('report is sent')).toBeInTheDocument();
  expect(screen.getByText(/SAP differs by 402.10/)).toBeInTheDocument();
  const unmet = document.querySelector('.bd-crit.is-unmet');
  expect(unmet).not.toBeNull();
});

it('lets the crew define done and saves it via PATCH', async () => {
  const calls = stubFetch([
    ['/api/board/proposals', { proposals: [] }],
    ['/api/board/tasks/t-1', {
      task_id: 't-1', title: 'Close the quarter', status: 'todo',
      priority: 3, crew: [], messages: [], plan: [], acceptance: [],
      blocked_by: [], past_work: [], run: null,
    }],
    ['/api/board/tasks', {
      ...emptyBoard,
      tasks: [{ task_id: 't-1', title: 'Close the quarter', status: 'todo', priority: 3, crew: [], blocked_by: [], waiting_on: 0 }],
    }],
  ]);

  render(<Board session={session} />);
  fireEvent.click(await screen.findByText('Close the quarter'));

  fireEvent.click(await screen.findByText(/Define what .done. means/));
  const box = screen.getByLabelText(/Acceptance criteria/);
  fireEvent.change(box, { target: { value: 'report sent\ntotals reconcile\n' } });
  fireEvent.click(screen.getByText('Save'));

  await waitFor(() => {
    const c = calls.find(
      (x) => x.url.endsWith('/api/board/tasks/t-1') && x.opts.method === 'PATCH'
    );
    expect(c).toBeTruthy();
    expect(JSON.parse(c.opts.body).acceptance).toEqual(['report sent', 'totals reconcile']);
  });
});
