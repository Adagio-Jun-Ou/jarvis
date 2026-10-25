// Elicitation on the board: the card badge, the "Alex is waiting on
// you" panel with one-click options, and the NowLine parked state.

import React from 'react';
import { it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import Board from '../components/Board.jsx';
import NowLine from '../components/NowLine.jsx';
import { reduceTurn } from '../components/Chat.jsx';
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

const QUESTION = {
  question_id: 'q-1', task_id: 't-1', status: 'open',
  question: 'Which quarter should the report cover?',
  options: ['Q2', 'Q3'], answer: '', answered_by: '',
};

it('badges a card whose run is waiting on a human answer', async () => {
  stubFetch([
    ['/api/board/proposals', { proposals: [] }],
    ['/api/board/tasks', {
      ...emptyBoard,
      tasks: [{
        task_id: 't-1', title: 'Quarterly report', status: 'in_progress',
        priority: 2, crew: [], blocked_by: [], waiting_on: 0,
        open_questions: 1, run: { status: 'running' },
      }],
    }],
  ]);

  render(<Board session={session} />);
  await screen.findByText('Quarterly report');
  expect(screen.getByText(/needs your answer/)).toBeInTheDocument();
});

it('shows the question panel and an option click posts the answer', async () => {
  const calls = stubFetch([
    ['/questions/q-1/answer', { ...QUESTION, status: 'answered', answer: 'Q3' }],
    ['/api/board/tasks/t-1/messages', { messages: [], questions: [QUESTION] }],
    ['/api/board/tasks/t-1', {
      task_id: 't-1', title: 'Quarterly report', status: 'in_progress',
      priority: 2, crew: [], messages: [], plan: [], acceptance: [],
      blocked_by: [], past_work: [], run: null, questions: [QUESTION],
    }],
    ['/api/board/tasks', {
      ...emptyBoard,
      tasks: [{
        task_id: 't-1', title: 'Quarterly report', status: 'in_progress',
        priority: 2, crew: [], blocked_by: [], waiting_on: 0, open_questions: 1,
      }],
    }],
  ]);

  render(<Board session={session} />);
  fireEvent.click(await screen.findByText('Quarterly report'));

  await screen.findByText('Alex is waiting on you');
  expect(screen.getByText(/Which quarter should the report cover/)).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Q3' }));
  await waitFor(() => {
    const c = calls.find((x) => x.url.includes('/questions/q-1/answer'));
    expect(c).toBeTruthy();
    expect(c.opts.method).toBe('POST');
    expect(JSON.parse(c.opts.body).answer).toBe('Q3');
    expect(c.opts.headers.Authorization).toBe('Bearer tok');
  });
});

it('NowLine goes amber-and-parked while a question is open', () => {
  let turn = {
    role: 'ai', content: '', timeline: [], activity: [], reasoning: [],
    streaming: true, gotFirstEvent: true, startedAt: Date.now(),
  };
  turn = reduceTurn(turn, { type: 'question', body: QUESTION });
  render(<NowLine turn={turn} />);
  expect(screen.getByText(/Waiting for an answer from the crew/)).toBeInTheDocument();
  expect(document.querySelector('.now-line.waiting')).not.toBeNull();

  // Answered: the parked state clears and the timeline records both acts.
  cleanup();
  turn = reduceTurn(turn, {
    type: 'question',
    body: { ...QUESTION, status: 'answered', answer: 'Q3', answered_by: 'dana@alex.dev' },
  });
  render(<NowLine turn={turn} />);
  expect(screen.queryByText(/Waiting for an answer/)).toBeNull();
  expect(turn.timeline.filter((i) => i.kind === 'question')).toHaveLength(2);
});
