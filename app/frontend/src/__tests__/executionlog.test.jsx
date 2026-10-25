// Tests for components/ExecutionLog.jsx — the visible record of agent work.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import React from 'react';
import ExecutionLog, { formatDuration } from '../components/ExecutionLog.jsx';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const tool = (over = {}) => ({ kind: 'tool', tool: 'search_knowledge', ...over });

describe('formatDuration', () => {
  it('scales the unit to the magnitude', () => {
    expect(formatDuration(420)).toBe('420ms');
    expect(formatDuration(4200)).toBe('4.2s');
    expect(formatDuration(125000)).toBe('2m 5s');
  });

  it('returns nothing for missing or nonsense input', () => {
    expect(formatDuration(null)).toBe('');
    expect(formatDuration(-5)).toBe('');
    expect(formatDuration(Infinity)).toBe('');
  });
});

describe('ExecutionLog', () => {
  it('renders nothing for an empty timeline', () => {
    const { container } = render(<ExecutionLog timeline={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('numbers every step in order', () => {
    render(
      <ExecutionLog
        timeline={[tool({ tool: 'a', result: 'x' }), tool({ tool: 'b', result: 'y' })]}
      />
    );
    const steps = screen.getAllByText(/^[12]$/).map((n) => n.textContent);
    expect(steps).toEqual(['1', '2']);
  });

  it('marks a step still running versus finished', () => {
    const { container } = render(
      <ExecutionLog timeline={[tool({ result: null }), tool({ result: 'done' })]} />
    );
    const rows = container.querySelectorAll('.exec-row');
    expect(rows[0].getAttribute('data-status')).toBe('running');
    expect(rows[1].getAttribute('data-status')).toBe('ok');
  });

  it('summarises step count and state in the header', () => {
    render(
      <ExecutionLog timeline={[tool({ result: 'x' }), tool({ result: 'y' })]} />
    );
    expect(screen.getByText(/2 steps/)).toBeInTheDocument();
    expect(screen.getByText(/done/)).toBeInTheDocument();
  });

  it('reports that work is still running', () => {
    render(<ExecutionLog timeline={[tool({ result: null })]} streaming />);
    expect(screen.getByText(/running/)).toBeInTheDocument();
  });

  it('shows a duration once a step completes', () => {
    render(
      <ExecutionLog
        timeline={[tool({ result: 'ok', startedAt: 1000, endedAt: 3500 })]}
      />
    );
    expect(screen.getByText('2.5s')).toBeInTheDocument();
  });

  it('expands a step to reveal its arguments and result', () => {
    render(
      <ExecutionLog
        timeline={[tool({ args: { query: 'atlas budget' }, result: { count: 3 } })]}
      />
    );
    // Collapsed by default: detail is not in the DOM.
    expect(screen.queryByText(/atlas budget/)).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /called search_knowledge/ }));
    expect(screen.getByText(/arguments/)).toBeInTheDocument();
    expect(screen.getByText(/atlas budget/)).toBeInTheDocument();
    expect(screen.getByText(/"count": 3/)).toBeInTheDocument();
  });

  it('collapses and reopens the whole log', () => {
    const { container } = render(
      <ExecutionLog timeline={[tool({ result: 'ok' })]} />
    );
    expect(container.querySelectorAll('.exec-row')).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: /Execution/ }));
    expect(container.querySelectorAll('.exec-row')).toHaveLength(0);
  });

  it('renders a blocked step distinctly', () => {
    const { container } = render(
      <ExecutionLog
        timeline={[{ kind: 'law_block', article: 'A8', detail: 'needs confirmation' }]}
      />
    );
    expect(container.querySelector('.exec-row').getAttribute('data-status'))
      .toBe('blocked');
    expect(screen.getByText(/Blocked by A8/)).toBeInTheDocument();
    expect(screen.getByText(/needs confirmation/)).toBeInTheDocument();
  });

  it('renders an error step as an alert with retry', () => {
    const onRetry = vi.fn();
    render(
      <ExecutionLog
        timeline={[{ kind: 'error', message: 'boom', code: 'E9' }]}
        onRetry={onRetry}
      />
    );
    expect(screen.getByRole('alert')).toHaveTextContent('boom');
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('keeps memory notes visible in the sequence', () => {
    render(<ExecutionLog timeline={[{ kind: 'memory', fact: 'user is in CET' }]} />);
    expect(screen.getByText(/learned: user is in CET/)).toBeInTheDocument();
  });

  it('renders a delivered steering step with its author and note', () => {
    const { container } = render(
      <ExecutionLog
        timeline={[
          {
            kind: 'steering',
            author: 'bob@x.com',
            text: 'focus on billing',
            delivered: true,
          },
        ]}
      />
    );
    expect(
      container.querySelector('.exec-row').getAttribute('data-status')
    ).toBe('steering');
    expect(screen.getByText(/steered by bob@x.com/)).toBeInTheDocument();
    expect(screen.getByText(/focus on billing/)).toBeInTheDocument();
  });

  it('renders a queued steering acknowledgement as a quiet note', () => {
    const { container } = render(
      <ExecutionLog
        timeline={[
          { kind: 'steering', author: 'ana@x.com', text: 'use v2', delivered: false },
        ]}
      />
    );
    // Queued, not yet read: a plain note, not the warm steering accent.
    expect(container.querySelector('.exec-row').getAttribute('data-status'))
      .toBe('note');
    expect(screen.getByText(/ana@x.com steered — queued/)).toBeInTheDocument();
  });
});
