// Tests for the Chat surface: activity chips, the elapsed-time
// waiting indicator, Stop (abort), and error Retry.

import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import Chat, { ActivityTimeline, ErrorCard, WaitingIndicator } from '../components/Chat.jsx';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

// --- helpers ----------------------------------------------------------

/** Build a fake SSE res.body from a list of event objects. */
function sseBody(events) {
  const enc = new TextEncoder();
  const chunks = events.map((e) => enc.encode(`data: ${JSON.stringify(e)}\n\n`));
  let i = 0;
  return {
    getReader: () => ({
      read: async () =>
        i < chunks.length ? { value: chunks[i++], done: false } : { done: true, value: undefined },
    }),
  };
}

/** Install a fetch stub; chatImpl handles POST /api/chat. */
function stubFetch(chatImpl) {
  const fn = vi.fn((url, opts = {}) => {
    if (typeof url === 'string' && url.startsWith('/api/chat')) return chatImpl(url, opts);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

function renderChat() {
  return render(<Chat sessionId="s1" model="auto" session={null} />);
}

function sendMessage(text) {
  fireEvent.change(screen.getByPlaceholderText(/ask alex/i), { target: { value: text } });
  fireEvent.click(screen.getByRole('button', { name: /^send$/i }));
}

// --- ActivityTimeline chips ------------------------------------------

describe('ActivityTimeline', () => {
  it('renders a chip per event kind', () => {
    const timeline = [
      { kind: 'tool', tool: 'list_my_tasks', args: { limit: 5 } },
      { kind: 'law_block', article: 'Article 4', detail: 'privacy' },
      { kind: 'memory', fact: 'user is in CET' },
      { kind: 'artifact', path: 'out/report.pdf', artifactKind: 'pdf' },
    ];
    render(<ActivityTimeline timeline={timeline} onRetry={() => {}} />);
    expect(screen.getByText(/called list_my_tasks/)).toBeInTheDocument();
    expect(screen.getByText(/blocked by Article 4/i)).toBeInTheDocument();
    expect(screen.getByText(/learned: user is in CET/)).toBeInTheDocument();
    // Artifact download is a button that fetches with auth (not a bare
    // <a href>, which can't carry the bearer token).
    const dl = screen.getByText(/report\.pdf/).closest('button');
    expect(dl).toBeInTheDocument();
  });

  it('artifact chip downloads via authenticated fetch on click', async () => {
    const blob = new Blob(['x'], { type: 'application/pdf' });
    const fetchFn = vi.fn(() => Promise.resolve({ ok: true, blob: async () => blob }));
    vi.stubGlobal('fetch', fetchFn);
    vi.stubGlobal('URL', { ...URL, createObjectURL: vi.fn(() => 'blob:x'), revokeObjectURL: vi.fn() });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const timeline = [{ kind: 'artifact', path: 'out/report.pdf', artifactKind: 'pdf' }];
    render(<ActivityTimeline timeline={timeline} onRetry={() => {}} authHeaders={() => ({ Authorization: 'Bearer tok' })} />);
    fireEvent.click(screen.getByText(/report\.pdf/).closest('button'));
    await waitFor(() => {
      const call = fetchFn.mock.calls.find((c) => String(c[0]).includes('/api/files/download'));
      expect(call).toBeTruthy();
      expect(call[1].headers.Authorization).toBe('Bearer tok');
    });
  });

  it('marks resolved tool calls with a check', () => {
    render(<ActivityTimeline timeline={[{ kind: 'tool', tool: 'x', result: 'ok' }]} />);
    expect(screen.getByText(/called x ✓/)).toBeInTheDocument();
  });

  it('renders error entries as an ErrorCard with Retry', () => {
    const onRetry = vi.fn();
    render(
      <ActivityTimeline timeline={[{ kind: 'error', message: 'boom', code: 'E9' }]} onRetry={onRetry} />
    );
    expect(screen.getByRole('alert')).toHaveTextContent('boom');
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});

describe('ErrorCard', () => {
  it('shows message and code', () => {
    render(<ErrorCard error={{ message: 'nope', code: 'E1' }} onRetry={() => {}} />);
    expect(screen.getByRole('alert')).toHaveTextContent('nope');
    expect(screen.getByRole('alert')).toHaveTextContent('E1');
  });
});

// --- WaitingIndicator -------------------------------------------------

describe('WaitingIndicator', () => {
  it('ticks the elapsed seconds and shows a warmup note past 10s', () => {
    vi.useFakeTimers();
    render(<WaitingIndicator since={Date.now()} />);
    expect(screen.getByTestId('waiting')).toHaveTextContent('thinking… 0s');
    act(() => { vi.advanceTimersByTime(3000); });
    expect(screen.getByTestId('waiting')).toHaveTextContent('thinking… 3s');
    expect(screen.queryByText(/warming up/)).toBeNull();
    act(() => { vi.advanceTimersByTime(9000); });
    expect(screen.getByTestId('waiting')).toHaveTextContent('12s');
    expect(screen.getByText(/local model warming up/)).toBeInTheDocument();
  });
});

// --- Full chat: streaming chips, abort, retry ------------------------

describe('Chat streaming', () => {
  it('renders activity chips inline as SSE events arrive', async () => {
    stubFetch(() =>
      Promise.resolve({
        ok: true,
        body: sseBody([
          { type: 'tool', tool: 'list_my_tasks', args: null, result: null },
          { type: 'memory', fact: 'likes short answers' },
          { type: 'text', text: 'Here you go.' },
          { type: 'done' },
        ]),
      })
    );
    renderChat();
    sendMessage('show my tasks');
    // Optimistic user bubble is instant.
    expect(screen.getByText('show my tasks')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/called list_my_tasks/)).toBeInTheDocument();
      expect(screen.getByText(/learned: likes short answers/)).toBeInTheDocument();
      expect(screen.getByText('Here you go.')).toBeInTheDocument();
    });
  });

  it('Stop aborts the in-flight request cleanly', async () => {
    stubFetch((url, opts) =>
      new Promise((resolve, reject) => {
        opts.signal.addEventListener('abort', () =>
          reject(new DOMException('aborted', 'AbortError'))
        );
      })
    );
    renderChat();
    sendMessage('slow question');
    const stopBtn = await screen.findByRole('button', { name: /stop/i });
    fireEvent.click(stopBtn);
    await waitFor(() => {
      expect(screen.getByText('Stopped.')).toBeInTheDocument();
      // Composer is back to Send (not busy/frozen).
      expect(screen.getByRole('button', { name: /^send$/i })).toBeInTheDocument();
    });
  });

  it('SSE error events show an error card whose Retry re-sends the last message', async () => {
    let calls = 0;
    const fetchFn = stubFetch(() => {
      calls += 1;
      return Promise.resolve({
        ok: true,
        body: sseBody(
          calls === 1
            ? [{ type: 'error', code: 'E500', message: 'model crashed' }, { type: 'done' }]
            : [{ type: 'text', text: 'second try worked' }, { type: 'done' }]
        ),
      });
    });
    renderChat();
    sendMessage('do the thing');
    const retryBtn = await screen.findByRole('button', { name: /retry/i });
    expect(screen.getByRole('alert')).toHaveTextContent('model crashed');
    fireEvent.click(retryBtn);
    await waitFor(() => {
      expect(screen.getByText('second try worked')).toBeInTheDocument();
    });
    const chatCalls = fetchFn.mock.calls.filter(([u]) => String(u).startsWith('/api/chat'));
    expect(chatCalls).toHaveLength(2);
    expect(JSON.parse(chatCalls[1][1].body).text).toBe('do the thing');
  });
});
