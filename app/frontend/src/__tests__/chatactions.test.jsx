// Chat interaction refinements: starter suggestions, the Build/Plan
// mode control, the autonomy selector (a REAL control — it patches the
// preference the governor budgets from), and message-level actions.

import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  render, screen, fireEvent, cleanup, waitFor,
} from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import Chat, { TurnActions } from '../components/Chat.jsx';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function sseBody(events) {
  const enc = new TextEncoder();
  const chunks = events.map((e) => enc.encode(`data: ${JSON.stringify(e)}\n\n`));
  let i = 0;
  return {
    getReader: () => ({
      read: async () =>
        i < chunks.length
          ? { value: chunks[i++], done: false }
          : { done: true, value: undefined },
    }),
  };
}

function stubFetch() {
  const calls = [];
  const fn = vi.fn((url, opts = {}) => {
    calls.push({ url: String(url), opts });
    if (String(url).startsWith('/api/chat')) {
      return Promise.resolve({
        ok: true,
        body: sseBody([
          { type: 'text', text: 'done.' },
          { type: 'done' },
        ]),
      });
    }
    if (String(url).startsWith('/api/preferences')) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ preferences: { autonomy: 'suggest' } }),
      });
    }
    return Promise.resolve({ ok: true, json: async () => ({}) });
  });
  vi.stubGlobal('fetch', fn);
  return calls;
}

describe('starters', () => {
  it('a starter chip sends its suggestion as a real message', async () => {
    const calls = stubFetch();
    render(<Chat sessionId="s1" model="auto" session={null} />);
    fireEvent.click(screen.getByText(/Summarise my inbox/));
    await waitFor(() => {
      const chat = calls.find((c) => c.url.startsWith('/api/chat'));
      expect(chat).toBeTruthy();
      expect(JSON.parse(chat.opts.body).text).toMatch(/Summarise my inbox/);
    });
  });
});

describe('mode control', () => {
  it('Plan mode rides in the chat request when selected', async () => {
    const calls = stubFetch();
    render(<Chat sessionId="s1" model="auto" session={null} />);
    fireEvent.click(screen.getByRole('radio', { name: 'Plan' }));
    fireEvent.change(screen.getByPlaceholderText(/ask alex/i), {
      target: { value: 'draft the approach' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^send$/i }));
    await waitFor(() => {
      const chat = calls.find((c) => c.url.startsWith('/api/chat'));
      expect(chat).toBeTruthy();
      expect(JSON.parse(chat.opts.body).plan_mode).toBe(true);
    });
  });
});

describe('autonomy', () => {
  it('loads the stored preference and persists a change via PATCH', async () => {
    const calls = stubFetch();
    render(<Chat sessionId="s1" model="auto" session={null} />);
    const select = await screen.findByLabelText(/how much alex may do/i);
    await waitFor(() => expect(select.value).toBe('suggest'));
    fireEvent.change(select, { target: { value: 'autopilot' } });
    await waitFor(() => {
      const patch = calls.find(
        (c) => c.url.startsWith('/api/preferences') && c.opts.method === 'PATCH'
      );
      expect(patch).toBeTruthy();
      expect(JSON.parse(patch.opts.body).autonomy).toBe('autopilot');
    });
  });
});

describe('TurnActions', () => {
  it('offers copy + regenerate on AI turns, copy + edit on user turns', () => {
    const { rerender } = render(
      <TurnActions turn={{ role: 'ai' }} onCopy={() => {}} onRegenerate={() => {}} />
    );
    expect(screen.getByTitle('Copy')).toBeInTheDocument();
    expect(screen.getByTitle('Ask this again')).toBeInTheDocument();

    rerender(
      <TurnActions turn={{ role: 'user' }} onCopy={() => {}} onEdit={() => {}} />
    );
    expect(screen.getByTitle('Edit and resend')).toBeInTheDocument();
    expect(screen.queryByTitle('Ask this again')).toBeNull();
  });

  it('the callbacks fire', () => {
    const copy = vi.fn();
    const regen = vi.fn();
    render(<TurnActions turn={{ role: 'ai' }} onCopy={copy} onRegenerate={regen} />);
    fireEvent.click(screen.getByTitle('Copy'));
    fireEvent.click(screen.getByTitle('Ask this again'));
    expect(copy).toHaveBeenCalled();
    expect(regen).toHaveBeenCalled();
  });
});
