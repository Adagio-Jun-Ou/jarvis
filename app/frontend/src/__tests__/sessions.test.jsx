// Sessions surface: rename (PATCH) and delete (DELETE) are wired with auth.

import React from 'react';
import { it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import Sessions from '../components/Sessions.jsx';
import { clearCache } from '../lib/swr.js';

afterEach(() => { cleanup(); vi.restoreAllMocks(); clearCache(); });

const session = { email: 'a@x.com', authHeaders: () => ({ Authorization: 'Bearer tok' }) };

function stubFetch(calls) {
  return vi.fn((url, opts = {}) => {
    calls.push({ url: String(url), opts });
    if (String(url).includes('/api/sessions') && (!opts.method || opts.method === 'GET')) {
      return Promise.resolve({ ok: true, json: async () => ({ sessions: [
        { session_id: 's-1', title: 'Draft', last_active_at: null, created_at: null },
      ] }) });
    }
    return Promise.resolve({ ok: true, json: async () => ({ ok: true }) });
  });
}

it('renames a session via PATCH with auth', async () => {
  const calls = [];
  vi.stubGlobal('fetch', stubFetch(calls));
  vi.stubGlobal('prompt', vi.fn(() => 'Renamed'));
  render(<Sessions session={session} />);
  const btn = await screen.findByTitle('Rename');
  fireEvent.click(btn);
  await waitFor(() => {
    const c = calls.find((x) => x.url.includes('/api/sessions/s-1') && x.opts.method === 'PATCH');
    expect(c).toBeTruthy();
    expect(c.opts.headers.Authorization).toBe('Bearer tok');
    expect(JSON.parse(c.opts.body).title).toBe('Renamed');
  });
});

it('deletes a session via DELETE after confirm', async () => {
  const calls = [];
  vi.stubGlobal('fetch', stubFetch(calls));
  vi.stubGlobal('confirm', vi.fn(() => true));
  render(<Sessions session={session} />);
  const btn = await screen.findByTitle('Delete');
  fireEvent.click(btn);
  await waitFor(() => {
    const c = calls.find((x) => x.url.includes('/api/sessions/s-1') && x.opts.method === 'DELETE');
    expect(c).toBeTruthy();
    expect(c.opts.headers.Authorization).toBe('Bearer tok');
  });
});

it('does not delete when confirm is cancelled', async () => {
  const calls = [];
  vi.stubGlobal('fetch', stubFetch(calls));
  vi.stubGlobal('confirm', vi.fn(() => false));
  render(<Sessions session={session} />);
  fireEvent.click(await screen.findByTitle('Delete'));
  await Promise.resolve();
  expect(calls.find((x) => x.opts.method === 'DELETE')).toBeFalsy();
});
