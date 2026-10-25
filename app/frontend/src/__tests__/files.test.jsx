// Files surface: downloads must be fetched WITH auth headers (a plain
// <a href> can't carry the bearer token used in dev/token-auth mode).

import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import Files from '../components/Files.jsx';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const session = { email: 'a@x.com', authHeaders: () => ({ Authorization: 'Bearer tok' }) };

it('downloads via fetch with auth headers, not a bare anchor navigation', async () => {
  const blob = new Blob(['hello'], { type: 'text/plain' });
  const fetchFn = vi.fn((url) => {
    if (String(url).includes('/api/files/list')) {
      return Promise.resolve({ ok: true, json: async () => ({ outputs: ['report.md'] }) });
    }
    if (String(url).includes('/api/files/download')) {
      return Promise.resolve({ ok: true, blob: async () => blob });
    }
    return Promise.resolve({ ok: true, json: async () => ({}) });
  });
  vi.stubGlobal('fetch', fetchFn);
  vi.stubGlobal('URL', { ...URL, createObjectURL: vi.fn(() => 'blob:x'), revokeObjectURL: vi.fn() });
  const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

  render(<Files session={session} />);

  const btn = await screen.findByTitle('Download report.md');
  fireEvent.click(btn);

  await waitFor(() => {
    const call = fetchFn.mock.calls.find((c) => String(c[0]).includes('/api/files/download'));
    expect(call).toBeTruthy();
    // path is passed and the auth header rides along.
    expect(String(call[0])).toContain('path=outputs%2Freport.md');
    expect(call[1].headers.Authorization).toBe('Bearer tok');
  });
  expect(clickSpy).toHaveBeenCalled();
  expect(URL.createObjectURL).toHaveBeenCalled();
});
