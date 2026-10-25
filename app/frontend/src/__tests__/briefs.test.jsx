// BriefsStrip: renders the "Prepared overnight" briefs and dismisses one
// via POST /api/briefs/consume (the lifecycle's terminal hop).

import React from 'react';
import { it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import BriefsStrip from '../components/BriefsStrip.jsx';
import { clearCache } from '../lib/swr.js';

afterEach(() => { cleanup(); vi.restoreAllMocks(); clearCache(); });

const session = { email: 'a@x.com', authHeaders: () => ({ Authorization: 'Bearer tok' }) };

it('renders briefs and consumes one with auth', async () => {
  const briefs = {
    brief: { brief: 'Two priority emails and a due task.' },
    standing_orders: [
      { brief_id: 'b-1', order_id: 'so-1', compiled_context: 'Follow up on the Q1 contract.',
        suggested_actions: [{ action: 'Draft reply', capability: 'email.draft' }] },
    ],
  };
  const calls = [];
  const fetchFn = vi.fn((url, opts = {}) => {
    calls.push({ url: String(url), opts });
    if (String(url).includes('/api/briefs/consume')) return Promise.resolve({ ok: true, json: async () => ({ ok: true }) });
    return Promise.resolve({ ok: true, json: async () => briefs });
  });
  vi.stubGlobal('fetch', fetchFn);

  render(<BriefsStrip session={session} />);

  // The morning brief + the standing-order brief render.
  await screen.findByText(/Two priority emails/);
  expect(screen.getByText(/Follow up on the Q1 contract/)).toBeInTheDocument();
  expect(screen.getByText(/Draft reply/)).toBeInTheDocument();

  fireEvent.click(screen.getByText('Dismiss'));
  await waitFor(() => {
    const c = calls.find((x) => x.url.includes('/api/briefs/consume'));
    expect(c).toBeTruthy();
    expect(c.opts.method).toBe('POST');
    expect(c.opts.headers.Authorization).toBe('Bearer tok');
    expect(JSON.parse(c.opts.body).brief_id).toBe('b-1');
  });
});

it('renders nothing when there are no briefs', async () => {
  const fetchFn = vi.fn(() => Promise.resolve({ ok: true, json: async () => ({ brief: null, standing_orders: [] }) }));
  vi.stubGlobal('fetch', fetchFn);
  const { container } = render(<BriefsStrip session={session} />);
  await waitFor(() => expect(fetchFn).toHaveBeenCalled());
  expect(container.querySelector('.briefs-strip')).toBeNull();
});
