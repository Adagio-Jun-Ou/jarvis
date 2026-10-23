// Tests for the dev-auth bootstrap: the session must mint a dev token
// FIRST (the only unauthenticated route) and only then call /api/me
// with the Authorization header — and it must degrade gracefully in
// production where /api/dev/token 404s.

import { describe, it, expect, vi } from 'vitest';
import { bootstrapSession, buildAuthHeaders } from '../App.jsx';

const json = (body, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => body,
});

describe('bootstrapSession', () => {
  it('mints the dev token before calling /api/me, and sends it', async () => {
    const calls = [];
    const fetchFn = vi.fn(async (url, opts = {}) => {
      calls.push({ url, opts });
      if (String(url).startsWith('/api/dev/token')) return json({ token: 'tok-123' });
      if (url === '/api/me') return json({ email: 'demo@alex.dev' });
      return json({}, false, 404);
    });
    const s = await bootstrapSession(fetchFn);
    expect(calls[0].url).toMatch(/^\/api\/dev\/token/);
    expect(calls[1].url).toBe('/api/me');
    expect(calls[1].opts.headers.Authorization).toBe('Bearer tok-123');
    expect(calls[1].opts.headers['X-Forwarded-Access-Token']).toBe('tok-123');
    expect(s.token).toBe('tok-123');
    expect(s.email).toBe('demo@alex.dev');
    expect(s.authHeaders()).toEqual({
      Authorization: 'Bearer tok-123',
      'X-Forwarded-Access-Token': 'tok-123',
    });
  });

  it('keeps working when /api/dev/token 404s (production)', async () => {
    const fetchFn = vi.fn(async (url, opts = {}) => {
      if (String(url).startsWith('/api/dev/token')) return json({}, false, 404);
      if (url === '/api/me') {
        // Platform-injected headers authenticate; no Authorization set.
        expect(opts.headers).toEqual({});
        return json({ email: 'real.user@corp.com' });
      }
      return json({}, false, 404);
    });
    const s = await bootstrapSession(fetchFn);
    expect(s.token).toBe('');
    expect(s.email).toBe('real.user@corp.com');
    expect(s.authHeaders()).toEqual({});
  });

  it('survives a network error on /api/me', async () => {
    const fetchFn = vi.fn(async (url) => {
      if (String(url).startsWith('/api/dev/token')) return json({ token: 't' });
      throw new Error('offline');
    });
    const s = await bootstrapSession(fetchFn);
    expect(s.token).toBe('t');
    expect(typeof s.email).toBe('string');
  });
});

describe('buildAuthHeaders', () => {
  it('is empty without a token', () => {
    expect(buildAuthHeaders('')).toEqual({});
  });
});
