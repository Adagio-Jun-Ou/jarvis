// The Workflows surface: lists the n8n instance's workflows, runs one
// through the capability-gated route, and animates the flow map while
// the execution is live.

import React from 'react';
import { it, expect, vi, afterEach, describe } from 'vitest';
import {
  render, screen, fireEvent, cleanup, waitFor,
} from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import Workflows, { FlowMap } from '../components/Workflows.jsx';
import { clearCache } from '../lib/swr.js';

afterEach(() => { cleanup(); vi.restoreAllMocks(); clearCache(); });

const session = { email: 'dana@alex.dev', authHeaders: () => ({ Authorization: 'Bearer t' }) };

const WF = {
  id: 'wf-1',
  name: 'Invoice sync [alex:dana@alex.dev]',
  active: false,
  node_count: 3,
  nodes: [
    { name: 'Trigger', type: 'webhook', position: [0, 0] },
    { name: 'Fetch', type: 'httpRequest', position: [200, 0] },
    { name: 'Post', type: 'slack', position: [400, 0] },
  ],
};

function stubFetch({ runStatus = 'success' } = {}) {
  const calls = [];
  vi.stubGlobal('fetch', vi.fn((url, opts = {}) => {
    calls.push({ url: String(url), opts });
    const u = String(url);
    if (u.includes('/run')) {
      return Promise.resolve({ ok: true, json: async () => ({
        execution_id: 'ex-9', status: runStatus,
      }) });
    }
    if (u.includes('/api/n8n/workflows')) {
      return Promise.resolve({ ok: true, json: async () => ({
        workflows: [WF], health: { mode: 'dev-fake' },
      }) });
    }
    return Promise.resolve({ ok: true, json: async () => ({}) });
  }));
  return calls;
}

describe('FlowMap', () => {
  it('renders one node per step, ordered by canvas position', () => {
    const { container } = render(<FlowMap nodes={WF.nodes} running={false} />);
    expect(container.querySelectorAll('.wf-node')).toHaveLength(3);
    expect(container.querySelector('.wf-map.running')).toBeNull();
  });

  it('carries the running animation class while an execution is live', () => {
    const { container } = render(<FlowMap nodes={WF.nodes} running />);
    expect(container.querySelector('.wf-map.running')).not.toBeNull();
    expect(container.querySelector('.wf-spine')).not.toBeNull();
  });
});

describe('Workflows', () => {
  it('lists workflows with their provenance names and step counts', async () => {
    stubFetch();
    render(<Workflows session={session} />);
    await screen.findByText(/Invoice sync/);
    expect(screen.getByText('3 steps')).toBeInTheDocument();
    expect(screen.getByText('inactive')).toBeInTheDocument();
  });

  it('Run posts to the gated route and shows the outcome', async () => {
    const calls = stubFetch({ runStatus: 'success' });
    render(<Workflows session={session} />);
    await screen.findByText(/Invoice sync/);
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    await waitFor(() => {
      const run = calls.find((c) => c.url.includes('/api/n8n/workflows/wf-1/run'));
      expect(run).toBeTruthy();
      expect(run.opts.method).toBe('POST');
      expect(run.opts.headers.Authorization).toBe('Bearer t');
    });
    await screen.findByText('success');
  });
});
