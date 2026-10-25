// Workflows — the enterprise n8n arm, manifested.
//
// Lists the workflows on the organisation's n8n instance, runs them
// (capability-gated server-side), and shows each one as a small
// animated flow map: nodes in execution order, the connecting line
// carrying a moving dash while a run is live — the same "you can see
// it working" principle as the NowLine, for automations.

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useResource } from '../lib/swr.js';
import { toast } from '../lib/toast.jsx';

/* The animated mini-map. Nodes are laid out left-to-right by their n8n
   canvas x-position; the spine carries flowing dashes while running. */
export function FlowMap({ nodes, running }) {
  const shown = [...(nodes || [])]
    .sort((a, b) => (a.position?.[0] ?? 0) - (b.position?.[0] ?? 0))
    .slice(0, 8);
  if (!shown.length) return null;
  const W = 300;
  const H = 46;
  const pad = 22;
  const step = shown.length > 1 ? (W - pad * 2) / (shown.length - 1) : 0;
  return (
    <svg
      className={'wf-map' + (running ? ' running' : '')}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Workflow with ${shown.length} steps${running ? ', running' : ''}`}
    >
      {shown.length > 1 && (
        <line
          className="wf-spine"
          x1={pad} y1={H / 2} x2={W - pad} y2={H / 2}
        />
      )}
      {shown.map((n, i) => (
        <g key={i} transform={`translate(${pad + i * step}, ${H / 2})`}>
          <circle className="wf-node" r="6" style={{ animationDelay: `${i * 0.18}s` }} />
          <title>{n.name || n.type}</title>
        </g>
      ))}
    </svg>
  );
}

function StatusChip({ status }) {
  if (!status) return null;
  const cls = status === 'success' ? 'ok'
    : status === 'error' ? 'err'
    : 'live';
  return <span className={`wf-status ${cls}`}>{status}</span>;
}

export default function Workflows({ session }) {
  const authHeaders = useCallback(() => session?.authHeaders?.() || {}, [session]);
  const [nonce, setNonce] = useState(0);
  const [runs, setRuns] = useState({}); // workflow_id -> {execution_id, status}
  const pollers = useRef({});

  const { data, loading } = useResource(
    session?.email ? `n8n:${session.email}:${nonce}` : null,
    (signal) =>
      fetch('/api/n8n/workflows', { signal, headers: authHeaders() }).then(
        (r) => (r.ok ? r.json() : r.json().then((d) => ({ error: d.error, workflows: [] })))
      ),
    { maxAge: 15000 }
  );
  const workflows = data?.workflows || [];

  useEffect(() => () => {
    Object.values(pollers.current).forEach(clearInterval);
  }, []);

  const poll = (workflowId, executionId) => {
    clearInterval(pollers.current[workflowId]);
    pollers.current[workflowId] = setInterval(async () => {
      try {
        const r = await fetch(`/api/n8n/executions/${executionId}`, {
          headers: authHeaders(),
        });
        if (!r.ok) return;
        const d = await r.json();
        setRuns((cur) => ({ ...cur, [workflowId]: { execution_id: executionId, status: d.status } }));
        if (d.status !== 'running') clearInterval(pollers.current[workflowId]);
      } catch { /* transient */ }
    }, 1500);
  };

  const run = async (wf) => {
    setRuns((cur) => ({ ...cur, [wf.id]: { status: 'running' } }));
    try {
      const r = await fetch(`/api/n8n/workflows/${wf.id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ payload: {} }),
      });
      const d = await r.json();
      if (!r.ok) {
        toast(d.error || 'Could not run the workflow');
        setRuns((cur) => ({ ...cur, [wf.id]: { status: 'error' } }));
        return;
      }
      const status = d.status || 'running';
      setRuns((cur) => ({ ...cur, [wf.id]: { execution_id: d.execution_id, status } }));
      if (status === 'running' && d.execution_id) poll(wf.id, d.execution_id);
    } catch {
      toast('Could not reach the workflow arm');
      setRuns((cur) => ({ ...cur, [wf.id]: { status: 'error' } }));
    }
  };

  return (
    <div className="wf-page" aria-label="Workflows">
      <div className="wf-head">
        <h2>Workflows</h2>
        <span className="wf-sub">
          The organisation&apos;s n8n instance
          {data?.health?.mode === 'dev-fake' ? ' · dev sandbox' : ''}
        </span>
        <button type="button" className="btn" onClick={() => setNonce((n) => n + 1)}>
          Refresh
        </button>
      </div>
      {data?.error ? (
        <div className="wf-empty">
          {String(data.error).includes('forbidden')
            ? 'You do not hold the workflow.read capability.'
            : data.error}
        </div>
      ) : loading && !workflows.length ? (
        <div className="wf-empty">Loading workflows…</div>
      ) : !workflows.length ? (
        <div className="wf-empty">
          No workflows yet. Ask Alex to build one — it can create
          workflows here (they arrive inactive, for a person to review
          and switch on).
        </div>
      ) : (
        <div className="wf-grid">
          {workflows.map((wf) => {
            const runState = runs[wf.id];
            const running = runState?.status === 'running';
            return (
              <div className="wf-card" key={wf.id}>
                <div className="wf-card-top">
                  <span className="wf-name">{wf.name}</span>
                  <span className={'wf-active' + (wf.active ? ' on' : '')}>
                    {wf.active ? 'active' : 'inactive'}
                  </span>
                </div>
                <FlowMap nodes={wf.nodes} running={running} />
                <div className="wf-card-foot">
                  <span className="wf-nodes">{wf.node_count} step{wf.node_count === 1 ? '' : 's'}</span>
                  <StatusChip status={runState?.status} />
                  <button
                    type="button"
                    className="btn primary"
                    disabled={running}
                    onClick={() => run(wf)}
                  >
                    {running ? 'Running…' : 'Run'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
