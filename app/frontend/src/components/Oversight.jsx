// Oversight.jsx — is the fleet behaving?
//
// Administrators only. This is deliberately not another audit log: the
// audit log answers "what happened", which nobody has time to read. This
// answers "did anything go wrong", which is the question leadership
// actually has.
//
// Three panes, in the order an executive needs them: what needs
// attention now, what is running right now, and what the limits are —
// because bounds nobody can inspect are folklore, not controls.

import React, { useCallback, useState } from 'react';
import { useResource } from '../lib/swr.js';

const SEVERITY_LABEL = {
  violation: 'Violation',
  concern: 'Concern',
  info: 'Noted',
};

const PATHOLOGY_BLURB = {
  budget_time: 'Ran past its time budget.',
  budget_tools: 'Ran past its tool-call budget.',
  budget_turns: 'Ran past its turn budget.',
  repetition: 'Repeated an identical call.',
  oscillation: 'Alternated between two approaches without progressing.',
  stagnation: 'Kept working without producing anything.',
  denial_persistence: 'Kept going after being refused.',
  denial_escalation: 'Reached for a more powerful tool after a refusal.',
  scope_drift: 'Acted outside what the goal called for.',
};

function Stat({ label, value, tone }) {
  return (
    <div className={'ov-stat' + (tone ? ` ${tone}` : '')}>
      <div className="ov-stat-v">{value}</div>
      <div className="ov-stat-l">{label}</div>
    </div>
  );
}

function Finding({ row, onAcknowledge }) {
  const [open, setOpen] = useState(false);
  let detail = null;
  try {
    detail = row.detail ? JSON.parse(row.detail) : null;
  } catch {
    detail = row.detail || null;
  }
  return (
    <article className={'ov-finding ' + (row.severity || 'info')}>
      <header className="ov-finding-head">
        <span className={'gd-authority ' + (row.severity || 'info')}>
          {SEVERITY_LABEL[row.severity] || row.severity}
        </span>
        <button type="button" className="ov-finding-summary" onClick={() => setOpen((v) => !v)}>
          {row.summary}
        </button>
        <span className="ov-finding-src">{row.source}</span>
      </header>
      <div className="ov-finding-meta">
        <span>{row.user_email || 'unknown user'}</span>
        {row.run_id ? <code className="gd-cap">{String(row.run_id).slice(0, 18)}</code> : null}
        {row.acknowledged_by ? (
          <span className="ov-ack">reviewed by {row.acknowledged_by}</span>
        ) : (
          <button
            type="button"
            className="btn ov-ack-btn"
            onClick={() => onAcknowledge(row.run_id)}
          >
            Mark reviewed
          </button>
        )}
      </div>
      {open && detail ? (
        <pre className="exec-pre">
          {typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2)}
        </pre>
      ) : null}
    </article>
  );
}

export default function Oversight({ session }) {
  const authHeaders = useCallback(() => session?.authHeaders?.() || {}, [session]);
  const [nonce, setNonce] = useState(0);
  const [filter, setFilter] = useState('');
  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  const summary = useResource(
    session?.email ? `ov-summary:${session.email}:${nonce}` : null,
    (signal) =>
      fetch('/api/oversight/summary', { signal, headers: authHeaders() }).then((r) =>
        r.ok ? r.json() : { forbidden: r.status === 403 }
      ),
    { maxAge: 5000 }
  );

  const findings = useResource(
    session?.email ? `ov-findings:${session.email}:${filter}:${nonce}` : null,
    (signal) =>
      fetch(`/api/oversight/findings?severity=${filter}`, {
        signal,
        headers: authHeaders(),
      }).then((r) => (r.ok ? r.json() : { findings: [] })),
    { maxAge: 5000 }
  );

  const runs = useResource(
    session?.email ? `ov-runs:${session.email}:${nonce}` : null,
    (signal) =>
      fetch('/api/oversight/runs', { signal, headers: authHeaders() }).then((r) =>
        r.ok ? r.json() : { live: [], recent: [] }
      ),
    { maxAge: 4000 }
  );

  const violations = useResource(
    session?.email ? `ov-violations:${session.email}:${nonce}` : null,
    (signal) =>
      fetch('/api/oversight/violations', { signal, headers: authHeaders() }).then((r) =>
        r.ok ? r.json() : { violations: [], summary: null }
      ),
    { maxAge: 10000 }
  );

  const limits = useResource(
    session?.email ? `ov-limits:${session.email}` : null,
    (signal) =>
      fetch('/api/oversight/limits', { signal, headers: authHeaders() }).then((r) =>
        r.ok ? r.json() : null
      ),
    { maxAge: 60000 }
  );

  const acknowledge = async (runId) => {
    await fetch('/api/oversight/findings/acknowledge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ run_id: runId }),
    });
    refresh();
  };

  if (summary.data?.forbidden) {
    return (
      <div className="ov">
        <div className="bd-empty">
          Oversight is limited to administrators. Your role does not include it.
        </div>
      </div>
    );
  }

  const s = summary.data?.summary;
  const live = runs.data?.live || [];

  return (
    <div className="ov">
      <section className="ov-row">
        <Stat label="Needs attention" value={s?.needs_attention ?? '—'}
              tone={s?.needs_attention ? 'bad' : 'good'} />
        <Stat label="Violations" value={s?.by_severity?.violation ?? '—'}
              tone={s?.by_severity?.violation ? 'bad' : 'good'} />
        <Stat label="Concerns" value={s?.by_severity?.concern ?? '—'} />
        <Stat label="Agents running now" value={runs.data?.live_count ?? '—'}
              tone={live.length ? 'live' : ''} />
        <Stat label="Users with findings" value={s?.users_with_findings ?? '—'} />
      </section>

      {runs.data?.interrupted_count ? (
        <section className="ov-panel">
          <div className="gd-group-h">
            Work that was dropped
            <span className="gd-group-blurb">
              These runs stopped because the process working them went away.
              Nothing was concluded and the tasks are back on the board — but
              somebody asked for this work and has not been told it failed.
            </span>
          </div>
          {runs.data.interrupted.map((r) => (
            <div className="ov-run" key={r.run_id}>
              <span className="ov-vio-warned">● interrupted</span>
              <span className="ov-run-id">{String(r.run_id).slice(0, 20)}</span>
              <span className="ov-run-who">{r.started_by}</span>
              <span className="ov-run-steps">task {r.task_id}</span>
            </div>
          ))}
        </section>
      ) : null}

      {live.length ? (
        <section className="ov-panel">
          <h3 className="gd-group-h">Running now</h3>
          {live.map((r) => (
            <div className="ov-run" key={r.run_id}>
              <span className="bd-live">● running</span>
              <span className="ov-run-id">{String(r.run_id).slice(0, 20)}</span>
              <span className="ov-run-who">{r.started_by}</span>
              <span className="ov-run-steps">{r.steps} steps</span>
            </div>
          ))}
        </section>
      ) : null}

      <section className="ov-panel">
        <div className="gd-group-h">
          Findings
          <span className="gd-group-blurb">
            Deterministic checks and model review of what the agents did.
          </span>
        </div>
        <div className="ov-filters" role="group" aria-label="Filter by severity">
          {['', 'violation', 'concern', 'info'].map((sev) => (
            <button
              key={sev || 'all'}
              type="button"
              aria-pressed={filter === sev}
              className={'bd-tab' + (filter === sev ? ' active' : '')}
              onClick={() => setFilter(sev)}
            >
              {sev ? SEVERITY_LABEL[sev] : 'All'}
            </button>
          ))}
        </div>
        {(findings.data?.findings || []).map((row, i) => (
          <Finding key={`${row.run_id}-${row.code}-${i}`} row={row}
                   onAcknowledge={acknowledge} />
        ))}
        {findings.data && !findings.data.findings.length ? (
          <div className="bd-empty">
            Nothing flagged. Every run so far stayed inside its bounds.
          </div>
        ) : null}
      </section>

      <section className="ov-panel">
        <div className="gd-group-h">
          Constitutional record
          <span className="gd-group-blurb">
            What the law stopped, and what it only noted. Separate from the
            findings above: those review a run afterwards, these are refusals
            made while the agent was attempting the action.
          </span>
        </div>
        {violations.data?.summary?.total ? (
          <div className="ov-vio-counts">
            <span className="ov-vio-blocked">
              {violations.data.summary.by_disposition.blocked || 0} blocked
            </span>
            <span className="ov-vio-warned">
              {violations.data.summary.by_disposition.warned || 0} noted
            </span>
            <span className="gd-group-blurb">
              across {violations.data.summary.distinct_users} user
              {violations.data.summary.distinct_users === 1 ? '' : 's'}
            </span>
          </div>
        ) : null}
        {(violations.data?.violations || []).length ? (
          <table className="ov-table">
            <thead>
              <tr>
                <th>When</th><th>Article</th><th>Action</th>
                <th>Outcome</th><th>Who</th><th>Why</th>
              </tr>
            </thead>
            <tbody>
              {violations.data.violations.map((v) => (
                <tr key={v.id}>
                  <td>{v.ts}</td>
                  <td>{v.article_id}</td>
                  <td>{v.action}</td>
                  <td className={
                    v.disposition === 'blocked' ? 'ov-vio-blocked' : 'ov-vio-warned'
                  }>{v.disposition}</td>
                  <td>{v.user_email}</td>
                  <td>{v.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="bd-empty">
            No constitutional events recorded.
          </div>
        )}
      </section>

      {limits.data ? (
        <section className="ov-panel">
          <div className="gd-group-h">
            Execution limits in force
            <span className="gd-group-blurb">
              Every agent run is bounded by these. A run that exceeds one is
              stopped and recorded.
            </span>
          </div>
          <table className="ov-table">
            <thead>
              <tr>
                <th>Autonomy</th><th>Tool calls</th><th>Turns</th><th>Time</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(limits.data.budgets).map(([level, b]) => (
                <tr key={level}>
                  <td>{level}</td>
                  <td>{b.max_tool_calls}</td>
                  <td>{b.max_llm_turns}</td>
                  <td>{Math.round(b.max_seconds)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="ov-pathologies">
            {(limits.data.pathologies || []).map((p) => (
              <span key={p} className="gd-tag" title={PATHOLOGY_BLURB[p] || p}>
                {p.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
