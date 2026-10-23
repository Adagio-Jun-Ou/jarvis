// Guidance.jsx — standing orders and the best-practice library.
//
// These are the two surfaces that make Alex behave like *this* company's
// assistant. A standing order is an instrument: written once, assigned
// to people, and runnable on demand against the real enterprise system,
// with its capability ceiling shown before you press the button. A
// practice is policy: it enters every relevant turn as a directive, and
// this screen shows exactly which ones are in force and why.

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useResource } from '../lib/swr.js';
import ExecutionLog from './ExecutionLog.jsx';
import { renderMarkdown } from '../lib/markdown.jsx';
import { IconSpark } from './icons.jsx';
import { toast } from '../lib/toast.jsx';

const AUTHORITY_ORDER = ['mandatory', 'recommended', 'advisory'];
const AUTHORITY_BLURB = {
  mandatory: 'Must be followed, and cited in the answer.',
  recommended: 'Followed unless the request gives a reason not to.',
  advisory: 'Context the agent should weigh.',
};

/* ---------------------------------------------------------------- orders */

function OrderRunPanel({ orderId, runId, authHeaders, onDone }) {
  const [events, setEvents] = useState([]);
  const acRef = useRef(null);

  useEffect(() => {
    if (!runId) return undefined;
    setEvents([]);
    const ac = new AbortController();
    acRef.current = ac;
    (async () => {
      try {
        const r = await fetch(`/api/orders/runs/${runId}/stream`, {
          headers: authHeaders(),
          signal: ac.signal,
        });
        if (!r.ok || !r.body) return;
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() ?? '';
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
              const ev = JSON.parse(line.slice(6));
              setEvents((p) => [...p, ev]);
              if (ev.type === 'run_finished') onDone?.(ev);
            } catch { /* keep-alive */ }
          }
        }
      } catch { /* aborted */ }
    })();
    return () => ac.abort();
  }, [runId, orderId, authHeaders, onDone]);

  const finished = events.find((e) => e.type === 'run_finished');
  const timeline = events
    .filter((e) => ['tool', 'context', 'memory', 'law_block', 'artifact', 'error'].includes(e.type))
    .map((e) =>
      e.type === 'tool'
        ? { kind: 'tool', tool: e.tool, args: e.args, result: e.result }
        : e.type === 'context'
          ? { kind: 'context', source: e.source, summary: e.summary, items: e.items }
          : e.type === 'memory'
            ? { kind: 'memory', fact: e.text || e.fact }
            : e.type === 'law_block'
              ? { kind: 'law_block', article: e.article, detail: e.detail }
              : e.type === 'artifact'
                ? { kind: 'artifact', path: e.path, artifactKind: e.kind, preview: e.preview }
                : { kind: 'error', message: e.message, code: e.code }
    );

  if (!runId) return null;
  return (
    <div className="gd-run">
      <ExecutionLog timeline={timeline} streaming={!finished} authHeaders={authHeaders} />
      {finished ? (
        <div className="bd-summary">
          <div className="bd-summary-h">Result</div>
          {renderMarkdown(finished.summary || '')}
        </div>
      ) : null}
    </div>
  );
}

function OrderCard({ order, authHeaders, onChanged, myCapabilities }) {
  const [open, setOpen] = useState(false);
  const [runId, setRunId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const [invite, setInvite] = useState('');

  const runNow = async () => {
    setBusy(true);
    try {
      const r = await fetch(`/api/orders/${order.order_id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ note }),
      });
      if (!r.ok) {
        toast('Could not run this order.');
        return;
      }
      const d = await r.json();
      setRunId(d.run_id);
      setOpen(true);
      setNote('');
    } finally {
      setBusy(false);
    }
  };

  const assign = async () => {
    const member = invite.trim();
    if (!member) return;
    const r = await fetch(`/api/orders/${order.order_id}/assign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ members: [member] }),
    });
    if (!r.ok) {
      toast('Only the owner can assign this order.');
      return;
    }
    setInvite('');
    onChanged?.();
  };

  const toggleActive = async () => {
    await fetch(`/api/orders/${order.order_id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ active: !order.active }),
    });
    onChanged?.();
  };

  return (
    <article className={'gd-card' + (order.active ? '' : ' inactive')}>
      <header className="gd-card-head">
        <button type="button" className="gd-card-title" onClick={() => setOpen((v) => !v)}>
          <span className="gd-caret" aria-hidden="true">{open ? '▾' : '▸'}</span>
          {order.title}
        </button>
        {!order.active ? <span className="gd-tag off">paused</span> : null}
        {order.ambient ? <span className="gd-tag">always on</span> : null}
        {order.runnable ? (
          <button type="button" className="btn primary gd-run-btn" onClick={runNow} disabled={busy}>
            <IconSpark width={13} height={13} /> Run now
          </button>
        ) : null}
      </header>

      <p className="gd-instruction">{order.instruction}</p>

      <div className="gd-meta">
        <span className="gd-meta-k">Runs with</span>
        {order.effective_capabilities?.length ? (
          order.effective_capabilities.map((c) => (
            <code key={c} className="gd-cap">{c}</code>
          ))
        ) : (
          <span className="gd-none">nothing — you hold none of its capabilities</span>
        )}
      </div>

      {open ? (
        <div className="gd-expand">
          <div className="gd-meta">
            <span className="gd-meta-k">Assigned to</span>
            {order.audience.map((a) => <span key={a} className="gd-tag">{a}</span>)}
          </div>
          {order.applies_to?.length ? (
            <div className="gd-meta">
              <span className="gd-meta-k">Applies to</span>
              {order.applies_to.map((s) => <span key={s} className="gd-tag">{s}</span>)}
            </div>
          ) : null}

          {order.runnable ? (
            <div className="gd-row">
              <input
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Note for this run (optional)…"
                aria-label="Note for this run"
              />
            </div>
          ) : null}

          {order.is_owner ? (
            <div className="gd-row">
              <input
                value={invite}
                onChange={(e) => setInvite(e.target.value)}
                placeholder="Assign to someone (email, or 'all')…"
                aria-label="Assign this order"
              />
              <button type="button" className="btn" onClick={assign}>Assign</button>
              <button type="button" className="btn" onClick={toggleActive}>
                {order.active ? 'Pause' : 'Resume'}
              </button>
            </div>
          ) : null}

          <OrderRunPanel
            orderId={order.order_id}
            runId={runId}
            authHeaders={authHeaders}
            onDone={onChanged}
          />
          {!runId && order.last_run ? (
            <div className="bd-summary">
              <div className="bd-summary-h">
                Last run · {order.last_run.status} · {order.last_run.steps} steps
              </div>
              {renderMarkdown(order.last_run.summary || '')}
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

function NewOrder({ authHeaders, myCapabilities, onCreated }) {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState('');
  const [instruction, setInstruction] = useState('');
  const [appliesTo, setAppliesTo] = useState('');
  const [caps, setCaps] = useState([]);

  const toggleCap = (c) =>
    setCaps((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]));

  const submit = async (e) => {
    e.preventDefault();
    if (!title.trim() || !instruction.trim()) return;
    await fetch('/api/orders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({
        title,
        instruction,
        applies_to: appliesTo.split(',').map((s) => s.trim()).filter(Boolean),
        allowed_capabilities: caps,
      }),
    });
    setTitle(''); setInstruction(''); setAppliesTo(''); setCaps([]);
    setOpen(false);
    onCreated?.();
  };

  if (!open) {
    return (
      <button type="button" className="btn" onClick={() => setOpen(true)}>
        + New standing order
      </button>
    );
  }
  return (
    <form className="gd-form" onSubmit={submit}>
      <input
        autoFocus value={title} onChange={(e) => setTitle(e.target.value)}
        placeholder="What is this order called?" aria-label="Order title"
      />
      <textarea
        value={instruction} onChange={(e) => setInstruction(e.target.value)}
        placeholder="What should Alex do when this runs?" rows={3}
        aria-label="Instruction"
      />
      <input
        value={appliesTo} onChange={(e) => setAppliesTo(e.target.value)}
        placeholder="Subjects it applies to, comma separated (blank = always)"
        aria-label="Applies to"
      />
      <fieldset className="gd-caps">
        <legend>What may it use? You can only grant what you hold.</legend>
        <div className="gd-cap-grid">
          {(myCapabilities || []).map((c) => (
            <label key={c} className="gd-cap-pick">
              <input
                type="checkbox" checked={caps.includes(c)}
                onChange={() => toggleCap(c)}
              />
              <code>{c}</code>
            </label>
          ))}
        </div>
      </fieldset>
      <div className="gd-row">
        <button type="submit" className="btn primary" disabled={!title.trim() || !instruction.trim()}>
          Create order
        </button>
        <button type="button" className="btn" onClick={() => setOpen(false)}>Cancel</button>
      </div>
    </form>
  );
}

/* ------------------------------------------------------------- practices */

function PracticeCard({ practice }) {
  const [open, setOpen] = useState(false);
  return (
    <article className={'gd-card practice ' + practice.authority}>
      <header className="gd-card-head">
        <button type="button" className="gd-card-title" onClick={() => setOpen((v) => !v)}>
          <span className="gd-caret" aria-hidden="true">{open ? '▾' : '▸'}</span>
          {practice.title}
        </button>
        <span className={'gd-authority ' + practice.authority}>{practice.authority}</span>
        {practice.version > 1 ? <span className="gd-tag">v{practice.version}</span> : null}
      </header>
      <p className="gd-instruction">{practice.guidance}</p>
      {open ? (
        <div className="gd-expand">
          {practice.rationale ? (
            <p className="gd-why"><b>Why:</b> {practice.rationale}</p>
          ) : null}
          {practice.anti_patterns?.length ? (
            <div className="gd-meta">
              <span className="gd-meta-k">Do not</span>
              {practice.anti_patterns.map((a) => <span key={a} className="gd-tag off">{a}</span>)}
            </div>
          ) : null}
          {practice.capabilities?.length ? (
            <div className="gd-meta">
              <span className="gd-meta-k">Fires on</span>
              {practice.capabilities.map((c) => <code key={c} className="gd-cap">{c}</code>)}
            </div>
          ) : null}
          <div className="gd-meta">
            <span className="gd-meta-k">Applies to</span>
            {practice.applies_to?.length
              ? practice.applies_to.map((s) => <span key={s} className="gd-tag">{s}</span>)
              : <span className="gd-none">every request (house style)</span>}
          </div>
        </div>
      ) : null}
    </article>
  );
}

/* Try a phrasing and see which practices would govern it. */
function WhyPanel({ authHeaders }) {
  const [q, setQ] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  const check = async (e) => {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    try {
      const r = await fetch(`/api/practices/match?q=${encodeURIComponent(q)}`, {
        headers: authHeaders(),
      });
      setResult(r.ok ? await r.json() : null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="gd-why-panel">
      <form className="gd-row" onSubmit={check}>
        <input
          value={q} onChange={(e) => setQ(e.target.value)}
          placeholder="Try a request — see which guidance would apply…"
          aria-label="Test a request against the library"
        />
        <button type="submit" className="btn" disabled={busy || !q.trim()}>Check</button>
      </form>
      {result ? (
        result.matches.length ? (
          <ul className="gd-matches">
            {result.matches.map((m) => (
              <li key={m.practice_id}>
                <span className={'gd-authority ' + m.authority}>{m.authority}</span>
                <span className="gd-match-title">{m.title}</span>
                <span className="gd-match-why">matched on {m.matched_on}</span>
              </li>
            ))}
          </ul>
        ) : (
          <div className="gd-none">No practice governs that request.</div>
        )
      ) : null}
    </section>
  );
}

/* ------------------------------------------------------------------ shell */

export default function Guidance({ session }) {
  const authHeaders = useCallback(() => session?.authHeaders?.() || {}, [session]);
  const [tab, setTab] = useState('orders');
  const [nonce, setNonce] = useState(0);
  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  const orders = useResource(
    session?.email ? `orders:${session.email}:${nonce}` : null,
    (signal) => fetch('/api/orders', { signal, headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : { orders: [], my_capabilities: [] })),
    { maxAge: 3000 }
  );
  const practices = useResource(
    session?.email ? `practices:${session.email}:${nonce}` : null,
    (signal) => fetch('/api/practices', { signal, headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : { practices: [] })),
    { maxAge: 8000 }
  );

  const byAuthority = AUTHORITY_ORDER.map((a) => ({
    authority: a,
    items: (practices.data?.practices || []).filter((p) => p.authority === a),
  })).filter((g) => g.items.length);

  return (
    <div className="gd">
      <div className="gd-tabs" role="tablist">
        {[
          ['orders', 'Standing orders'],
          ['practices', 'Best practices'],
        ].map(([id, label]) => (
          <button
            key={id} role="tab" type="button"
            aria-selected={tab === id}
            className={'bd-tab' + (tab === id ? ' active' : '')}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'orders' ? (
        <section className="gd-body">
          <p className="gd-lede">
            Instructions you write once and can run any time. Each order shows
            the capabilities it will actually use — the overlap of what the
            order permits and what you hold.
          </p>
          <NewOrder
            authHeaders={authHeaders}
            myCapabilities={orders.data?.my_capabilities}
            onCreated={refresh}
          />
          {(orders.data?.orders || []).map((o) => (
            <OrderCard
              key={o.order_id} order={o} authHeaders={authHeaders}
              onChanged={refresh} myCapabilities={orders.data?.my_capabilities}
            />
          ))}
          {orders.data && !orders.data.orders.length ? (
            <div className="bd-empty">No standing orders yet.</div>
          ) : null}
        </section>
      ) : (
        <section className="gd-body">
          <p className="gd-lede">
            How this organisation works. These are not search results — a
            matching practice is put to Alex as an instruction on every
            relevant turn, and a mandatory one must be followed and cited.
          </p>
          <WhyPanel authHeaders={authHeaders} />
          {byAuthority.map((group) => (
            <div key={group.authority} className="gd-group">
              <h3 className="gd-group-h">
                {group.authority}
                <span className="gd-group-blurb">{AUTHORITY_BLURB[group.authority]}</span>
              </h3>
              {group.items.map((p) => (
                <PracticeCard key={p.practice_id} practice={p} />
              ))}
            </div>
          ))}
          {practices.data && !practices.data.practices.length ? (
            <div className="bd-empty">The library is empty.</div>
          ) : null}
        </section>
      )}
    </div>
  );
}
