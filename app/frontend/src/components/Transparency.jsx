import React, { useEffect, useState } from 'react';
import { toast } from '../lib/toast.jsx';

const TABS = [
  { id: 'memory', label: 'Memories' },
  { id: 'graph', label: 'Knowledge' },
  { id: 'capabilities', label: 'Capabilities' },
  { id: 'actions', label: 'Actions' },
  { id: 'library', label: 'Library' },
];

export default function Transparency({ me, onClose, session }) {
  const [tab, setTab] = useState('memory');
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const authHeaders = () => session?.authHeaders?.() || {};

  useEffect(() => {
    let cancel = false;
    setLoading(true);
    setErr('');
    fetch(`/api/transparency/${tab}`, { credentials: 'include', headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((d) => { if (!cancel) setData(d); })
      .catch((e) => { if (!cancel) setErr(e.message); })
      .finally(() => { if (!cancel) setLoading(false); });
    return () => { cancel = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, session]);

  const forget = (item_id) =>
    fetch('/api/transparency/forget', {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ item_id, reason: 'user_requested' }),
    })
      .then((r) => {
        // Forgetting a memory is a trust-critical control: removing it
        // from the UI while the server kept it would tell the user
        // their data is gone when it is not. Only remove on success.
        if (!r.ok) {
          toast('Could not forget that item — it is still stored.');
          return;
        }
        setData((d) => ({
          ...d,
          memories: (d.memories || []).filter((m) => m.id !== item_id),
        }));
      })
      .catch(() => toast('Could not reach the server — nothing was forgotten.'));

  return (
    <div className="tx-backdrop" onClick={onClose}>
      <div className="tx-modal" role="dialog" aria-label="What my AI knows"
           onClick={(e) => e.stopPropagation()}>
        <header className="tx-head">
          <div>
            <h2 className="tx-title">What my AI knows</h2>
            <div className="tx-me">Signed in as <code>{me}</code></div>
          </div>
          <button className="tx-close" onClick={onClose} aria-label="Close">×</button>
        </header>
        <nav className="tx-tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={tab === t.id ? 'tx-tab is-active' : 'tx-tab'}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          ))}
        </nav>
        <div className="tx-body">
          {loading ? <div className="tx-loading">Loading…</div>
            : err ? <div className="tx-error">Could not load this view ({err}).</div>
            : tab === 'memory' ? <Memories data={data} onForget={forget} />
            : tab === 'graph' ? <Graph data={data} />
            : tab === 'capabilities' ? <Capabilities data={data} />
            : tab === 'actions' ? <Actions data={data} />
            : <Library data={data} />}
        </div>
      </div>
    </div>
  );
}

function Empty({ children }) {
  return <div className="tx-empty">{children}</div>;
}

export function Memories({ data, onForget }) {
  if (!data.memories?.length) return <Empty>No memories yet.</Empty>;
  return (
    <ul className="tx-list">
      {data.memories.map((m, i) => (
        <li className="tx-row" key={i}>
          <span className="tx-tag">{m.kind}</span>
          <span className="tx-main">{m.content}</span>
          <button className="tx-forget" onClick={() => onForget(m.id)}>Forget</button>
        </li>
      ))}
    </ul>
  );
}

function Graph({ data }) {
  if (!data.edges?.length) return <Empty>No edges yet.</Empty>;
  return (
    <ul className="tx-list">
      {data.edges.map((e, i) => (
        <li className="tx-row" key={i}>
          <span className="tx-main tx-edge">
            <code>{e.src}</code>
            <span className="tx-rel">{e.rel}</span>
            <code>{e.dst}</code>
          </span>
          <span className="tx-conf" title="confidence">
            {Number(e.confidence ?? 0).toFixed(2)}
          </span>
        </li>
      ))}
    </ul>
  );
}

function Capabilities({ data }) {
  const empty = (!data.effective?.length) && (!data.delegations?.length);
  if (empty) return <Empty>No capabilities.</Empty>;
  return (
    <div className="tx-sections">
      <section>
        <h4 className="tx-subhead">What you can do</h4>
        <ul className="tx-list">
          {(data.effective || []).map((c, i) => (
            <li className="tx-row" key={i}>
              <code className="tx-main">{c.capability}</code>
              <span className="tx-meta">
                {c.expires_at ? `until ${c.expires_at}` : 'no expiry'}
              </span>
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h4 className="tx-subhead">Granted to you by someone else</h4>
        <ul className="tx-list">
          {(data.delegations || []).map((c, i) => (
            <li className="tx-row" key={i}>
              <code className="tx-main">{c.capability}</code>
              <span className="tx-meta">from <code>{c.supervisor_email}</code></span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

export function Actions({ data }) {
  if (!data.actions?.length) return <Empty>No actions logged yet.</Empty>;
  return (
    <ul className="tx-list">
      {data.actions.map((a, i) => {
        const outcome = a.outcome || 'ok';
        return (
          <li className={`tx-row tx-action is-${outcome}`} key={i}>
            <span className="tx-ts">{a.ts}</span>
            <span className="tx-tool">{a.tool_name || a.action_type}</span>
            {outcome !== 'ok'
              ? <span className={`tx-outcome is-${outcome}`}>{outcome}</span>
              : null}
            <span className="tx-main tx-detail">{a.detail}</span>
          </li>
        );
      })}
    </ul>
  );
}

function Library({ data }) {
  if (!data.library?.length) return <Empty>No library items yet.</Empty>;
  return (
    <ul className="tx-list">
      {data.library.map((f, i) => (
        <li className="tx-row" key={i}>
          <code className="tx-main">{f.path}</code>
          <span className="tx-meta">via {f.source_channel}</span>
        </li>
      ))}
    </ul>
  );
}
