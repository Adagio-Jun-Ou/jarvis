import React from 'react';
import { useResource } from '../lib/swr.js';
import { toast } from '../lib/toast.jsx';

// The "Prepared overnight" strip: today's morning brief plus any unconsumed
// standing-order briefs (with their suggested actions). Dismissing a brief
// calls /api/briefs/consume so it stops surfacing — the terminal hop of the
// standing-order lifecycle.
export default function BriefsStrip({ session }) {
  const authHeaders = () => session?.authHeaders?.() || {};
  const { data, retry } = useResource(
    session?.email ? `briefs:${session.email}` : null,
    (signal) => fetch('/api/briefs', { signal, credentials: 'include', headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`)))),
  );
  const morning = data?.brief || null;
  const briefs = data?.standing_orders || [];
  if (!morning?.brief && !briefs.length) return null;

  const consume = async (briefId) => {
    try {
      const r = await fetch('/api/briefs/consume', {
        method: 'POST',
        credentials: 'include',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ brief_id: briefId }),
      });
      if (!r.ok) throw new Error(`${r.status}`);
    } catch { toast('Could not dismiss brief'); return; }
    retry();
  };

  return (
    <div className="briefs-strip" aria-label="Prepared overnight">
      <span className="section-label">Prepared overnight</span>
      {morning?.brief ? <div className="brief-card morning"><p>{morning.brief}</p></div> : null}
      {briefs.map((b) => (
        <div className="brief-card" key={b.brief_id}>
          <p>{b.compiled_context}</p>
          {(b.suggested_actions || []).length ? (
            <ul className="brief-actions">
              {b.suggested_actions.map((a, i) => (
                <li key={i}>{a.action}{a.capability ? ` · ${a.capability}` : ''}</li>
              ))}
            </ul>
          ) : null}
          <button type="button" className="btn ghost brief-dismiss" onClick={() => consume(b.brief_id)}>Dismiss</button>
        </div>
      ))}
    </div>
  );
}
