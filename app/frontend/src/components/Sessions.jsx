import React from 'react';
import { useResource } from '../lib/swr.js';
import { toast } from '../lib/toast.jsx';
import { relTime } from '../lib/format.js';
import { IconPlus, IconSessions } from './icons.jsx';

export default function Sessions({ session, onOpen, onCreated }) {
  const authHeaders = () => session?.authHeaders?.() || {};
  const { data, error, loading, retry } = useResource(
    session?.email ? `sessions:${session.email}` : null,
    (signal) => fetch('/api/sessions', { signal, credentials: 'include', headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`)))),
  );
  const sessions = data?.sessions || [];

  const create = async () => {
    try {
      const r = await fetch('/api/sessions', {
        method: 'POST', credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ title: 'New session' }),
      });
      if (!r.ok) throw new Error(`${r.status}`);
      const { session_id } = await r.json();
      onCreated?.();
      retry();
      onOpen?.(session_id);
    } catch { toast('Could not create session'); }
  };

  const rename = async (s) => {
    const title = window.prompt('Rename session', s.title || '');
    if (!title || !title.trim()) return;
    try {
      const r = await fetch(`/api/sessions/${encodeURIComponent(s.session_id)}`, {
        method: 'PATCH', credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ title: title.trim() }),
      });
      if (!r.ok) throw new Error(`${r.status}`);
      retry();
    } catch { toast('Could not rename session'); }
  };

  const remove = async (s) => {
    if (!window.confirm(`Delete "${s.title || 'this session'}"? This cannot be undone.`)) return;
    try {
      const r = await fetch(`/api/sessions/${encodeURIComponent(s.session_id)}`, {
        method: 'DELETE', credentials: 'include', headers: authHeaders(),
      });
      if (!r.ok) throw new Error(`${r.status}`);
      retry();
    } catch { toast('Could not delete session'); }
  };

  return (
    <div className="page">
      <div className="page-head">
        <h2>Sessions</h2>
        <p>Every conversation with Alex, scoped and resumable. {sessions.length ? `${sessions.length} total.` : ''}</p>
      </div>

      {loading && !sessions.length ? (
        <div className="sessions-grid">
          {Array.from({ length: 6 }).map((_, i) => (
            <div className="card session-card" key={i}>
              <div className="skeleton skeleton-row" style={{ width: '70%' }} />
              <div className="skeleton skeleton-row" style={{ width: '40%' }} />
            </div>
          ))}
        </div>
      ) : error ? (
        <div className="error-inline">Couldn’t load sessions. <button className="btn ghost" onClick={retry}>Retry</button></div>
      ) : (
        <div className="sessions-grid">
          <button className="card session-card session-new" onClick={create}>
            <span className="plus"><IconPlus /></span>
            <b>New session</b>
          </button>
          {sessions.map((s) => (
            <div className="card session-card" key={s.session_id} role="button" tabIndex={0}
              onClick={() => onOpen?.(s.session_id)}
              onKeyDown={(e) => { if (e.key === 'Enter') onOpen?.(s.session_id); }}>
              <div className="session-actions">
                <button type="button" className="icon-btn-sm" title="Rename"
                  onClick={(e) => { e.stopPropagation(); rename(s); }}>✎</button>
                <button type="button" className="icon-btn-sm" title="Delete"
                  onClick={(e) => { e.stopPropagation(); remove(s); }}>🗑</button>
              </div>
              <h4>{s.title || 'Untitled session'}</h4>
              <div className="meta">
                {s.model ? <span className="chip">{s.model}</span> : <span className="chip">Alex VM</span>}
                <span>{relTime(s.last_active_at || s.created_at)}</span>
              </div>
            </div>
          ))}
          {!sessions.length && (
            <div className="empty-state" style={{ gridColumn: '1 / -1' }}>
              <div className="glyph"><IconSessions width={26} height={26} /></div>
              <h3>No sessions yet</h3>
              <p>Start your first conversation with Alex — pick up right where you leave off, any time.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
