import React, { useCallback, useEffect, useState } from 'react';
import { IconPlus, IconClose } from './icons.jsx';
import { toast } from '../lib/toast.jsx';

// The agent's VM browser. Tabs, an address/omnibar, and a framed
// viewport. Fixture pages render as NATIVE dark panels (never a raw
// white iframe). Form submits use the governed two-phase confirm.
export default function Browser({ session }) {
  const authHeaders = () => session?.authHeaders?.() || {};
  const api = useCallback(
    async (path, method = 'GET', body) => {
      const r = await fetch(path, {
        method,
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: body ? JSON.stringify(body) : undefined,
      });
      return { ok: r.ok, status: r.status, data: await r.json().catch(() => ({})) };
    },
    [session] // eslint-disable-line react-hooks/exhaustive-deps
  );

  const [sid, setSid] = useState(null);
  const [tabs, setTabs] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [url, setUrl] = useState('');
  const [extract, setExtract] = useState(null);

  const refresh = useCallback(
    async (s) => {
      if (!s) return;
      const r = await api(`/api/browser/sessions/${s}/tabs`);
      if (r.ok) {
        setTabs(r.data.tabs || []);
        setActiveId((cur) => cur || r.data.tabs?.[0]?.tab_id || null);
      }
    },
    [api]
  );

  useEffect(() => {
    (async () => {
      const r = await api('/api/browser/sessions', 'POST');
      if (r.ok) {
        setSid(r.data.session_id);
        await refresh(r.data.session_id);
      }
    })();
  }, [api, refresh]);

  const active = tabs.find((t) => t.tab_id === activeId) || null;

  const newTab = async () => {
    if (!sid) return;
    const r = await api(`/api/browser/sessions/${sid}/tabs?url=alex://welcome`, 'POST');
    if (r.ok) { setActiveId(r.data.tab_id); await refresh(sid); }
  };
  const closeTab = async (tid) => {
    if (!sid) return;
    await api(`/api/browser/sessions/${sid}/tabs/${tid}`, 'DELETE');
    if (activeId === tid) setActiveId(null);
    await refresh(sid);
  };
  const navigate = async () => {
    if (!sid || !activeId || !url) return;
    await api(`/api/browser/sessions/${sid}/tabs/${activeId}/navigate`, 'POST', { url });
    await refresh(sid);
  };
  const doExtract = async () => {
    if (!sid || !activeId) return;
    const r = await api(`/api/browser/sessions/${sid}/tabs/${activeId}/extract`, 'POST', {});
    if (r.ok) setExtract(r.data);
  };
  const doSubmit = async () => {
    if (!sid || !activeId) return;
    const r = await api(`/api/browser/sessions/${sid}/tabs/${activeId}/submit`, 'POST', {
      selector: 'form',
    });
    if (r.data?.awaiting_confirmation) {
      toast('Alex staged a form submit — approve it in the agent activity rail.');
    } else if (r.data?.ok) {
      toast('Submitted.');
    } else {
      toast(r.data?.error || 'Nothing to submit.');
    }
  };
  const toggleMirror = async () => {
    if (!sid || !activeId || !active) return;
    await api(`/api/browser/sessions/${sid}/tabs/${activeId}/mirror`, 'POST', {
      mode: active.agent_piloting ? 'co-drive' : 'yield',
    });
    await refresh(sid);
  };

  return (
    <div className="browser">
      <div className="tabstrip" role="tablist" aria-label="Browser tabs">
        {tabs.map((t) => (
          <div
            key={t.tab_id}
            role="tab"
            aria-selected={t.tab_id === activeId}
            className={
              'tabstrip-tab' +
              (t.tab_id === activeId ? ' active' : '') +
              (t.agent_piloting ? '' : ' user-controlled')
            }
            onClick={() => setActiveId(t.tab_id)}
          >
            <span className="pilot-dot" />
            <span className="t-title">{t.title || t.url || 'New tab'}</span>
            <button
              aria-label="Close tab"
              onClick={(e) => { e.stopPropagation(); closeTab(t.tab_id); }}
            >
              <IconClose width={13} height={13} />
            </button>
          </div>
        ))}
        <button className="tab-new" onClick={newTab} aria-label="New tab">
          <IconPlus width={15} height={15} />
        </button>
      </div>

      <div className="omnibar">
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && navigate()}
          placeholder="https://  or  alex://welcome"
          aria-label="Address bar"
        />
        <button className="btn" onClick={navigate}>Go</button>
        <button className="btn ghost" onClick={doExtract}>Extract</button>
        <button className="btn ghost" onClick={doSubmit}>Submit</button>
      </div>

      {active && (
        <div className={'pilot-banner ' + (active.agent_piloting ? 'agent' : 'user')}>
          <span className="pilot-dot" />
          <span>{active.agent_piloting ? 'Alex is piloting this tab.' : 'You have the wheel.'}</span>
          <button className="btn ghost" style={{ marginLeft: 'auto' }} onClick={toggleMirror}>
            {active.agent_piloting ? 'Take the wheel' : 'Yield to Alex'}
          </button>
        </div>
      )}

      <div className="viewport">
        {!active ? (
          <div className="empty-state">
            <span className="glyph">◫</span>
            <h3>No tab open</h3>
            <p>Press ＋ to open a tab, or ask Alex to browse for you.</p>
          </div>
        ) : (
          <div className="viewport-page">
            <h1>{active.title || active.url}</h1>
            <p className="muted">
              {(active.url || '').startsWith('alex://')
                ? 'Alex fixture page (dev backend).'
                : 'External page (production Chromium in the VM).'}
            </p>
            <p className="faint">URL: <code>{active.url}</code></p>
            {extract && (
              <div className="card" style={{ marginTop: 16, padding: 14 }}>
                <span className="section-label">Extracted text</span>
                {extract.pii_redacted && <span className="pill" style={{ marginLeft: 8 }}>PII redacted</span>}
                <p className="muted" style={{ marginTop: 8, whiteSpace: 'pre-wrap' }}>
                  {String(extract.text || '').slice(0, 1200)}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
