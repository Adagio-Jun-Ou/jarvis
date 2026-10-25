import React, { useEffect, useState } from 'react';
import { useResource } from '../lib/swr.js';
import { toast } from '../lib/toast.jsx';
import { relTime } from '../lib/format.js';
import { IconEmail, IconSpark } from './icons.jsx';

const TONES = ['professional', 'warm', 'brief'];

export default function Email({ session }) {
  const authHeaders = () => session?.authHeaders?.() || {};
  const { data, loading, error, retry } = useResource(
    session?.email ? `inbox:${session.email}` : null,
    (signal) => fetch('/api/inbox', { signal, credentials: 'include', headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`)))),
  );
  const inbox = data?.inbox || [];
  const [selectedId, setSelectedId] = useState(null);

  return (
    <div className={`email${selectedId ? ' has-selection' : ''}`}>
      <div className="email-list">
        <div className="email-list-head"><h2>Inbox</h2></div>
        {loading && !inbox.length ? (
          Array.from({ length: 6 }).map((_, i) => (
            <div className="email-row" key={i}>
              <div className="skeleton skeleton-row" style={{ width: '50%' }} />
              <div className="skeleton skeleton-row" style={{ width: '80%' }} />
            </div>
          ))
        ) : error ? (
          <div className="error-inline" style={{ margin: 16 }}>Couldn’t load inbox. <button className="btn ghost" onClick={retry}>Retry</button></div>
        ) : !inbox.length ? (
          <div className="empty-state"><div className="glyph"><IconEmail width={24} height={24} /></div><h3>Inbox zero</h3><p>Nothing needs your attention right now.</p></div>
        ) : (
          inbox.map((m) => (
            <button
              key={m.email_id}
              className={`email-row${selectedId === m.email_id ? ' active' : ''}`}
              onClick={() => setSelectedId(m.email_id)}
            >
              <div className="top">
                <span className={`prio-dot prio-${m.priority ?? 3}`} />
                <span className="sender">{m.from_email || m.sender || m.subject || 'Message'}</span>
                <span className="time">{relTime(m.received_at)}</span>
              </div>
              <div className="subject">{m.subject || '(no subject)'}</div>
              {m.summary && <div className="summary">{m.summary}</div>}
            </button>
          ))
        )}
      </div>
      <EmailDetail key={selectedId} emailId={selectedId} session={session} onBack={() => setSelectedId(null)} />
    </div>
  );
}

function EmailDetail({ emailId, session, onBack }) {
  const authHeaders = () => session?.authHeaders?.() || {};
  const [email, setEmail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tone, setTone] = useState('professional');
  const [draft, setDraft] = useState('');
  const [drafting, setDrafting] = useState(false);

  useEffect(() => {
    if (!emailId) { setEmail(null); return; }
    let cancel = false;
    setLoading(true); setDraft('');
    fetch(`/api/email/${encodeURIComponent(emailId)}`, { credentials: 'include', headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((d) => { if (!cancel) setEmail(d.email); })
      .catch(() => { if (!cancel) toast('Could not open email'); })
      .finally(() => { if (!cancel) setLoading(false); });
    return () => { cancel = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [emailId]);

  const makeDraft = async (t = tone) => {
    if (!emailId) return;
    setDrafting(true);
    try {
      const r = await fetch('/api/email/draft', {
        method: 'POST', credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ email_id: emailId, instruction: '', tone: t }),
      });
      if (!r.ok) throw new Error(`${r.status}`);
      const d = await r.json();
      setDraft(d.draft || '');
    } catch { toast('Draft failed'); } finally { setDrafting(false); }
  };

  const copy = () => { navigator.clipboard?.writeText(draft); toast('Copied to clipboard', 'success'); };

  if (!emailId) {
    return (
      <div className="email-detail">
        <div className="empty-state">
          <div className="glyph"><IconEmail width={26} height={26} /></div>
          <h3>Select a message</h3>
          <p>Pick an email to read it, get an AI summary, and draft a reply in one click.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="email-detail">
      <button className="btn ghost email-back" onClick={onBack}>← Inbox</button>
      {loading || !email ? (
        <><div className="skeleton skeleton-row" style={{ width: '60%', height: 22 }} /><div className="skeleton skeleton-row" style={{ width: '90%' }} /><div className="skeleton skeleton-row" style={{ width: '80%' }} /></>
      ) : (
        <>
          <div className="email-detail-head">
            <h1>{email.subject || '(no subject)'}</h1>
            <div className="from">{email.from_email || 'Unknown sender'} · {relTime(email.received_at)}</div>
          </div>
          {email.summary && (
            <div className="email-summary">
              <div className="section-label">AI summary</div>
              {email.summary}
            </div>
          )}
          <div className="email-body">{email.body || email.summary || 'No body available for this message.'}</div>

          <div className="draft-box">
            <div className="section-label">Reply</div>
            <div className="tones">
              <button className="btn primary" onClick={() => makeDraft()} disabled={drafting}>
                <IconSpark width={14} height={14} /> {drafting ? 'Drafting…' : 'AI Draft Reply'}
              </button>
              {TONES.map((tt) => (
                <button key={tt} className={`pill${tone === tt ? ' active' : ''}`} onClick={() => { setTone(tt); if (draft) makeDraft(tt); }}>{tt}</button>
              ))}
            </div>
            {(draft || drafting) && (
              <>
                <textarea className="draft-editor" value={draft} placeholder={drafting ? 'Alex is writing…' : ''} onChange={(e) => setDraft(e.target.value)} />
                <div className="draft-actions">
                  <button className="btn" onClick={copy} disabled={!draft}>Copy</button>
                  <button className="btn" onClick={() => makeDraft()} disabled={drafting}>Regenerate</button>
                </div>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
