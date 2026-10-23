import React, { useEffect, useRef, useState } from 'react';
import ReasoningPanel from './Reasoning.jsx';
import { createFrameBatcher } from '../lib/frameBatcher.js';
import { toast } from '../lib/toast.jsx';
import { formatBytes } from '../lib/format.js';
import { renderMarkdown } from '../lib/markdown.jsx';
import { downloadArtifact } from '../lib/download.js';
import { startRecording, transcribe, speak, stopSpeaking } from '../lib/voice.js';
import ExecutionLog from './ExecutionLog.jsx';
import NowLine from './NowLine.jsx';
import Presence from './Presence.jsx';
import SwarmPanel from './Swarm.jsx';
import { IconSpark, IconPaperclip, IconDownload } from './icons.jsx';

const WINDOW_SIZE = 30;

/* Inline activity timeline: non-text events render as compact chips in
   transcript flow. Exported for tests. */
export function ActivityTimeline({ timeline, onRetry, authHeaders }) {
  if (!timeline?.length) return null;
  // One renderer for the turn's work: the execution log. This wrapper
  // stays for callers (and tests) that predate it.
  return <ExecutionLog timeline={timeline} onRetry={onRetry} authHeaders={authHeaders} />;
}

/* Distinct inline card for SSE 'error' events. Exported for tests. */
export function ErrorCard({ error, onRetry }) {
  return (
    <div className="error-card" role="alert">
      <div className="error-card-msg">
        ⚠ {error.message || 'Something went wrong.'}
        {error.code ? <span className="error-card-code"> ({error.code})</span> : null}
      </div>
      {onRetry && <button className="error-retry" onClick={onRetry}>Retry</button>}
    </div>
  );
}

/* Elapsed-time "thinking…" indicator. Exported for tests. */
export function WaitingIndicator({ since, warmupAfter = 10 }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const secs = Math.max(0, Math.floor((now - since) / 1000));
  return (
    <span className="thinking waiting" data-testid="waiting">
      <i /><i /><i /> thinking… {secs}s
      {secs > warmupAfter && <span className="warmup-note"> · local model warming up — hang tight</span>}
    </span>
  );
}

/* Hover actions on a settled turn: copy always; regenerate on AI turns
   (re-runs the question that produced it); edit on user turns (loads
   the text back into the composer). The state-of-the-art chat surfaces
   all converge on exactly these three. */
export function TurnActions({ turn, onCopy, onRegenerate, onEdit }) {
  return (
    <div className="turn-actions" role="toolbar" aria-label="Message actions">
      <button type="button" className="turn-act" onClick={onCopy} title="Copy">
        Copy
      </button>
      {turn.role === 'ai' && onRegenerate ? (
        <button type="button" className="turn-act" onClick={onRegenerate}
                title="Ask this again">
          Regenerate
        </button>
      ) : null}
      {turn.role === 'user' && onEdit ? (
        <button type="button" className="turn-act" onClick={onEdit}
                title="Edit and resend">
          Edit
        </button>
      ) : null}
    </div>
  );
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text || '');
    toast('Copied');
  } catch {
    toast('Could not copy');
  }
}

/* Conversation starters for the empty state — each is a real capability
   the workspace ships, so the first click teaches what Alex can do. */
const STARTERS = [
  'Summarise my inbox and flag anything urgent',
  'What is on my task board this week?',
  'Draft a plan before doing anything: prepare the Q3 review',
  'What do you remember about me?',
];

const AUTONOMY_LABEL = {
  suggest: 'Ask first',
  'co-pilot': 'Co-pilot',
  autopilot: 'Autopilot',
};
const AUTONOMY_HINT = {
  suggest: 'Small budgets; Alex proposes more than it does.',
  'co-pilot': 'Balanced budgets — the default.',
  autopilot: 'Larger budgets; Alex runs longer before stopping.',
};

function ConfirmCard({ card, onDecide }) {
  return (
    <div className="confirm-card" role="group" aria-label="Confirmation needed">
      <h4>Your confirmation runs this — not the AI</h4>
      <div>{card.summary || `${card.integration || card.workflow || 'Action'} is staged.`}</div>
      {card.body && <pre className="payload">{JSON.stringify(card.body, null, 2)}</pre>}
      <div className="confirm-actions">
        <button className="btn primary" onClick={() => onDecide(true)}>Run it</button>
        <button className="btn" onClick={() => onDecide(false)}>Cancel</button>
      </div>
    </div>
  );
}

/* SSE line decoder — handles multibyte UTF-8 split across reads. */
function makeDecoder() {
  const td = new TextDecoder('utf-8');
  let buf = '';
  return (chunk) => {
    buf += td.decode(chunk, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() ?? '';
    return lines;
  };
}

/* Flatten a turn's streaming state into an ordered list of rail items
   for the shell's Agent activity rail. */
export function railItemsFor(turn) {
  if (!turn) return [];
  const out = [];
  for (const r of turn.reasoning || []) out.push({ kind: 'reasoning', text: r.text });
  for (const item of turn.timeline || []) if (item.kind !== 'error') out.push(item);
  return out;
}

export default function Chat({ sessionId, model, session, browserSessionId, activeTabId, onActivity, onBusy }) {
  const authHeaders = () => session?.authHeaders?.() || {};
  const [turns, setTurns] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [speakReplies, setSpeakReplies] = useState(false);
  const [recording, setRecording] = useState(null);  // { stop } while the mic is live
  const speakRef = useRef(false);
  speakRef.current = speakReplies;
  // Plan mode: read + plan, change nothing. Enforced structurally on
  // the backend (mutating tools are not exposed to the model at all).
  const [planMode, setPlanMode] = useState(false);
  const [attachments, setAttachments] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [showAll, setShowAll] = useState(false);
  // Autonomy is a REAL control (the governor scales its budgets from
  // it), stored server-side per user; the composer is where the person
  // deciding how much rope to give the agent actually is.
  const [autonomy, setAutonomyState] = useState('co-pilot');
  const [showJump, setShowJump] = useState(false);
  const scroller = useRef(null);
  const textareaRef = useRef(null);
  const fileInput = useRef(null);
  const abortRef = useRef(null);
  const lastSentRef = useRef('');

  // Session history: load this session's saved turns when it changes,
  // and persist completed turns so switching sessions (and reloads)
  // restore the conversation. Stored client-side, keyed by session id.
  useEffect(() => {
    setShowAll(false);
    let saved = [];
    try {
      const raw = localStorage.getItem(`alex-chat:${sessionId}`);
      if (raw) saved = JSON.parse(raw);
    } catch { /* ignore */ }
    if (Array.isArray(saved) && saved.length) {
      setTurns(saved);
      return;
    }
    // No local copy (e.g. a fresh device/browser) — pull the persisted
    // conversation from the server so history is cross-device.
    setTurns([]);
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`/api/sessions/${sessionId}/messages`, { headers: authHeaders() });
        if (!r.ok) return;
        const d = await r.json();
        if (cancelled || !Array.isArray(d.messages) || !d.messages.length) return;
        setTurns(d.messages.map((m) => ({
          role: m.role === 'assistant' ? 'ai' : 'user',
          content: m.content || '',
          timeline: [], activity: [], reasoning: [],
        })));
      } catch { /* offline or no server history */ }
    })();
    return () => { cancelled = true; };
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (busy) return; // don't persist mid-stream
    try {
      // Persist only settled fields (drop transient streaming state).
      const clean = turns
        .filter((t) => !t.pending && !t.streaming)
        .map((t) => ({
          role: t.role,
          content: t.content,
          timeline: t.timeline,
          activity: t.activity,
          reasoning: t.reasoning,
        }));
      if (clean.length) localStorage.setItem(`alex-chat:${sessionId}`, JSON.stringify(clean.slice(-100)));
    } catch { /* ignore quota */ }
  }, [turns, busy, sessionId]);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    // Autoscroll only when the user is already near the bottom, so
    // reading earlier history isn't yanked away by a new token.
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (nearBottom || busy) el.scrollTo?.({ top: el.scrollHeight, behavior: 'smooth' });
  }, [turns, busy]);
  useEffect(() => () => abortRef.current?.abort(), []);
  // Switching sessions aborts the in-flight stream: otherwise the old
  // session's events keep applying into the new transcript (or worse,
  // pad the turns array with holes and white-screen the app).
  useEffect(() => { abortRef.current?.abort(); setBusy(false); }, [sessionId]);
  useEffect(() => { onBusy?.(busy); }, [busy, onBusy]);

  // Surface the current (last) AI turn's activity to the shell rail.
  useEffect(() => {
    const last = [...turns].reverse().find((t) => t.role === 'ai');
    onActivity?.(railItemsFor(last));
  }, [turns, onActivity]);

  const stop = () => abortRef.current?.abort();

  // Load the user's autonomy preference once; persist changes through
  // the same preferences API the settings panel uses.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch('/api/preferences', { headers: authHeaders() });
        if (!r.ok || cancelled) return;
        const d = await r.json();
        const a = d?.preferences?.autonomy;
        if (a) setAutonomyState(a);
      } catch { /* default stands */ }
    })();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const setAutonomy = async (value) => {
    setAutonomyState(value);
    try {
      await fetch('/api/preferences', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ autonomy: value }),
      });
    } catch {
      toast('Could not save the autonomy setting');
    }
  };

  // Scroll pill: visible whenever the user has scrolled up away from
  // the live tail, so streaming never yanks them and returning is one
  // click.
  useEffect(() => {
    const el = scroller.current;
    if (!el) return undefined;
    const onScroll = () => {
      setShowJump(el.scrollHeight - el.scrollTop - el.clientHeight > 240);
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  const jumpToLatest = () => {
    scroller.current?.scrollTo?.({
      top: scroller.current.scrollHeight, behavior: 'smooth',
    });
  };

  // Auto-grow the composer with its content, capped so it never eats
  // the transcript.
  const autoGrow = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  };
  useEffect(autoGrow, [input]);

  /** The user message that produced the AI turn at index i. */
  const questionFor = (i) => {
    for (let j = i - 1; j >= 0; j--) {
      if (turns[j]?.role === 'user') return turns[j].content;
    }
    return '';
  };

  const toggleMic = async () => {
    if (recording) {
      const rec = recording;
      setRecording(null);
      try {
        const blob = await rec.stop();
        const text = await transcribe(blob, authHeaders);
        if (text) setInput((cur) => (cur ? `${cur} ${text}` : text));
        textareaRef.current?.focus();
      } catch (e) {
        toast(e.message || 'Could not transcribe.');
      }
      return;
    }
    try {
      stopSpeaking();
      setRecording(await startRecording());
    } catch (e) {
      toast(e.message || 'Microphone not available.');
    }
  };

  const send = async (text) => {
    const content = (text ?? input).trim();
    if (!content || busy || !sessionId) return;
    if (text == null) setInput('');
    lastSentRef.current = content;
    setBusy(true);
    const turnIdx = turns.length;
    setTurns((t) => [
      ...t,
      { role: 'user', content },
      { role: 'ai', content: '', activity: [], reasoning: [], lawBlocks: [], artifacts: [],
        timeline: [], streaming: true, card: null, startedAt: Date.now(), gotFirstEvent: false },
    ]);

    const ac = new AbortController();
    abortRef.current = ac;

    const batcher = createFrameBatcher((events) => {
      setTurns((t) => {
        let out = t;
        for (const ev of events) out = applyEvent(out, turnIdx + 1, ev);
        return out;
      });
    });

    const finish = (mutate) => {
      batcher.flushNow();
      setTurns((t) => updateTurn(t, turnIdx + 1, (x) => {
        // Spoken replies: the finished answer goes to the server's TTS.
        // Best-effort and off the render path; silence is never faked.
        if (speakRef.current && x.content) {
          speak(x.content, authHeaders).catch(() => {});
        }
        return {
          ...x, streaming: false, gotFirstEvent: true, endedAt: Date.now(),
          ...(mutate ? mutate(x) : null),
        };
      }));
      if (abortRef.current === ac) abortRef.current = null;
      setBusy(false);
    };

    let res;
    try {
      res = await fetch('/api/chat', {
        method: 'POST', credentials: 'include', signal: ac.signal,
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ session_id: sessionId, model, show_reasoning: showReasoning, plan_mode: planMode, text: content }),
      });
    } catch (e) {
      if (e?.name === 'AbortError') finish((x) => ({ content: x.content || 'Stopped.' }));
      else finish((x) => ({ timeline: [...(x.timeline || []), { kind: 'error', message: `Network error: ${e.message}` }] }));
      return;
    }
    if (!res.ok || !res.body) {
      finish((x) => ({ timeline: [...(x.timeline || []), { kind: 'error', message: `Agent error: ${res.status} ${res.statusText}` }] }));
      return;
    }

    const reader = res.body.getReader();
    const nextLine = makeDecoder();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of nextLine(value)) {
          if (!line.startsWith('data: ')) continue;
          const chunk = line.slice(6);
          if (chunk === '[DONE]') continue;
          let ev;
          try { ev = JSON.parse(chunk); } catch { continue; }
          batcher.push(ev);
        }
      }
    } catch (e) {
      if (e?.name === 'AbortError') finish((x) => ({ content: x.content || 'Stopped.' }));
      else finish((x) => ({ timeline: [...(x.timeline || []), { kind: 'error', message: String(e.message || e) }] }));
      return;
    }
    finish();
  };

  const retry = () => { if (lastSentRef.current) send(lastSentRef.current); };

  const decideCard = async (turnIdx, yes) => {
    const card = turns[turnIdx]?.card;
    if (!card) return;
    setTurns((t) => updateTurn(t, turnIdx, (x) => ({ ...x, card: null })));
    if (yes) {
      const sid = card.session_id || card.body?.session_id || browserSessionId;
      const tid = card.tab_id || card.body?.tab_id || activeTabId;
      if (!sid || !tid) return;
      await fetch(`/api/browser/sessions/${sid}/tabs/${tid}/submit`, {
        method: 'POST', credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ selector: card.body?.selector || 'form', confirmation_token: card.token }),
      });
    }
  };

  // Upload adds prominent chips above the composer; files POST to the
  // uploads endpoint (field "file").
  const addFiles = async (fileList) => {
    const files = Array.from(fileList || []);
    for (const f of files) {
      const id = `${f.name}-${f.size}-${Math.random().toString(36).slice(2, 7)}`;
      setAttachments((a) => [...a, { id, name: f.name, size: f.size, status: 'uploading' }]);
      try {
        const fd = new FormData();
        fd.append('file', f);
        const r = await fetch('/api/files/upload', { method: 'POST', credentials: 'include', headers: authHeaders(), body: fd });
        if (!r.ok) throw new Error(`${r.status}`);
        setAttachments((a) => a.map((x) => (x.id === id ? { ...x, status: 'ready' } : x)));
      } catch {
        toast(`Upload failed: ${f.name}`);
        setAttachments((a) => a.filter((x) => x.id !== id));
      }
    }
  };
  const removeAttachment = (id) => setAttachments((a) => a.filter((x) => x.id !== id));

  const hiddenCount = showAll ? 0 : Math.max(0, turns.length - WINDOW_SIZE);
  const visibleTurns = hiddenCount ? turns.slice(hiddenCount) : turns;

  return (
    <div className="chat"
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false); }}
      onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}
    >
      {dragging && <div className="drop-overlay">Drop files to attach</div>}
      <div className="chat-scroll" ref={scroller}>
        <div className="chat-inner">
          {!turns.length && (
            <div className="empty-state">
              <div className="glyph"><IconSpark width={26} height={26} /></div>
              <h3>How can I help today?</h3>
              <p>Ask Alex anything, or drop a file into the conversation to work on it together.</p>
              <div className="starters" aria-label="Suggestions">
                {STARTERS.map((s) => (
                  <button key={s} type="button" className="starter-chip"
                          onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {hiddenCount > 0 && (
            <button className="btn ghost" onClick={() => setShowAll(true)}>
              Show earlier ({hiddenCount} message{hiddenCount > 1 ? 's' : ''})
            </button>
          )}
          {visibleTurns.map((t, vi) => {
            const i = vi + hiddenCount;
            return t.role === 'user' ? (
              <div className="turn user" key={i}>
                <div className="bubble-user">{t.content}</div>
                <TurnActions
                  turn={t}
                  onCopy={() => copyText(t.content)}
                  onEdit={() => { setInput(t.content); textareaRef.current?.focus(); }}
                />
              </div>
            ) : (
              <div className="turn" key={i}>
                <div className="ai-avatar"><Presence turn={t} /></div>
                <div className="ai-body">
                  {t.reasoning?.length ? <ReasoningPanel events={t.reasoning} /> : null}
                  <ExecutionLog
                    timeline={t.timeline}
                    onRetry={retry}
                    authHeaders={authHeaders}
                    streaming={!!t.streaming}
                    startedAt={t.startedAt}
                  />
                  {t.swarm && <SwarmPanel swarm={t.swarm} />}
                  {renderMarkdown(t.content)}
                  {t.card && <ConfirmCard card={t.card} onDecide={(y) => decideCard(i, y)} />}
                  <div className="ai-meta">
                    {t.streaming
                      ? !t.gotFirstEvent && t.startedAt
                        ? <WaitingIndicator since={t.startedAt} />
                        : <NowLine turn={t} />
                      : (
                        <>
                          {t.activity?.length ? (
                            <span className="faint">
                              {t.activity.length} governed tool call{t.activity.length > 1 ? 's' : ''}
                              {t.endedAt && t.startedAt
                                ? ` · ${Math.max(1, Math.round((t.endedAt - t.startedAt) / 1000))}s`
                                : ''}
                            </span>
                          ) : null}
                          {t.content ? (
                            <TurnActions
                              turn={t}
                              onCopy={() => copyText(t.content)}
                              onRegenerate={() => {
                                const q = questionFor(i);
                                if (q) send(q); else toast('Nothing to regenerate');
                              }}
                            />
                          ) : null}
                        </>
                      )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {showJump ? (
        <button type="button" className="jump-latest" onClick={jumpToLatest}
                aria-label="Jump to the latest message">
          ↓ Latest
        </button>
      ) : null}

      <div className="composer">
        <div className="composer-inner-wrap">
          {attachments.length > 0 && (
            <div className="upload-chips">
              {attachments.map((a) => (
                <div className="upload-chip" key={a.id}>
                  <IconPaperclip width={13} height={13} />
                  <span className="fname">{a.name}</span>
                  <span className="fsize">{a.status === 'uploading' ? 'uploading…' : formatBytes(a.size)}</span>
                  <button aria-label={`Remove ${a.name}`} onClick={() => removeAttachment(a.id)}>×</button>
                </div>
              ))}
            </div>
          )}
          <div className="composer-inner">
            <button className="icon-btn" title="Attach file" aria-label="Attach file" onClick={() => fileInput.current?.click()}>
              <IconPaperclip width={18} height={18} />
            </button>
            <input type="file" hidden multiple ref={fileInput} onChange={(e) => addFiles(e.target.files)} />
            <button
              className={'icon-btn mic-btn' + (recording ? ' live' : '')}
              title={recording ? 'Stop and transcribe' : 'Hold a thought: talk to Jarvis'}
              aria-label={recording ? 'Stop recording' : 'Start recording'}
              aria-pressed={!!recording}
              onClick={toggleMic}
            >
              {recording ? '■' : '🎙'}
            </button>
            <textarea
              rows={1}
              ref={textareaRef}
              placeholder="Ask Alex, or drop files to work on…  (Shift+Enter for a new line)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
            />
            {busy ? (
              <button className="send-btn stop" onClick={stop} aria-label="Stop">Stop</button>
            ) : (
              <button className="send-btn" disabled={!input.trim()} onClick={() => send()}>Send</button>
            )}
          </div>
          <div className="composer-hint">
            <div className="mode-seg" role="radiogroup" aria-label="Working mode"
                 title="Plan: Alex reads and drafts, and can change nothing.">
              <button type="button" role="radio" aria-checked={!planMode}
                      className={'mode-opt' + (!planMode ? ' active' : '')}
                      onClick={() => setPlanMode(false)}>
                Build
              </button>
              <button type="button" role="radio" aria-checked={planMode}
                      className={'mode-opt' + (planMode ? ' active' : '')}
                      onClick={() => setPlanMode(true)}>
                Plan
              </button>
            </div>
            <label className="autonomy-pick" title={AUTONOMY_HINT[autonomy] || ''}>
              <span className="autonomy-label">Autonomy</span>
              <select
                value={autonomy}
                onChange={(e) => setAutonomy(e.target.value)}
                aria-label="How much Alex may do before asking"
              >
                {Object.keys(AUTONOMY_LABEL).map((k) => (
                  <option key={k} value={k}>{AUTONOMY_LABEL[k]}</option>
                ))}
              </select>
            </label>
            <label className="reason-toggle">
              <input type="checkbox" checked={showReasoning} onChange={(e) => setShowReasoning(e.target.checked)} />
              Show thinking
            </label>
            <label className="reason-toggle" title="Read finished replies aloud">
              <input type="checkbox" checked={speakReplies}
                     onChange={(e) => { setSpeakReplies(e.target.checked); if (!e.target.checked) stopSpeaking(); }} />
              Speak replies
            </label>
            {planMode ? (
              <span className="plan-mode-note">· Plan mode: Alex can look and plan, not act.</span>
            ) : (
              <span className="hint-tail">· Every action is checked against the constitution and your approved capabilities.</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* SSE reducer — the ONLY place event types map to UI state. Exported
   so tests exercise the real production reducer. */
function toolKey(ev) {
  try { return `${ev.tool}:${JSON.stringify(ev.args ?? {})}`; } catch { return `${ev.tool}:?`; }
}

export function reduceTurn(t, ev) {
  const seen = t.gotFirstEvent ? t : { ...t, gotFirstEvent: true };
  const tl = seen.timeline || [];
  switch (ev.type) {
    case 'text':
      return { ...seen, content: seen.content + (ev.text || '') };
    case 'reasoning':
    case 'thinking':
      return { ...seen, reasoning: [...(seen.reasoning || []), ev] };
    case 'tool':
      // Steps are timed client-side: the user's perceived latency is
      // what the log should report, and it needs no backend clock.
      // Matching a completion to its start is by STABLE KEY (tool +
      // serialized args), never object identity: every SSE event is a
      // fresh JSON.parse, so `a.args === ev.args` never matched and
      // every step stayed "running" forever.
      return ev.result == null
        ? { ...seen,
            activity: [...(seen.activity || []), { tool: ev.tool, args: ev.args, key: toolKey(ev) }],
            timeline: [...tl, { kind: 'tool', tool: ev.tool, args: ev.args, key: toolKey(ev), startedAt: Date.now() }] }
        : { ...seen,
            activity: (seen.activity || []).map((a) => (a.key === toolKey(ev) ? { ...a, result: ev.result } : a)),
            timeline: tl.map((item) => (item.kind === 'tool' && item.key === toolKey(ev) ? { ...item, result: ev.result, endedAt: Date.now() } : item)) };
    case 'law_block':
      return { ...seen, lawBlocks: [...(seen.lawBlocks || []), ev], timeline: [...tl, { kind: 'law_block', article: ev.article, detail: ev.detail }] };
    case 'memory':
      return { ...seen, timeline: [...tl, { kind: 'memory', fact: ev.fact || ev.text }] };
    case 'context':
      // What the agent consulted before answering. Shown as its own step
      // so grounding is visible rather than implied by a citation marker.
      return { ...seen,
        citations: ev.source === 'knowledge' ? (ev.items || []) : (seen.citations || []),
        timeline: [...tl, { kind: 'context', source: ev.source, summary: ev.summary, items: ev.items || [] }] };
    case 'subagent': {
      // One named worker's lane in a swarm run. Keyed by name; ordered
      // by first appearance so lanes do not jump as events interleave.
      const sw = seen.swarm || { order: [], agents: {} };
      const name = ev.source || 'worker';
      const existing = sw.agents[name];
      const lane = existing
        ? { ...existing }
        : { name, status: 'running', events: [], stream: '', startedAt: Date.now() };
      if (ev.phase === 'spawned') lane.brief = ev.text || '';
      else if (ev.phase === 'done') {
        lane.status = 'done'; lane.output = ev.text || ''; lane.endedAt = Date.now();
      } else if (ev.phase === 'failed') {
        lane.status = 'failed'; lane.error = ev.text || ''; lane.endedAt = Date.now();
      } else if (ev.phase === 'text') {
        lane.stream = (lane.stream || '') + ((ev.body && ev.body.text) || '');
      } else if (lane.events.length < 200) {
        // tool / reasoning / artifact / error / law_block / governor.
        // Capped: this is a live view, not an archive — the durable
        // record lives server-side.
        lane.events = [...lane.events, { phase: ev.phase, body: ev.body || {} }];
      }
      return { ...seen, swarm: {
        order: existing ? sw.order : [...sw.order, name],
        agents: { ...sw.agents, [name]: lane },
      } };
    }
    case 'governor': {
      // The run's own supervisor speaking. The latest state feeds the
      // NowLine budget meter; a HALT is a first-class timeline step —
      // a run the governor stopped must say so where the user is
      // looking, not fade into an unexplained ending.
      const next = { ...seen, governor: ev };
      if (ev.action === 'halt') {
        next.timeline = [...tl, {
          kind: 'law_block',
          article: 'governor',
          detail: ev.reason || `run halted (${ev.pathology || 'bounds'})`,
        }];
      }
      return next;
    }
    case 'steering':
      // A human's mid-run course correction is part of the work record.
      // The board rendered these; the chat reducer silently dropped
      // them — one reducer, one truth.
      return { ...seen, timeline: [...tl, {
        kind: 'steering', author: ev.source, text: ev.text,
        delivered: ev.delivered ?? ev.extras?.delivered,
      }] };
    case 'question': {
      // Elicitation: Alex asked the crew and the run is parked on the
      // answer. Whole-state contract (like plan/acceptance): the event
      // carries the full question row, so the latest one wins. Cleared
      // from "waiting" the moment it is answered or cancelled.
      const q = ev.body || {};
      return { ...seen, question: q, timeline: [...tl, {
        kind: 'question', question: q.question, status: q.status,
        options: q.options, answer: q.answer, answered_by: q.answered_by,
      }] };
    }
    case 'confirmation_required':
      return { ...seen, card: ev };
    case 'artifact':
      return { ...seen, artifacts: [...(seen.artifacts || []), ev], timeline: [...tl, { kind: 'artifact', path: ev.path, artifactKind: ev.kind, preview: ev.preview }] };
    case 'done':
      return { ...seen, streaming: false };
    case 'error':
      return { ...seen, timeline: [...tl, { kind: 'error', message: ev.message, code: ev.code }] };
    default:
      return seen;
  }
}

function applyEvent(turns, turnIdx, ev) { return updateTurn(turns, turnIdx, (t) => reduceTurn(t, ev)); }
function updateTurn(turns, idx, fn) {
  // An event from an aborted stream can still arrive for an index that
  // no longer exists (the turns array was reset on session switch);
  // padding with holes here meant `t.role` of undefined crashed render.
  if (idx < 0 || idx >= turns.length || !turns[idx]) return turns;
  const out = [...turns];
  out[idx] = fn(out[idx]);
  return out;
}
