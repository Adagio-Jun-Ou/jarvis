// Board.jsx — the shared queue people and Alex work together.
//
// The board is not a to-do list: each card is a small workspace with a
// crew, a thread, and — when Alex is assigned — a live run whose every
// step is visible to everyone on the task. Selecting a card opens the
// detail pane: the discussion on one side, Alex's execution log on the
// other, so "what is it doing?" and "what did we decide?" sit together.

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useResource } from '../lib/swr.js';
import ExecutionLog from './ExecutionLog.jsx';
import NowLine from './NowLine.jsx';
import Presence from './Presence.jsx';
import SwarmPanel from './Swarm.jsx';
import { reduceTurn } from './Chat.jsx';
import { IconSpark } from './icons.jsx';
import { toast } from '../lib/toast.jsx';
import { renderMarkdown } from '../lib/markdown.jsx';

const LANE_LABEL = {
  backlog: 'Backlog',
  todo: 'To do',
  in_progress: 'In progress',
  review: 'Review',
  done: 'Done',
};

const PRIORITY_LABEL = { 1: 'P1', 2: 'P2', 3: 'P3', 4: 'P4', 5: 'P5' };

function initials(name) {
  const s = String(name || '').trim();
  if (!s) return '?';
  if (s === 'alex') return 'AI';
  const local = s.split('@')[0];
  const bits = local.split(/[._-]+/).filter(Boolean);
  return ((bits[0]?.[0] || '') + (bits[1]?.[0] || '')).toUpperCase() || local.slice(0, 2).toUpperCase();
}

function Avatar({ member, size = 22 }) {
  const isAI = member === 'alex';
  return (
    <span
      className={'bd-avatar' + (isAI ? ' ai' : '')}
      style={{ width: size, height: size, fontSize: size * 0.42 }}
      title={isAI ? 'Alex (AI)' : member}
    >
      {isAI ? <IconSpark width={size * 0.55} height={size * 0.55} /> : initials(member)}
    </span>
  );
}

/* Due-date chip. Overdue is the loudest state on a card short of a live
 * run: a deadline that already passed is the one fact a task system must
 * never let a card whisper. Done tasks carry no urgency — the date is
 * history there, not pressure. */
function DueChip({ due, status }) {
  if (!due || status === 'done') return null;
  const d = new Date(due);
  if (Number.isNaN(d.getTime())) return null;
  const days = Math.floor((d - Date.now()) / 86400000);
  const label = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  if (days < 0) return <span className="bd-due overdue" title={`Due ${label}`}>overdue · {label}</span>;
  if (days <= 2) return <span className="bd-due soon" title={`Due ${label}`}>due {label}</span>;
  return <span className="bd-due" title={`Due ${label}`}>{label}</span>;
}

function Card({ task, active, onOpen }) {
  const running = task.run?.status === 'running';
  return (
    <button
      type="button"
      className={'bd-card' + (active ? ' active' : '') + (running ? ' running' : '')}
      onClick={() => onOpen(task.task_id)}
      aria-current={active ? 'true' : undefined}
    >
      <div className="bd-card-top">
        <span className={'bd-prio p' + task.priority}>{PRIORITY_LABEL[task.priority] || 'P3'}</span>
        {task.source === 'email' ? (
          <span className="bd-src" title="Adopted from an email ask">✉ email</span>
        ) : null}
        {task.waiting_on > 0 ? (
          <span className="bd-waiting" title="Blocked by unfinished tasks">
            ⛓ waiting on {task.waiting_on}
          </span>
        ) : null}
        <DueChip due={task.due_at} status={task.status} />
        {task.open_questions > 0 ? (
          <span className="bd-needs" title="Alex asked a question and is waiting">
            ? needs your answer
          </span>
        ) : running ? (
          <span className="bd-live">● Alex working</span>
        ) : null}
      </div>
      <div className="bd-card-title">{task.title}</div>
      <div className="bd-card-foot">
        <span className="bd-crew">
          {(task.crew || []).slice(0, 4).map((c) => (
            <Avatar key={c.member} member={c.member} size={20} />
          ))}
        </span>
        {task.run && !running ? (
          <span className="bd-run-note">{task.run.steps} steps</span>
        ) : null}
      </div>
    </button>
  );
}

/* The task thread — humans and Alex in one record. When a run is
 * live, a post is also steered into the loop: the compose box says
 * so, because "this will reach Alex now" is different information
 * from "this will be read later". */
function Thread({ messages, onSend, busy, live }) {
  const [draft, setDraft] = useState('');
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: 'end' });
  }, [messages]);

  const submit = (e) => {
    e.preventDefault();
    const body = draft.trim();
    if (!body) return;
    setDraft('');
    onSend(body);
  };

  return (
    <div className="bd-thread">
      <div className="bd-msgs">
        {(messages || []).map((m) => (
          <div key={m.message_id} className={'bd-msg ' + m.kind}>
            {m.kind === 'system' ? (
              <div className="bd-sys">{m.body}</div>
            ) : (
              <>
                <Avatar member={m.author} size={26} />
                <div className="bd-msg-body">
                  <div className="bd-msg-author">
                    {m.author === 'alex' ? 'Alex' : m.author}
                  </div>
                  <div className="bd-msg-text">
                    {m.kind === 'agent' ? renderMarkdown(m.body) : m.body}
                  </div>
                </div>
              </>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <form className={'bd-compose' + (live ? ' live' : '')} onSubmit={submit}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={
            live
              ? 'Message the crew — steers Alex mid-run…'
              : 'Message the crew…'
          }
          aria-label="Message the crew"
          disabled={busy}
        />
        <button type="submit" className="btn primary" disabled={busy || !draft.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

/* Acceptance criteria: the crew writes the bar, Alex only verifies it.
 * The [x]/[!]/[ ] marks mirror the run prompt so what the reviewer sees
 * is exactly what the agent was told. */
function Acceptance({ task, taskId, authHeaders, onChanged }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const criteria = task.acceptance || [];

  const save = async (e) => {
    e.preventDefault();
    const lines = draft.split('\n').map((s) => s.trim()).filter(Boolean);
    const r = await fetch(`/api/board/tasks/${taskId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ acceptance: lines }),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      toast(d.error || 'Could not save the criteria.');
      return;
    }
    setEditing(false);
    onChanged?.();
  };

  if (editing) {
    return (
      <form className="bd-accept bd-accept-edit" onSubmit={save}>
        <div className="bd-accept-h">Definition of done <span className="bd-accept-hint">one criterion per line</span></div>
        <textarea
          autoFocus
          rows={Math.max(3, criteria.length + 1)}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          aria-label="Acceptance criteria, one per line"
        />
        <div className="bd-newform-actions">
          <button type="submit" className="btn primary">Save</button>
          <button type="button" className="btn" onClick={() => setEditing(false)}>Cancel</button>
        </div>
      </form>
    );
  }

  if (!criteria.length) {
    return (
      <button
        type="button"
        className="bd-accept-add"
        onClick={() => { setDraft(''); setEditing(true); }}
      >
        + Define what “done” means
      </button>
    );
  }

  const met = criteria.filter((c) => c.status === 'met').length;
  return (
    <div className="bd-accept" aria-label="Acceptance criteria">
      <div className="bd-accept-h">
        Definition of done
        <span className="bd-plan-count">{met}/{criteria.length} met</span>
        <button
          type="button"
          className="bd-accept-editbtn"
          onClick={() => { setDraft(criteria.map((c) => c.text).join('\n')); setEditing(true); }}
          aria-label="Edit criteria"
        >
          edit
        </button>
      </div>
      <ol className="bd-accept-list">
        {criteria.map((c) => (
          <li key={c.crit_no} className={`bd-crit is-${c.status}`}>
            <span className="bd-step-mark" aria-hidden="true" />
            <span className="bd-crit-text">{c.text}</span>
            {c.note && <span className="bd-step-note">{c.note}</span>}
          </li>
        ))}
      </ol>
    </div>
  );
}

/* Alex's open questions — the "waiting on YOU" panel. Elicitation is
 * the one state where the run is parked on a person: it must be the
 * loudest thing in the pane, with the options one click away. */
function Questions({ task, taskId, authHeaders, onAnswered }) {
  const [drafts, setDrafts] = useState({});
  const [busyQ, setBusyQ] = useState(null);
  const open = (task.questions || []).filter((q) => q.status === 'open');
  if (!open.length) return null;

  const answer = async (q, text) => {
    const body = String(text || '').trim();
    if (!body) return;
    setBusyQ(q.question_id);
    try {
      const r = await fetch(
        `/api/board/tasks/${taskId}/questions/${q.question_id}/answer`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeaders() },
          body: JSON.stringify({ answer: body }),
        }
      );
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        toast(d.error || 'Could not send the answer.');
        return;
      }
      setDrafts((cur) => ({ ...cur, [q.question_id]: '' }));
      onAnswered?.();
    } finally {
      setBusyQ(null);
    }
  };

  return (
    <div className="bd-ask" aria-label="Alex is waiting for your answer">
      <div className="bd-ask-h">Alex is waiting on you</div>
      {open.map((q) => (
        <div key={q.question_id} className="bd-ask-q">
          <div className="bd-ask-text">{q.question}</div>
          {q.options?.length ? (
            <div className="bd-ask-opts">
              {q.options.map((o) => (
                <button
                  key={o}
                  type="button"
                  className="btn bd-ask-opt"
                  disabled={busyQ === q.question_id}
                  onClick={() => answer(q, o)}
                >
                  {o}
                </button>
              ))}
            </div>
          ) : null}
          <form
            className="bd-ask-free"
            onSubmit={(e) => { e.preventDefault(); answer(q, drafts[q.question_id]); }}
          >
            <input
              value={drafts[q.question_id] || ''}
              onChange={(e) =>
                setDrafts((cur) => ({ ...cur, [q.question_id]: e.target.value }))}
              placeholder={q.options?.length ? 'Or answer in your own words…' : 'Your answer…'}
              aria-label="Answer Alex"
              disabled={busyQ === q.question_id}
            />
            <button
              type="submit"
              className="btn primary"
              disabled={busyQ === q.question_id || !(drafts[q.question_id] || '').trim()}
            >
              Answer
            </button>
          </form>
        </div>
      ))}
    </div>
  );
}

/* Detail pane: discussion + Alex's live execution log. */
function Detail({ taskId, session, onChanged, onClose }) {
  const authHeaders = useCallback(
    () => session?.authHeaders?.() || {},
    [session]
  );
  const [task, setTask] = useState(null);
  const [events, setEvents] = useState([]);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState('thread');
  const esRef = useRef(null);
  const taskIdRef = useRef(taskId);
  taskIdRef.current = taskId;

  const load = useCallback(async () => {
    // Task A clicked, then task B: A's slower response must not
    // overwrite B's detail pane while actions still target B. The
    // closure's taskId is the one the load was FOR; the ref's is the
    // one the user is LOOKING at now.
    const wanted = taskId;
    const r = await fetch(`/api/board/tasks/${taskId}`, { headers: authHeaders() });
    if (!r.ok || taskIdRef.current !== wanted) return;
    const d = await r.json();
    if (taskIdRef.current !== wanted) return;
    setTask(d);
    if (d.run?.events) {
      setEvents((cur) => {
        // Only adopt the snapshot when we're not already streaming a
        // newer live record for this same task.
        if (cur.length && cur[0]?.type === 'run_started') return cur;
        return d.run.events;
      });
    }
  }, [taskId, authHeaders]);

  useEffect(() => {
    setTask(null);
    setEvents([]);
    load();
  }, [taskId, load]);

  // Live team chat: while the pane is open, the thread and question
  // state refresh on a light poll of one cheap route — an agent
  // question, another crew member's message, or a mid-run agent post
  // appears without reopening the card.
  const refreshThread = useCallback(async () => {
    const wanted = taskId;
    const r = await fetch(`/api/board/tasks/${taskId}/messages`, {
      headers: authHeaders(),
    });
    if (!r.ok || taskIdRef.current !== wanted) return;
    const d = await r.json();
    if (taskIdRef.current !== wanted) return;
    setTask((cur) => (cur ? {
      ...cur,
      messages: d.messages || cur.messages,
      questions: d.questions ?? cur.questions,
    } : cur));
  }, [taskId, authHeaders]);

  useEffect(() => {
    const id = setInterval(refreshThread, 4000);
    return () => clearInterval(id);
  }, [refreshThread]);

  // Follow the live run. EventSource can't carry the bearer token, so
  // the stream is read with fetch + a reader, same as the chat surface.
  // `onChanged` rides in a ref: as a fresh closure per Board render it
  // used to change `follow`'s identity every 4-second poll, which
  // reconnected the stream — and the server replays the whole backlog
  // to each new subscriber, multiplying every step in the timeline.
  const onChangedRef = useRef(onChanged);
  onChangedRef.current = onChanged;
  const follow = useCallback(async () => {
    esRef.current?.abort?.();
    const ac = new AbortController();
    esRef.current = ac;
    try {
      const r = await fetch(`/api/board/tasks/${taskId}/stream`, {
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
            // A reconnect replays the full backlog; replace on
            // run_started (a replay marker) rather than append, and
            // cap so a long run cannot grow the array unboundedly.
            setEvents((prev) => {
              if (ev.type === 'run_started') return [ev];
              const next = [...prev, ev];
              return next.length > 2000 ? next.slice(-1000) : next;
            });
            if (ev.type === 'run_finished') {
              load();
              onChangedRef.current?.();
            }
          } catch { /* keep-alive or partial frame */ }
        }
      }
    } catch { /* aborted or disconnected */ }
  }, [taskId, authHeaders, load]);

  useEffect(() => {
    if (task?.run?.status === 'running') follow();
    return () => esRef.current?.abort?.();
  }, [task?.run?.status, follow]);

  const assignAI = async () => {
    setBusy(true);
    try {
      const r = await fetch(`/api/board/tasks/${taskId}/assign-ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        if (r.status === 409 && d.blocked_by?.length) {
          toast(
            'Blocked — waiting on: '
            + d.blocked_by.map((b) => b.title || b.task_id).join(', ')
          );
        } else {
          toast(d.error || 'Could not hand this to Alex.');
        }
        return;
      }
      setEvents([]);
      setTab('work');
      await load();
      follow();
      onChanged?.();
    } finally {
      setBusy(false);
    }
  };

  const send = async (body) => {
    await fetch(`/api/board/tasks/${taskId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ body }),
    });
    load();
  };

  const move = async (status) => {
    await fetch(`/api/board/tasks/${taskId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ status }),
    });
    load();
    onChanged?.();
  };

  if (!task) return <aside className="bd-detail"><div className="bd-empty">Loading…</div></aside>;

  const running = task.run?.status === 'running';
  // ONE reducer, one truth: the run's events flow through the same
  // reduceTurn the chat uses, so the board renders everything the chat
  // can — tools, context, law blocks, steering, GOVERNOR verdicts and
  // SUB-AGENT lanes. The old hand-rolled mapping silently dropped
  // governor and subagent events: a halted or swarming task run showed
  // none of it on the board.
  const turnView = (() => {
    let t = {
      role: 'ai', content: '', timeline: [], activity: [], reasoning: [],
      streaming: running, gotFirstEvent: true,
      startedAt: task.run?.started_at
        ? Number(task.run.started_at) * 1000 : Date.now(),
    };
    for (const ev of events) {
      if (!ev || !ev.type || ev.type === 'run_started' || ev.type === 'run_finished') continue;
      t = reduceTurn(t, ev);
    }
    return t;
  })();
  const timeline = turnView.timeline;

  return (
    <aside className="bd-detail" aria-label="Task detail">
      <header className="bd-detail-head">
        <div>
          <div className="bd-detail-title">{task.title}</div>
          <div className="bd-detail-sub">
            {LANE_LABEL[task.status]} · {PRIORITY_LABEL[task.priority]} ·{' '}
            {(task.crew || []).length} on this task
            {' '}<DueChip due={task.due_at} status={task.status} />
          </div>
        </div>
        <button type="button" className="bd-x" onClick={onClose} aria-label="Close task">×</button>
      </header>

      {task.description ? <p className="bd-desc">{task.description}</p> : null}

      <Questions
        task={task}
        taskId={taskId}
        authHeaders={authHeaders}
        onAnswered={() => { refreshThread(); load(); onChanged?.(); }}
      />

      {task.plan?.length ? (
        <div className="bd-plan" aria-label="Task plan">
          <div className="bd-plan-h">
            Plan
            <span className="bd-plan-count">
              {task.plan.filter((s) => s.status === 'done').length}/{task.plan.length} done
            </span>
          </div>
          <ol className="bd-plan-steps">
            {task.plan.map((s) => (
              <li key={s.step_no} className={`bd-step is-${s.status}`}>
                <span className="bd-step-mark" aria-hidden="true" />
                <span className="bd-step-title">{s.title}</span>
                {s.note && <span className="bd-step-note">{s.note}</span>}
                {s.status === 'blocked' && (
                  <span className="bd-step-flag">needs a decision</span>
                )}
              </li>
            ))}
          </ol>
        </div>
      ) : null}

      {/* Acceptance criteria — the crew's bar for the work. People write
        * them; Alex may only mark each met/unmet with evidence. */}
      <Acceptance
        task={task}
        taskId={taskId}
        authHeaders={authHeaders}
        onChanged={load}
      />

      {task.blocked_by?.filter((b) => b.open).length ? (
        <div className="bd-blockers" aria-label="Waiting on">
          <div className="bd-blockers-h">Waiting on</div>
          <ul>
            {task.blocked_by.filter((b) => b.open).map((b) => (
              <li key={b.task_id}>
                <span className="bd-blocker-title">{b.title || b.task_id}</span>
                <span className="bd-blocker-status">{LANE_LABEL[b.status] || b.status}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {task.past_work?.length ? (
        <div className="bd-past">
          <div className="bd-past-h">Related past work</div>
          <ul>
            {task.past_work.map((w, i) => (
              <li key={i}>
                <span className="bd-past-text">{w.content}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="bd-actions">
        {running ? (
          <button
            type="button"
            className="btn"
            onClick={async () => {
              await fetch(`/api/board/tasks/${taskId}/stop`, {
                method: 'POST', headers: authHeaders(),
              });
              load();
            }}
          >
            Stop Alex
          </button>
        ) : (
          <button type="button" className="btn primary" onClick={assignAI} disabled={busy}>
            <IconSpark width={13} height={13} /> {task.run ? 'Run again' : 'Hand to Alex'}
          </button>
        )}
        <select
          value={task.status}
          onChange={(e) => move(e.target.value)}
          aria-label="Move task"
          className="bd-move"
        >
          {Object.keys(LANE_LABEL).map((k) => (
            <option key={k} value={k}>{LANE_LABEL[k]}</option>
          ))}
        </select>
        <span className="bd-crew">
          {(task.crew || []).map((c) => <Avatar key={c.member} member={c.member} />)}
        </span>
      </div>

      <div className="bd-tabs" role="tablist">
        {['thread', 'work'].map((t) => (
          <button
            key={t}
            role="tab"
            type="button"
            aria-selected={tab === t}
            className={'bd-tab' + (tab === t ? ' active' : '')}
            onClick={() => setTab(t)}
          >
            {t === 'thread' ? 'Discussion' : 'Alex’s work'}
            {t === 'work' && running ? <span className="bd-dot" /> : null}
          </button>
        ))}
      </div>

      {tab === 'thread' ? (
        <Thread messages={task.messages} onSend={send} busy={busy} live={running} />
      ) : (
        <div className="bd-work">
          {running ? (
            <div className="bd-nowrow">
              <Presence turn={turnView} size={22} />
              <NowLine turn={turnView} />
            </div>
          ) : null}
          {turnView.swarm ? <SwarmPanel swarm={turnView.swarm} /> : null}
          {timeline.length ? (
            <ExecutionLog timeline={timeline} streaming={running} authHeaders={authHeaders} />
          ) : (
            <div className="bd-empty">
              {running ? 'Alex just started — steps will appear here.'
                : 'Alex has not worked this task yet.'}
            </div>
          )}
          {task.run?.summary && !running ? (
            <div className="bd-summary">
              <div className="bd-summary-h">Result</div>
              {task.run.summary}
            </div>
          ) : null}
          {!running ? (() => {
            // What the run consumed — the durable budget accounting the
            // governor kept, so cost is a fact on the card, not a guess.
            const usage = [...events].reverse().find((e) => e?.type === 'run_usage');
            if (!usage) return null;
            const bits = [
              usage.turn > 1 ? `${usage.turn} working turns` : '',
              usage.llm_turns != null ? `${usage.llm_turns} model passes` : '',
              usage.tool_calls != null ? `${usage.tool_calls} tool calls` : '',
              usage.elapsed_s != null ? `${Math.round(usage.elapsed_s)}s working` : '',
            ].filter(Boolean);
            return bits.length ? (
              <div className="bd-usage" aria-label="Run resource usage">
                {bits.join(' · ')}
              </div>
            ) : null;
          })() : null}
        </div>
      )}
    </aside>
  );
}

function NewTask({ session, onCreated }) {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [due, setDue] = useState('');

  const submit = async (e) => {
    e.preventDefault();
    if (!title.trim()) return;
    await fetch('/api/board/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(session?.authHeaders?.() || {}) },
      body: JSON.stringify({ title, description, due_at: due || null }),
    });
    setTitle('');
    setDescription('');
    setDue('');
    setOpen(false);
    onCreated?.();
  };

  if (!open) {
    return (
      <button type="button" className="btn bd-new" onClick={() => setOpen(true)}>
        + New task
      </button>
    );
  }
  return (
    <form className="bd-newform" onSubmit={submit}>
      <input
        autoFocus
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="What needs doing?"
        aria-label="Task title"
      />
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Any detail Alex or the team should know…"
        aria-label="Task detail"
        rows={2}
      />
      <label className="bd-newform-due">
        Due
        <input
          type="date"
          value={due}
          onChange={(e) => setDue(e.target.value)}
          aria-label="Due date"
        />
      </label>
      <div className="bd-newform-actions">
        <button type="submit" className="btn primary" disabled={!title.trim()}>Create</button>
        <button type="button" className="btn" onClick={() => setOpen(false)}>Cancel</button>
      </div>
    </form>
  );
}

/* Work the system noticed in the employee's email, waiting to be
 * adopted. Nothing here is a task yet — accepting is the human gate
 * that turns an (untrusted) email ask into agent-runnable work. */
function Proposals({ session, onAdopted }) {
  const authHeaders = useCallback(() => session?.authHeaders?.() || {}, [session]);
  const [nonce, setNonce] = useState(0);
  const { data } = useResource(
    session?.email ? `proposals:${session.email}:${nonce}` : null,
    (signal) =>
      fetch('/api/board/proposals', { signal, headers: authHeaders() }).then((r) =>
        r.ok ? r.json() : { proposals: [] }
      ),
    { maxAge: 30000 }
  );
  const proposals = data?.proposals || [];
  if (!proposals.length) return null;

  const decide = async (id, action) => {
    const r = await fetch(`/api/board/proposals/${id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ action }),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      toast(d.error || 'Could not update the proposal.');
    } else if (action === 'accept') {
      onAdopted?.();
    }
    setNonce((n) => n + 1);
  };

  return (
    <div className="bd-props" aria-label="Suggested from your email">
      <div className="bd-props-h">
        ✉ Found in your email
        <span className="bd-props-n">{proposals.length}</span>
      </div>
      <div className="bd-props-row">
        {proposals.slice(0, 6).map((p) => (
          <div key={p.proposal_id} className="bd-prop">
            <div className="bd-prop-title">{p.title}</div>
            {p.description ? <div className="bd-prop-desc">{p.description}</div> : null}
            <div className="bd-prop-actions">
              <button type="button" className="btn primary" onClick={() => decide(p.proposal_id, 'accept')}>
                Add to board
              </button>
              <button type="button" className="btn" onClick={() => decide(p.proposal_id, 'dismiss')}>
                Dismiss
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Board({ session }) {
  const [selected, setSelected] = useState(null);
  const [nonce, setNonce] = useState(0);
  const authHeaders = useCallback(() => session?.authHeaders?.() || {}, [session]);

  const { data, loading, retry } = useResource(
    session?.email ? `board:${session.email}:${nonce}` : null,
    (signal) =>
      fetch('/api/board/tasks', { signal, headers: authHeaders() }).then((r) =>
        r.ok ? r.json() : { tasks: [], columns: [] }
      ),
    { maxAge: 4000 }
  );

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  const columns = data?.columns || Object.keys(LANE_LABEL);
  const tasks = data?.tasks || [];
  // Keep the board fresh while a run is in flight.
  const anyRunning = tasks.some((t) => t.run?.status === 'running');
  useEffect(() => {
    if (!anyRunning) return undefined;
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, [anyRunning, refresh]);

  return (
    <div className="bd">
      <div className="bd-main">
        <Proposals session={session} onAdopted={refresh} />
        <div className="bd-bar">
          <NewTask session={session} onCreated={refresh} />
          <span className="bd-count">
            {tasks.length} task{tasks.length === 1 ? '' : 's'}
            {anyRunning ? ' · Alex is working' : ''}
          </span>
        </div>
        {loading && !tasks.length ? (
          <div className="bd-empty">Loading the board…</div>
        ) : (
          <div className="bd-lanes">
            {columns.map((lane) => {
              const inLane = tasks.filter((t) => t.status === lane);
              return (
                <section className="bd-lane" key={lane} aria-label={LANE_LABEL[lane]}>
                  <header className="bd-lane-head">
                    <span>{LANE_LABEL[lane] || lane}</span>
                    <span className="bd-lane-n">{inLane.length}</span>
                  </header>
                  <div className="bd-lane-cards">
                    {inLane.map((t) => (
                      <Card
                        key={t.task_id}
                        task={t}
                        active={selected === t.task_id}
                        onOpen={setSelected}
                      />
                    ))}
                    {!inLane.length ? <div className="bd-lane-empty">—</div> : null}
                  </div>
                </section>
              );
            })}
          </div>
        )}
      </div>
      {selected ? (
        <Detail
          taskId={selected}
          session={session}
          onChanged={() => { refresh(); retry?.(); }}
          onClose={() => setSelected(null)}
        />
      ) : null}
    </div>
  );
}
