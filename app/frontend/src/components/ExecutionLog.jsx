// ExecutionLog.jsx — the visible record of what the agent is doing.
//
// A long agentic run must never look like a frozen spinner. This renders
// the turn's events the way a terminal harness does: one numbered line
// per step, in order, each with a live status glyph and a duration, and
// a header that keeps ticking while work is in flight. Any step can be
// expanded to show the exact arguments sent and the result returned, so
// "what did it actually do?" is always answerable without a log dive.

import React, { useEffect, useState } from 'react';
import { IconDownload } from './icons.jsx';
import { downloadArtifact } from '../lib/download.js';

const GLYPH = { ok: '✓', error: '✗', running: '●', blocked: '⊘', note: '·', context: '◆', steering: '⇥', question: '?' };

/** Re-render on an interval so in-flight durations keep counting up. */
function useTick(active, ms = 500) {
  const [, setN] = useState(0);
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => setN((n) => n + 1), ms);
    return () => clearInterval(id);
  }, [active, ms]);
}

export function formatDuration(ms) {
  if (ms == null || !isFinite(ms) || ms < 0) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60000);
  return `${m}m ${Math.round((ms % 60000) / 1000)}s`;
}

function statusOf(item) {
  if (item.kind === 'error') return 'error';
  if (item.kind === 'law_block') return 'blocked';
  if (item.kind === 'context') return 'context';
  if (item.kind === 'steering') return item.delivered ? 'steering' : 'note';
  if (item.kind === 'question') {
    if (item.status === 'open') return 'question';
    return item.status === 'answered' ? 'ok' : 'note';
  }
  if (item.kind === 'tool') return item.result != null ? 'ok' : 'running';
  return 'note';
}

const SOURCE_LABEL = {
  knowledge: 'enterprise knowledge',
  memory: 'your memory',
  graph: 'knowledge graph',
  compaction: 'context compression',
  request: 'model request',
  budget: 'turn budget',
};

/* The citations / memories / facts behind one context step. */
function ContextItems({ source, items }) {
  if (!items?.length) return null;
  if (source === 'graph') {
    return (
      <ul className="ctx-list">
        {items.map((f, i) => (
          <li key={i}>
            <span className="ctx-node">{f.src}</span>
            <span className="ctx-rel">{f.rel}</span>
            <span className="ctx-node">{f.dst}</span>
          </li>
        ))}
      </ul>
    );
  }
  if (source === 'memory') {
    return (
      <ul className="ctx-list">
        {items.map((m, i) => (
          <li key={i}>
            <span className="ctx-kind">{m.kind}</span> {m.content}
          </li>
        ))}
      </ul>
    );
  }
  return (
    <ol className="ctx-cites">
      {items.map((c, i) => (
        <li key={i} id={`cite-${c.n ?? i + 1}`}>
          <div className="ctx-cite-head">
            <span className="ctx-cite-n">[{c.n ?? i + 1}]</span>
            <span className="ctx-cite-title">{c.title || c.source_ref}</span>
            {c.classification ? (
              <span className={'ctx-class ' + c.classification}>{c.classification}</span>
            ) : null}
          </div>
          {c.source_ref ? <div className="ctx-cite-ref">{c.source_ref}</div> : null}
          <div className="ctx-cite-text">{c.text}</div>
        </li>
      ))}
    </ol>
  );
}

function labelOf(item) {
  switch (item.kind) {
    case 'tool':
      // Text preserved verbatim: this is the contract the chat tests and
      // screen readers rely on.
      return `called ${item.tool}${item.result != null ? ' ✓' : '…'}`;
    case 'law_block':
      return `Blocked by ${item.article || 'the constitution'}`;
    case 'memory':
      return `learned: ${item.fact || item.text || ''}`;
    case 'context':
      return item.summary || `consulted ${SOURCE_LABEL[item.source] || item.source}`;
    case 'artifact':
      return item.artifactKind === 'tool_result'
        ? `→ ${item.preview || ''}`
        : String(item.path || '').split('/').pop();
    case 'error':
      return item.message || 'error';
    case 'steering':
      return item.delivered === false
        ? `${item.author || 'someone'} steered — queued`
        : `steered by ${item.author || 'someone'}`;
    case 'question':
      if (item.status === 'answered') {
        return `asked the crew — answered by ${item.answered_by || 'the crew'}`;
      }
      if (item.status === 'cancelled') return 'asked the crew — run stopped';
      return `asked the crew: ${item.question || ''}`;
    default:
      return '';
  }
}

function pretty(value) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/** One step. Tool steps expand to show arguments and result. */
function Row({ item, index, onRetry, authHeaders }) {
  const [open, setOpen] = useState(false);
  const status = statusOf(item);
  const label = labelOf(item);
  const expandable =
    (item.kind === 'tool' && (item.args != null || item.result != null)) ||
    (item.kind === 'context' && item.items?.length > 0);
  const duration =
    item.startedAt != null
      ? formatDuration((item.endedAt ?? Date.now()) - item.startedAt)
      : '';

  if (item.kind === 'error') {
    return (
      <div className="exec-row" data-status="error">
        <span className="exec-step">{index}</span>
        <span className="exec-glyph" aria-hidden="true">{GLYPH.error}</span>
        <div className="exec-body">
          <div className="error-card" role="alert">
            <b>Something went wrong</b>
            <span>{item.message || 'The turn could not be completed.'}</span>
            {item.code ? <code className="err-code">{item.code}</code> : null}
            {onRetry ? (
              <button type="button" className="btn-retry" onClick={onRetry}>
                Retry
              </button>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="exec-row" data-status={status}>
      <span className="exec-step">{index}</span>
      <span className="exec-glyph" aria-hidden="true">{GLYPH[status]}</span>
      <div className="exec-body">
        {expandable ? (
          <button
            type="button"
            className={'exec-label act-chip ' + item.kind}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {label}
          </button>
        ) : item.kind === 'artifact' && item.artifactKind !== 'tool_result' ? (
          <button
            type="button"
            className="act-chip artifact exec-label"
            onClick={() => downloadArtifact(item.path, authHeaders)}
          >
            <IconDownload width={13} height={13} /> {label}
          </button>
        ) : (
          <span className={'exec-label act-chip ' + item.kind}>{label}</span>
        )}
        {item.kind === 'law_block' ? (
          <span className="exec-detail">
            {item.detail || 'This action was refused by policy.'}
          </span>
        ) : null}
        {item.kind === 'steering' ? (
          <span className="exec-detail steering-note">{item.text}</span>
        ) : null}
        {open && item.kind === 'context' ? (
          <div className="exec-expand">
            <div className="exec-expand-h">
              from {SOURCE_LABEL[item.source] || item.source}
            </div>
            <ContextItems source={item.source} items={item.items} />
          </div>
        ) : null}
        {open && item.kind !== 'context' ? (
          <div className="exec-expand">
            {item.args != null ? (
              <>
                <div className="exec-expand-h">arguments</div>
                <pre className="exec-pre">{pretty(item.args)}</pre>
              </>
            ) : null}
            {item.result != null ? (
              <>
                <div className="exec-expand-h">result</div>
                <pre className="exec-pre">{pretty(item.result)}</pre>
              </>
            ) : null}
          </div>
        ) : null}
      </div>
      {duration ? <span className="exec-dur">{duration}</span> : null}
    </div>
  );
}

/**
 * The turn's execution log.
 *
 * `streaming` keeps the header's elapsed clock live and the log open;
 * once the turn finishes the log collapses to a one-line summary the
 * user can reopen.
 */
export default function ExecutionLog({
  timeline,
  onRetry,
  authHeaders,
  streaming = false,
  startedAt = null,
}) {
  const running = timeline?.some((i) => statusOf(i) === 'running') || streaming;
  useTick(running);
  const [open, setOpen] = useState(true);
  // Re-open automatically when a finished turn starts working again.
  useEffect(() => {
    if (running) setOpen(true);
  }, [running]);

  if (!timeline?.length) return null;

  const steps = timeline.filter((i) => i.kind === 'tool').length;
  const failed = timeline.some((i) => statusOf(i) === 'error');
  const elapsed = startedAt != null ? formatDuration(Date.now() - startedAt) : '';
  const summary = [
    steps ? `${steps} step${steps === 1 ? '' : 's'}` : `${timeline.length} events`,
    elapsed,
    running ? 'running' : failed ? 'failed' : 'done',
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <section
      className="exec-log"
      data-running={running ? 'true' : 'false'}
      aria-label="Agent execution log"
    >
      <button
        type="button"
        className="exec-head"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="exec-head-caret" aria-hidden="true">{open ? '▾' : '▸'}</span>
        <span className="exec-head-title">Execution</span>
        <span className="exec-head-meta">{summary}</span>
      </button>
      {open ? (
        <div className="exec-rows">
          {timeline.map((item, i) => (
            <Row
              key={i}
              item={item}
              index={i + 1}
              onRetry={onRetry}
              authHeaders={authHeaders}
            />
          ))}
        </div>
      ) : null}
    </section>
  );
}
