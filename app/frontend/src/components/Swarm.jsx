// The swarm panel — one lane per named sub-agent, live.
//
// Design signals this follows (agentic-UX guidance + the strong open
// harnesses): named lanes with a status readable at a glance; a live
// "now" verb per worker in PLAIN language, so every lane answers "what
// is this one doing?" the way the main status line does; a progress
// summary in the header; the worker's thinking behind progressive
// disclosure rather than dumped; worker output framed as a distinct
// result block (never something a person appears to have said); and
// failures loud, because a lane that quietly vanishes reads as
// "nothing to report" when it means "nobody looked".

import React, { useEffect, useState } from 'react';
import { verbFor } from '../lib/verbs.js';

function laneStatus(a) {
  if (a.status === 'done') return { cls: 'is-done', label: 'done' };
  if (a.status === 'failed') return { cls: 'is-failed', label: 'failed' };
  return { cls: 'is-running', label: 'working' };
}

/** Re-render each second while any lane is live, so elapsed ticks. */
function useTick(active, ms = 1000) {
  const [, setN] = useState(0);
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => setN((n) => n + 1), ms);
    return () => clearInterval(id);
  }, [active, ms]);
}

function elapsed(a) {
  const end = a.endedAt || Date.now();
  const s = Math.max(0, Math.round((end - (a.startedAt || end)) / 1000));
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
}

/** What this worker is doing right now, in plain language. */
function laneNow(a) {
  if (a.status !== 'running') return '';
  const tools = a.events.filter((e) => e.phase === 'tool' && e.body?.tool);
  if (tools.length) return `${verbFor(tools[tools.length - 1].body.tool)}…`;
  if (a.stream) return 'Writing…';
  return 'Thinking…';
}

function LaneEvents({ lane }) {
  const tools = lane.events.filter((e) => e.phase === 'tool' && e.body?.tool);
  const thinking = lane.events.filter((e) => e.phase === 'reasoning');
  const troubles = lane.events.filter(
    (e) => e.phase === 'error' || e.phase === 'law_block' || e.phase === 'governor'
  );
  return (
    <div className="sw-detail">
      {thinking.length > 0 && (
        <details className="sw-think-wrap">
          <summary className="sw-think-head">
            thinking ({thinking.length})
          </summary>
          <div className="sw-thinking">
            {thinking.map((e, i) => (
              <div key={i}>{e.body?.text || ''}</div>
            ))}
          </div>
        </details>
      )}
      {tools.length > 0 && (
        <div className="sw-tools">
          {tools.map((e, i) => (
            <span className="sw-tool" key={i} title={e.body.tool}>
              {verbFor(e.body.tool)}
            </span>
          ))}
        </div>
      )}
      {lane.stream && <div className="sw-stream">{lane.stream}</div>}
      {troubles.map((e, i) => (
        <div className="sw-trouble" key={i}>
          {e.phase === 'law_block'
            ? `Blocked by the constitution: ${e.body?.detail || e.body?.article || ''}`
            : e.phase === 'governor'
              ? `Governor: ${e.body?.reason || e.body?.action || ''}`
              : e.body?.message || e.body?.detail || 'error'}
        </div>
      ))}
    </div>
  );
}

export default function SwarmPanel({ swarm }) {
  const lanes = (swarm.order || []).map((n) => swarm.agents[n]).filter(Boolean);
  const anyRunning = lanes.some((a) => a.status === 'running');
  useTick(anyRunning);
  if (!lanes.length) return null;
  const done = lanes.filter((a) => a.status === 'done').length;
  const failed = lanes.filter((a) => a.status === 'failed').length;
  const settled = done + failed;
  return (
    <section className="sw" aria-label="Sub-agents">
      <div className="sw-head">
        <span className="sw-title">Swarm</span>
        <span className="sw-count">
          {lanes.length} worker{lanes.length === 1 ? '' : 's'} · {done} done
          {failed ? ` · ${failed} failed` : ''}
        </span>
        <span
          className="sw-progress"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={lanes.length}
          aria-valuenow={settled}
          aria-label="Sub-agent progress"
        >
          <span
            className="sw-progress-fill"
            style={{ width: `${lanes.length ? (settled / lanes.length) * 100 : 0}%` }}
          />
        </span>
      </div>
      <div className="sw-lanes">
        {lanes.map((a) => {
          const st = laneStatus(a);
          const now = laneNow(a);
          // What this worker has SPENT — tool calls are the unit the
          // child budget is sliced in, so the lane wears its meter.
          const toolCount = a.events.filter(
            (e) => e.phase === 'tool' && e.body?.tool
          ).length;
          return (
            <details className={`sw-lane ${st.cls}`} key={a.name}
                     open={a.status === 'failed'}>
              <summary className="sw-lane-head">
                <span className={`sw-dot ${st.cls}`} aria-hidden="true" />
                <span className="sw-name">{a.name}</span>
                {now
                  ? <span className="sw-now">{now}</span>
                  : <span className="sw-status">{st.label}</span>}
                {toolCount ? (
                  <span className="sw-usage" title="Tool calls this worker has made">
                    {toolCount} tool{toolCount === 1 ? '' : 's'}
                  </span>
                ) : null}
                <span className="sw-elapsed">{elapsed(a)}</span>
              </summary>
              {a.brief && <div className="sw-brief">{a.brief}</div>}
              <LaneEvents lane={a} />
              {a.output && a.status === 'done' && (
                <div className="sw-output">
                  <div className="sw-output-h">worker result</div>
                  {a.output}
                </div>
              )}
              {a.error && <div className="sw-trouble">{a.error}</div>}
            </details>
          );
        })}
      </div>
    </section>
  );
}
