// NowLine — the live "what is happening right now" status line.
//
// The single most recognisable signal of the strong agent harnesses
// (Claude Code's star spinner is the canonical form): one line that
// always answers "what is it doing?", in plain language, with the
// elapsed clock ticking and the run's budget visible. Transparency
// research calls this proactive status communication — the user should
// never have to infer state from silence.
//
//   ✳ Searching enterprise knowledge… · 24s · pass 3/12 · 7 tools
//
// The verb comes from lib/verbs.js (plain language, never tool names);
// the pass/tool counters come from the governor's state events, so the
// bounds the run actually lives under are the ones on screen.

import React, { useEffect, useState } from 'react';
import { nowVerb } from '../lib/verbs.js';

function useTick(active, ms = 1000) {
  const [, setN] = useState(0);
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => setN((n) => n + 1), ms);
    return () => clearInterval(id);
  }, [active, ms]);
}

function fmtElapsed(startedAt) {
  if (!startedAt) return '';
  const s = Math.max(0, Math.round((Date.now() - startedAt) / 1000));
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
}

export default function NowLine({ turn }) {
  const active = !!turn?.streaming;
  useTick(active);
  if (!active) return null;

  const waiting = turn.question?.status === 'open';
  const verb = waiting
    ? 'Waiting for an answer from the crew…'
    : nowVerb(turn);
  const elapsed = fmtElapsed(turn.startedAt);
  // Tool count is derived client-side from the timeline (always live);
  // governor state rides along only when the governor has spoken — it
  // emits on warnings and halts, not on every healthy step.
  const toolCount = (turn.timeline || []).filter((i) => i.kind === 'tool').length;
  const gov = turn.governor;
  const state = gov?.state || {};
  const budget = state.budget || {};
  const tok = (state.usage?.prompt_tokens || 0) + (state.usage?.completion_tokens || 0);
  const meta = [
    elapsed,
    toolCount ? `${toolCount} tool${toolCount === 1 ? '' : 's'}` : '',
    state.llm_turns != null && budget.max_llm_turns
      ? `pass ${Math.min(state.llm_turns + 1, budget.max_llm_turns)}/${budget.max_llm_turns}`
      : '',
    tok ? `${tok >= 1000 ? `${(tok / 1000).toFixed(1)}k` : tok} tok` : '',
  ].filter(Boolean);

  return (
    <div className={'now-line' + (waiting ? ' waiting' : '')} data-testid="now-line" role="status" aria-live="polite">
      <span className="now-spin" aria-hidden="true">✳</span>
      <span className="now-verb">{verb}</span>
      {meta.length ? <span className="now-meta">· {meta.join(' · ')}</span> : null}
      {gov?.action === 'warn' && gov?.reason ? (
        <span className="now-warn" title={gov.reason}>⚠ {gov.pathology || 'warning'}</span>
      ) : null}
    </div>
  );
}
