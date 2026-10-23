// Presence — the lifelike ink-ring avatar.
//
// A ring of ink that behaves like the agent: still when idle, slowly
// circling while it thinks, circling fast with a beating core while a
// tool runs, rippling outward as it speaks, and holding an amber
// breath when it is waiting on a person. One glyph that answers "is
// it alive, and what is it doing?" before a single word is read —
// the presence layer the voice-assistant orbs established, done here
// with pure SVG + CSS so it costs nothing and honors reduced-motion.
//
// The states are DERIVED from the same turn object the NowLine reads,
// so the avatar can never disagree with the status line beside it.

import React from 'react';

/** The presence state for a turn: idle | thinking | working | speaking | waiting. */
export function presenceOf(turn) {
  if (!turn) return 'idle';
  if (turn.question?.status === 'open') return 'waiting';
  if (!turn.streaming) return 'idle';
  const tl = turn.timeline || [];
  const last = tl[tl.length - 1];
  if (last && last.kind === 'tool' && last.result == null) return 'working';
  if ((turn.content || '').trim()) return 'speaking';
  return 'thinking';
}

export default function Presence({ turn, size = 26 }) {
  const state = presenceOf(turn);
  return (
    <span
      className={`presence is-${state}`}
      data-state={state}
      data-testid="presence"
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <svg viewBox="0 0 32 32" width={size} height={size} focusable="false">
        {/* the still water: a faint full circle the ink travels on */}
        <circle className="pr-track" cx="16" cy="16" r="11.5" />
        {/* the ink: an arc whose speed and length are the mood */}
        <circle className="pr-ink" cx="16" cy="16" r="11.5" />
        {/* the ripple: only visible while speaking */}
        <circle className="pr-ripple" cx="16" cy="16" r="11.5" />
        {/* the core: the drop of ink itself */}
        <circle className="pr-core" cx="16" cy="16" r="4" />
      </svg>
    </span>
  );
}
