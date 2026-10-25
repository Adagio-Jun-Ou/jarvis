import React, { useEffect, useState } from 'react';

// Collapsible "thinking" panel rendered inline in the transcript when
// show_reasoning is on and the stream emitted reasoning events.
export default function ReasoningPanel({ events }) {
  const [open, setOpen] = useState(true);
  useEffect(() => { if (events?.length) setOpen(true); }, [events?.length]);
  if (!events || !events.length) return null;
  return (
    <details className="reasoning-panel" open={open} onToggle={(e) => setOpen(e.target.open)}>
      <summary>{events.length} step{events.length > 1 ? 's' : ''} of thinking</summary>
      {events.map((e, i) => (
        <p key={i} className="r-line">{e.phase ? `${e.phase}: ` : ''}{e.text}</p>
      ))}
    </details>
  );
}
