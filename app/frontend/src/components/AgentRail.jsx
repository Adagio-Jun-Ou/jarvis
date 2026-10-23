import React from 'react';
import { IconClose } from './icons.jsx';
import { downloadArtifact } from '../lib/download.js';

// The right-hand Agent activity rail. It renders the live
// tool/reasoning/law_block/memory/artifact stream of the CURRENT turn
// as elegant timeline chips. `items` is a flat, ordered list built by
// Chat from the streaming turn.
export default function AgentRail({ items = [], busy, onCollapse, authHeaders }) {
  return (
    <aside className="agent-rail" aria-label="Agent activity">
      <div className="agent-rail-head">
        <span className="section-label">Agent activity</span>
        <button className="icon-btn-sm" aria-label="Collapse activity" onClick={onCollapse}>
          <IconClose width={16} height={16} />
        </button>
      </div>
      <div className="agent-rail-body">
        {!items.length && (
          <div className="rail-empty">
            {busy ? 'Alex is thinking…' : 'Tool calls, reasoning and governance checks appear here during a turn.'}
          </div>
        )}
        {items.map((it, i) => <RailChip key={i} item={it} authHeaders={authHeaders} />)}
      </div>
    </aside>
  );
}

function RailChip({ item, authHeaders }) {
  switch (item.kind) {
    case 'tool':
      return (
        <div className="rail-chip tool">
          <span className="glyph">▸</span>
          <span><b>{item.tool}</b>{item.result != null ? ' · done' : ' · running…'}</span>
        </div>
      );
    case 'reasoning':
      return <div className="rail-chip reasoning"><span className="glyph">…</span><span>{item.text}</span></div>;
    case 'memory':
      return <div className="rail-chip memory"><span className="glyph">✦</span><span>learned: <b>{item.fact}</b></span></div>;
    case 'law_block':
      return (
        <div className="rail-chip law">
          <span className="glyph">⚖</span>
          <span><b>Blocked</b> · {item.article || 'constitution'}{item.detail ? ` — ${item.detail}` : ''}</span>
        </div>
      );
    case 'artifact':
      if (item.artifactKind === 'tool_result' || String(item.path || '').startsWith('tool://')) {
        return item.preview ? (
          <div className="rail-chip result"><span className="glyph">→</span><span>{item.preview}</span></div>
        ) : null;
      }
      return (
        <div className="rail-chip">
          <span className="glyph">⬇</span>
          <button type="button" className="rail-dl" onClick={() => downloadArtifact(item.path, authHeaders)}>{String(item.path || '').split('/').pop()}</button>
        </div>
      );
    default:
      return null;
  }
}
