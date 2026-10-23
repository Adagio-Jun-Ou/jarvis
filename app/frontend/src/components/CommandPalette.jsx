// CommandPalette.jsx — ⌘K, the keyboard route to everything.
//
// Built as a proper combobox: the input keeps DOM focus and
// `aria-activedescendant` moves a virtual highlight over the listbox, so
// screen readers announce the highlighted option without focus ever
// leaving the text field. Arrow keys, Enter and Escape behave the way
// they do in every other palette, because muscle memory is the point.
//
// Commands are keyed by a stable id rather than their label, so a
// renamed command keeps its place in recents.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const RECENTS_KEY = 'alex-palette-recents';
const MAX_RECENTS = 6;

function loadRecents() {
  try {
    const raw = localStorage.getItem(RECENTS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function noteUse(id) {
  try {
    const next = [id, ...loadRecents().filter((x) => x !== id)].slice(0, MAX_RECENTS);
    localStorage.setItem(RECENTS_KEY, JSON.stringify(next));
  } catch { /* ignore */ }
}

/** Subsequence match, so "gob" finds "Go to Board". */
export function fuzzyScore(needle, haystack) {
  const n = needle.toLowerCase().trim();
  const h = haystack.toLowerCase();
  if (!n) return 0;
  if (h.includes(n)) return 100 - h.indexOf(n);
  let i = 0;
  let score = 0;
  for (const ch of h) {
    if (ch === n[i]) {
      i += 1;
      score += 1;
      if (i === n.length) return score;
    }
  }
  return -1;
}

export function rankCommands(commands, query, recents = []) {
  if (!query.trim()) {
    const recentSet = new Map(recents.map((id, i) => [id, recents.length - i]));
    return [...commands].sort(
      (a, b) => (recentSet.get(b.id) || 0) - (recentSet.get(a.id) || 0)
    );
  }
  return commands
    .map((c) => ({
      c,
      s: Math.max(
        fuzzyScore(query, c.label),
        fuzzyScore(query, `${c.group || ''} ${c.label}`) - 5,
        ...(c.keywords || []).map((k) => fuzzyScore(query, k) - 10)
      ),
    }))
    .filter((x) => x.s >= 0)
    .sort((a, b) => b.s - a.s)
    .map((x) => x.c);
}

export default function CommandPalette({ open, onClose, commands }) {
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);
  const recents = useMemo(() => (open ? loadRecents() : []), [open]);

  const results = useMemo(
    () => rankCommands(commands || [], query, recents).slice(0, 12),
    [commands, query, recents]
  );

  useEffect(() => {
    if (!open) return;
    setQuery('');
    setActive(0);
    // Focus after paint so the dialog is in the tree first.
    const id = requestAnimationFrame(() => inputRef.current?.focus());
    return () => cancelAnimationFrame(id);
  }, [open]);

  useEffect(() => setActive(0), [query]);

  useEffect(() => {
    const el = listRef.current?.querySelector('[data-active="true"]');
    el?.scrollIntoView?.({ block: 'nearest' });
  }, [active, results]);

  const run = useCallback(
    (cmd) => {
      if (!cmd) return;
      noteUse(cmd.id);
      onClose?.();
      // Let the dialog close before the command changes the view.
      setTimeout(() => cmd.run?.(), 0);
    },
    [onClose]
  );

  const onKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((i) => (results.length ? (i + 1) % results.length : 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((i) => (results.length ? (i - 1 + results.length) % results.length : 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      run(results[active]);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose?.();
    }
  };

  if (!open) return null;

  return (
    <div className="cp-backdrop" onMouseDown={onClose} role="presentation">
      <div
        className="cp"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          className="cp-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a command…"
          aria-label="Type a command"
          role="combobox"
          aria-expanded="true"
          aria-controls="cp-list"
          aria-autocomplete="list"
          aria-activedescendant={
            results[active] ? `cp-opt-${results[active].id}` : undefined
          }
        />
        <ul className="cp-list" id="cp-list" role="listbox" ref={listRef}>
          {results.map((c, i) => (
            <li
              key={c.id}
              id={`cp-opt-${c.id}`}
              role="option"
              aria-selected={i === active}
              data-active={i === active ? 'true' : 'false'}
              className={'cp-opt' + (i === active ? ' active' : '')}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                e.preventDefault();
                run(c);
              }}
            >
              <span className="cp-opt-label">{c.label}</span>
              {c.group ? <span className="cp-opt-group">{c.group}</span> : null}
              {c.hint ? <kbd className="cp-opt-hint">{c.hint}</kbd> : null}
            </li>
          ))}
          {!results.length ? (
            <li className="cp-empty" role="option" aria-selected="false">
              Nothing matches “{query}”.
            </li>
          ) : null}
        </ul>
        <footer className="cp-foot">
          <kbd>↑</kbd><kbd>↓</kbd> to move · <kbd>↵</kbd> to run ·{' '}
          <kbd>esc</kbd> to close
        </footer>
      </div>
    </div>
  );
}
