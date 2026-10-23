// Settings.jsx — the eight knobs worth having.
//
// Deliberately short. Most people never open settings, so the default is
// the real product; every control here changes something a daily user
// actually feels. A value an administrator has pinned renders as governed
// and disabled, with the reason shown — a setting that silently refuses
// to save is worse than one that says no.

import React, { useCallback, useEffect, useState } from 'react';
import { loadPrefs, savePrefs } from '../lib/prefs.js';
import { toast } from '../lib/toast.jsx';

const AUTONOMY_BLURB = {
  suggest: 'Alex proposes; you run everything.',
  'co-pilot': 'Alex acts on low-risk steps and asks before the rest.',
  autopilot: 'Alex runs the whole task, stopping only at irreversible steps.',
};

function Row({ label, hint, pinned, children }) {
  return (
    <div className={'st-row' + (pinned ? ' pinned' : '')}>
      <div className="st-label">
        <span>{label}</span>
        {hint ? <span className="st-hint">{hint}</span> : null}
        {pinned ? (
          <span className="st-pin" title="Fixed by workspace policy">
            set by your workspace
          </span>
        ) : null}
      </div>
      <div className="st-control">{children}</div>
    </div>
  );
}

export default function Settings({ session, onClose, onChanged }) {
  const authHeaders = useCallback(
    () => session?.authHeaders?.() || {},
    [session]
  );
  const [prefs, setPrefs] = useState(null);
  const [pinned, setPinned] = useState([]);
  const [options, setOptions] = useState({});
  const [instructions, setInstructions] = useState('');

  useEffect(() => {
    let cancelled = false;
    loadPrefs(authHeaders).then((body) => {
      if (cancelled) return;
      setPrefs(body.preferences);
      setPinned(body.pinned || []);
      setOptions(body.options || {});
      setInstructions(body.preferences.custom_instructions || '');
    });
    return () => { cancelled = true; };
  }, [authHeaders]);

  const update = async (changes) => {
    setPrefs((p) => ({ ...p, ...changes }));
    const out = await savePrefs(changes, authHeaders);
    const refused = Object.entries(out.refused || {});
    if (refused.length) {
      const [key, why] = refused[0];
      toast(`${key} was not changed — ${why}.`);
      const body = await loadPrefs(authHeaders);
      setPrefs(body.preferences);
      setPinned(body.pinned || []);
    }
    onChanged?.();
  };

  if (!prefs) {
    return (
      <div className="st-backdrop" role="presentation">
        <div className="st" role="dialog" aria-modal="true" aria-label="Settings">
          <div className="bd-empty">Loading…</div>
        </div>
      </div>
    );
  }

  const isPinned = (k) => pinned.includes(k);

  return (
    <div className="st-backdrop" onMouseDown={onClose} role="presentation">
      <div
        className="st"
        role="dialog"
        aria-modal="true"
        aria-label="Workspace settings"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="st-head">
          <h2>Your workspace</h2>
          <button type="button" className="bd-x" onClick={onClose} aria-label="Close settings">
            ×
          </button>
        </header>

        <div className="st-body">
          <Row label="Theme" pinned={isPinned('theme')}>
            <div className="st-segmented" role="group" aria-label="Theme">
              {(options.theme || ['dark', 'light']).map((t) => (
                <button
                  key={t}
                  type="button"
                  aria-pressed={prefs.theme === t}
                  className={prefs.theme === t ? 'active' : ''}
                  disabled={isPinned('theme')}
                  onClick={() => update({ theme: t })}
                >
                  {t}
                </button>
              ))}
            </div>
          </Row>

          <Row
            label="Density"
            hint="Compact fits more on screen."
            pinned={isPinned('density')}
          >
            <div className="st-segmented" role="group" aria-label="Density">
              {(options.density || ['comfortable', 'compact']).map((d) => (
                <button
                  key={d}
                  type="button"
                  aria-pressed={prefs.density === d}
                  className={prefs.density === d ? 'active' : ''}
                  disabled={isPinned('density')}
                  onClick={() => update({ density: d })}
                >
                  {d}
                </button>
              ))}
            </div>
          </Row>

          <Row
            label="Autonomy"
            hint={AUTONOMY_BLURB[prefs.autonomy]}
            pinned={isPinned('autonomy')}
          >
            <div className="st-segmented" role="group" aria-label="Autonomy">
              {(options.autonomy || ['suggest', 'co-pilot', 'autopilot']).map((a) => (
                <button
                  key={a}
                  type="button"
                  aria-pressed={prefs.autonomy === a}
                  className={prefs.autonomy === a ? 'active' : ''}
                  disabled={isPinned('autonomy')}
                  onClick={() => update({ autonomy: a })}
                >
                  {a}
                </button>
              ))}
            </div>
          </Row>

          <Row
            label="Show reasoning"
            hint="Display Alex's thinking as it works."
            pinned={isPinned('show_reasoning')}
          >
            <label className="st-switch">
              <input
                type="checkbox"
                checked={!!prefs.show_reasoning}
                disabled={isPinned('show_reasoning')}
                onChange={(e) => update({ show_reasoning: e.target.checked })}
              />
              <span />
            </label>
          </Row>

          <Row
            label="Reduce motion"
            hint="Turn off pulsing and animated reveals."
            pinned={isPinned('reduce_motion')}
          >
            <label className="st-switch">
              <input
                type="checkbox"
                checked={!!prefs.reduce_motion}
                disabled={isPinned('reduce_motion')}
                onChange={(e) => update({ reduce_motion: e.target.checked })}
              />
              <span />
            </label>
          </Row>

          <div className="st-row block">
            <div className="st-label">
              <span>Custom instructions</span>
              <span className="st-hint">
                How you like Alex to work. The constitution and mandatory
                practices always win.
              </span>
            </div>
            <textarea
              className="st-textarea"
              rows={4}
              value={instructions}
              maxLength={2000}
              disabled={isPinned('custom_instructions')}
              onChange={(e) => setInstructions(e.target.value)}
              onBlur={() =>
                instructions !== prefs.custom_instructions &&
                update({ custom_instructions: instructions })
              }
              placeholder="e.g. Answer in bullets. Always name the owner of a risk."
            />
            <div className="st-count">{instructions.length}/2000</div>
          </div>
        </div>

        <footer className="st-foot">
          <button
            type="button"
            className="btn"
            onClick={async () => {
              await fetch('/api/preferences/reset', {
                method: 'POST',
                headers: authHeaders(),
              });
              const body = await loadPrefs(authHeaders);
              setPrefs(body.preferences);
              setInstructions(body.preferences.custom_instructions || '');
              onChanged?.();
            }}
          >
            Reset to defaults
          </button>
          <button type="button" className="btn primary" onClick={onClose}>
            Done
          </button>
        </footer>
      </div>
    </div>
  );
}
