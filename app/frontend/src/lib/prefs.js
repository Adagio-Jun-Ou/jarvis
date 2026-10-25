// prefs.js — per-user preferences, server-backed with a local cache.
//
// The server is the source of truth so a person's workspace follows them
// across devices. localStorage is a write-through cache read synchronously
// at boot, purely so the first paint has the right theme and density
// instead of flashing the default and correcting itself.
//
// Genuinely device-local state (panel widths, last open tab) never comes
// here — it belongs in localStorage alone.

const CACHE_KEY = 'alex-prefs';

export const FALLBACK = {
  theme: 'dark',
  density: 'comfortable',
  autonomy: 'co-pilot',
  default_model: '',
  pinned_tabs: ['chat', 'board'],
  pinned_tools: [],
  custom_instructions: '',
  show_reasoning: false,
  notify_on: ['run_finished', 'needs_approval'],
  reduce_motion: false,
};

/** Read the cache synchronously — used before the first network round-trip. */
export function cached() {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return { ...FALLBACK };
    return { ...FALLBACK, ...JSON.parse(raw) };
  } catch {
    return { ...FALLBACK };
  }
}

function writeCache(prefs) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(prefs));
  } catch { /* quota or private mode — the server still has them */ }
}

/** Apply presentation preferences to the document. */
export function applyPrefs(prefs) {
  const root = document.documentElement;
  root.setAttribute('data-theme', prefs.theme || 'dark');
  root.setAttribute('data-density', prefs.density || 'comfortable');
  if (prefs.reduce_motion) root.setAttribute('data-reduce-motion', 'true');
  else root.removeAttribute('data-reduce-motion');
}

export async function loadPrefs(authHeaders, fetchFn = fetch) {
  try {
    const r = await fetchFn('/api/preferences', { headers: authHeaders() });
    if (!r.ok) return { preferences: cached(), pinned: [], options: {} };
    const body = await r.json();
    writeCache(body.preferences);
    applyPrefs(body.preferences);
    return body;
  } catch {
    return { preferences: cached(), pinned: [], options: {} };
  }
}

/**
 * Save changes. Returns `{applied, refused}` — a key pinned by workspace
 * policy comes back in `refused` so the UI can say so rather than letting
 * a setting appear to save and then revert.
 */
export async function savePrefs(changes, authHeaders, fetchFn = fetch) {
  // Optimistic: apply presentation immediately so the UI feels instant,
  // then reconcile with whatever the server actually accepted.
  const optimistic = { ...cached(), ...changes };
  applyPrefs(optimistic);
  writeCache(optimistic);
  try {
    const r = await fetchFn('/api/preferences', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(changes),
    });
    if (!r.ok) return { applied: {}, refused: {} };
    const body = await r.json();
    if (Object.keys(body.refused || {}).length) {
      // Something did not stick — re-sync from the server so the UI shows
      // the truth rather than the optimistic guess.
      await loadPrefs(authHeaders, fetchFn);
    }
    return body;
  } catch {
    return { applied: {}, refused: {} };
  }
}
