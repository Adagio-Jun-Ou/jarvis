import React, { Suspense, lazy, useEffect, useMemo, useState } from 'react';
import { useResource } from './lib/swr.js';
import { Toasts } from './lib/toast.jsx';
import { cached as cachedPrefs, applyPrefs, loadPrefs } from './lib/prefs.js';
import CommandPalette from './components/CommandPalette.jsx';
import AgentRail from './components/AgentRail.jsx';
import BriefsStrip from './components/BriefsStrip.jsx';
import {
  IconChat, IconEmail, IconSessions, IconFiles, IconBrowser,
  IconTools, IconTransparency, IconSpark, IconBoard, IconGuidance, IconOversight,
} from './components/icons.jsx';

import Chat from './components/Chat.jsx';
const Email = lazy(() => import('./components/Email.jsx'));
const Sessions = lazy(() => import('./components/Sessions.jsx'));
const Files = lazy(() => import('./components/Files.jsx'));
const Browser = lazy(() => import('./components/Browser.jsx'));
const Tools = lazy(() => import('./components/Tools.jsx'));
const Transparency = lazy(() => import('./components/Transparency.jsx'));
const Board = lazy(() => import('./components/Board.jsx'));
const Guidance = lazy(() => import('./components/Guidance.jsx'));
const Settings = lazy(() => import('./components/Settings.jsx'));
const Oversight = lazy(() => import('./components/Oversight.jsx'));
const Workflows = lazy(() => import('./components/Workflows.jsx'));

/* Three connected nodes — the flow-map in miniature. */
function IconWorkflows(p) {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <circle cx="5" cy="12" r="2.4" />
      <circle cx="19" cy="6" r="2.4" />
      <circle cx="19" cy="18" r="2.4" />
      <path d="M7.3 11.2 16.7 6.8M7.3 12.8l9.4 4.4" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Auth bootstrap (exported for tests)
// ---------------------------------------------------------------------------
const DEV_EMAIL_KEY = 'alex-dev-email';
const DEFAULT_DEV_EMAIL = 'demo@alex.dev';

export function buildAuthHeaders(token) {
  return token
    ? { Authorization: `Bearer ${token}`, 'X-Forwarded-Access-Token': token }
    : {};
}

export async function bootstrapSession(fetchFn = fetch) {
  let email = DEFAULT_DEV_EMAIL;
  try { email = localStorage.getItem(DEV_EMAIL_KEY) || DEFAULT_DEV_EMAIL; } catch { /* ignore */ }

  let token = '';
  try {
    const t = await fetchFn(`/api/dev/token?email=${encodeURIComponent(email)}`);
    if (t.ok) token = (await t.json()).token || '';
  } catch { /* production or backend down */ }

  const headers = buildAuthHeaders(token);
  let model = '';
  try {
    const me = await fetchFn('/api/me', { headers, credentials: 'include' });
    if (me.ok) {
      const body = await me.json();
      email = body.email || email;
      model = body.model || '';
    }
  } catch { /* keep fallback */ }

  return { email, token, model, authHeaders: () => buildAuthHeaders(token) };
}

function useAlexSession() {
  const [session, setSession] = useState({ email: '', token: '', authHeaders: () => ({}) });
  useEffect(() => {
    let cancelled = false;
    bootstrapSession().then((s) => { if (!cancelled) setSession(s); });
    return () => { cancelled = true; };
  }, []);
  return session;
}

// Live workspace clock — updates every second, HH:MM:SS.
function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/* Presentation comes from the user's preferences, which are server-backed
   so a workspace follows the person across devices. The cached copy is
   read synchronously at boot purely to avoid a flash of the wrong theme
   before the network round-trip lands. */
function usePreferences(session) {
  const [prefs, setPrefs] = useState(() => {
    const start = cachedPrefs();
    applyPrefs(start);
    return start;
  });

  const refresh = React.useCallback(() => {
    if (!session?.email) return;
    loadPrefs(session.authHeaders).then((body) => setPrefs(body.preferences));
  }, [session]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { applyPrefs(prefs); }, [prefs]);

  return [prefs, refresh];
}

function SunIcon(p) {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" {...p}>
      <circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19" />
    </svg>
  );
}
function GearIcon(p) {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.2.6.76 1 1.4 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function MoonIcon(p) {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" {...p}>
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Shell
// ---------------------------------------------------------------------------
const NAV = [
  { id: 'chat', label: 'Chat', Icon: IconChat },
  { id: 'board', label: 'Board', Icon: IconBoard },
  { id: 'guidance', label: 'Guidance', Icon: IconGuidance },
  { id: 'email', label: 'Email', Icon: IconEmail, badge: 'inbox' },
  { id: 'sessions', label: 'Sessions', Icon: IconSessions, badge: 'sessions' },
  { id: 'files', label: 'Files', Icon: IconFiles },
  { id: 'browser', label: 'Browser', Icon: IconBrowser },
  { id: 'tools', label: 'Tools', Icon: IconTools },
  { id: 'workflows', label: 'Workflows', Icon: IconWorkflows },
  { id: 'transparency', label: 'Transparency', Icon: IconTransparency },
  { id: 'oversight', label: 'Oversight', Icon: IconOversight, adminOnly: true },
];

const TITLES = {
  chat: 'Chat', board: 'Task board', guidance: 'Standing orders & best practices', email: 'Email', sessions: 'Sessions', files: 'Files',
  browser: 'Browser', tools: 'Tools', workflows: 'Workflows — the n8n automation arm',
  transparency: 'Transparency',
  oversight: 'Oversight — are the agents behaving?',
};

/**
 * Which view a user may actually occupy.
 *
 * Hiding an admin tab from the nav is not enough on its own: the initial
 * view comes from `?view=` in the URL, which is read before the role is
 * known, so a non-admin can land directly on an admin surface. The API
 * refuses and the surface renders a notice, but leaving someone parked
 * there teaches the same "permission errors are normal" lesson that
 * hiding the tab was meant to avoid.
 *
 * While the role is still loading the view is left alone, so an admin is
 * never bounced off their own surface by a slow request.
 */
export function allowedView(view, { isAdmin, roleLoaded }) {
  if (!roleLoaded) return view;
  const target = NAV.find((n) => n.id === view);
  return target?.adminOnly && !isAdmin ? 'chat' : view;
}

export default function App() {
  const session = useAlexSession();
  const clock = useClock();
  const [prefs, refreshPrefs] = usePreferences(session);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const theme = prefs.theme || 'dark';
  const toggleTheme = React.useCallback(() => {
    const next = theme === 'dark' ? 'light' : 'dark';
    import('./lib/prefs.js').then(({ savePrefs }) =>
      savePrefs({ theme: next }, session.authHeaders).then(refreshPrefs)
    );
  }, [theme, session, refreshPrefs]);
  const [view, setView] = useState(() => {
    try {
      const v = new URLSearchParams(window.location.search).get('view');
      return NAV.some((n) => n.id === v) ? v : 'chat';
    } catch { return 'chat'; }
  });
  const [sessionId, setSessionId] = useState('default');
  const [railItems, setRailItems] = useState([]);
  const [busy, setBusy] = useState(false);

  const authed = !!session.email;

  // Admin-only surfaces are hidden as well as refused. Showing a tab
  // that always 403s teaches people to ignore permission errors.
  const me = useResource(
    authed ? `security-me:${session.email}` : null,
    (signal) => fetch('/api/security/me', { signal, headers: session.authHeaders() })
      .then((r) => (r.ok ? r.json() : { role: 'member' })),
    { maxAge: 60000 }
  );
  const isAdmin = me.data?.role === 'admin';
  const nav = useMemo(
    () => NAV.filter((n) => !n.adminOnly || isAdmin),
    [isAdmin]
  );

  React.useEffect(() => {
    const next = allowedView(view, { isAdmin, roleLoaded: !!me.data });
    if (next !== view) setView(next);
  }, [me.data, isAdmin, view]);

  const inbox = useResource(
    authed ? `inbox-count:${session.email}` : null,
    (signal) => fetch('/api/inbox', { signal, headers: session.authHeaders() }).then((r) => (r.ok ? r.json() : { inbox: [] }))
  );
  const sess = useResource(
    authed ? `sessions-count:${session.email}` : null,
    (signal) => fetch('/api/sessions', { signal, headers: session.authHeaders() }).then((r) => (r.ok ? r.json() : { sessions: [] }))
  );
  const badges = {
    inbox: inbox.data?.inbox?.length || 0,
    sessions: sess.data?.sessions?.length || 0,
  };

  const conn = busy ? 'working' : 'ready';
  const connLabel = busy ? 'Working' : 'Ready';

  const openSession = (id) => { setSessionId(id); setView('chat'); };

  // Every command is keyed by a stable id, so renaming a label does not
  // orphan its place in the palette's recents.
  const commands = useMemo(() => {
    const cmds = nav.map((n) => ({
      id: `nav.${n.id}`,
      label: `Go to ${n.label}`,
      group: 'Navigate',
      keywords: [n.id],
      run: () => setView(n.id),
    }));
    cmds.push(
      {
        id: 'settings.open',
        label: 'Open settings',
        group: 'Workspace',
        hint: '⌘,',
        run: () => setSettingsOpen(true),
      },
      {
        id: 'settings.theme',
        label: `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`,
        group: 'Workspace',
        keywords: ['theme', 'dark', 'light'],
        run: toggleTheme,
      },
      {
        id: 'settings.density',
        label: `Use ${prefs.density === 'compact' ? 'comfortable' : 'compact'} density`,
        group: 'Workspace',
        keywords: ['density', 'compact', 'spacing'],
        run: () =>
          import('./lib/prefs.js').then(({ savePrefs }) =>
            savePrefs(
              {
                density:
                  prefs.density === 'compact' ? 'comfortable' : 'compact',
              },
              session.authHeaders
            ).then(refreshPrefs)
          ),
      },
      {
        id: 'session.new',
        label: 'Start a new chat session',
        group: 'Session',
        keywords: ['new', 'chat', 'session'],
        run: () => {
          setSessionId(`s-${Date.now()}`);
          setView('chat');
        },
      }
    );
    return cmds;
  }, [theme, prefs.density, session, refreshPrefs, toggleTheme, nav]);

  useEffect(() => {
    const onKey = (e) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      } else if (mod && e.key === ',') {
        e.preventDefault();
        setSettingsOpen(true);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const main = useMemo(() => {
    switch (view) {
      case 'chat':
        return (
          <div className="view">
            <div className="view-body">
              <BriefsStrip session={session} />
              <Chat
                sessionId={sessionId}
                model="auto"
                session={session}
                onActivity={setRailItems}
                onBusy={setBusy}
              />
            </div>
            <AgentRail items={railItems} busy={busy} onCollapse={() => {}} authHeaders={session.authHeaders} />
          </div>
        );
      case 'email': return <Email session={session} />;
      case 'sessions': return <Sessions session={session} onOpen={openSession} />;
      case 'files': return <Files session={session} />;
      case 'board': return <Board session={session} />;
      case 'workflows': return <Workflows session={session} />;
      case 'guidance': return <Guidance session={session} />;
      case 'browser': return <Browser session={session} />;
      case 'tools': return <Tools session={session} />;
      case 'oversight': return <Oversight session={session} />;
      case 'transparency': return <Transparency me={session.email} session={session} onClose={() => setView('chat')} />;
      default: return null;
    }
  }, [view, sessionId, session, railItems, busy]);

  if (!authed) {
    return (
      <div className="boot">
        <span className="brand-mark"><IconSpark /></span>
        <span>Connecting to Alex…</span>
      </div>
    );
  }

  const monogram = (session.email || '?').slice(0, 1).toUpperCase();

  return (
    <div className="shell">
      <nav className="nav" aria-label="Primary">
        <div className="brand">
          <span className="brand-mark"><IconSpark /></span>
          <span>
            <span className="brand-name">Alex</span>
            <span className="brand-sub">Jarvis · on this machine</span>
          </span>
        </div>
        <div className="nav-scroll">
          {nav.map(({ id, label, Icon, badge }) => (
            <button
              key={id}
              className={'nav-item' + (view === id ? ' active' : '')}
              onClick={() => setView(id)}
              aria-current={view === id ? 'page' : undefined}
            >
              <Icon />
              <span className="label">{label}</span>
              {badge && badges[badge] > 0 && <span className="nav-badge">{badges[badge]}</span>}
            </button>
          ))}
        </div>
        <div className="nav-user">
          <span className="avatar">{monogram}</span>
          <span className="who">
            <b>{session.email}</b>
            <span>Signed in</span>
          </span>
        </div>
      </nav>

      <div className="main">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{TITLES[view]}</h1>
            <span className="crumb">{view === 'chat' ? `session · ${sessionId}` : 'Alex workspace'}</span>
          </div>
          <div className="topbar-spacer" />
          <span className="clock" aria-label="Workspace time" title="Workspace time">{clock}</span>
          <span className="model-pill"><span className="dot" /><b>{session.model || 'model'}</b> · local</span>
          <span className={'conn ' + conn}><span className="dot" />{connLabel}</span>
          <button
            className="icon-btn theme-toggle"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
          </button>
          <button
            className="icon-btn"
            onClick={() => setSettingsOpen(true)}
            aria-label="Workspace settings"
            title="Settings (⌘,)"
          >
            <GearIcon />
          </button>
        </header>

        <Suspense fallback={<div className="page"><div className="skeleton" style={{ height: 200 }} /></div>}>
          {main}
        </Suspense>
      </div>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        commands={commands}
      />
      {settingsOpen ? (
        <Suspense fallback={null}>
          <Settings
            session={session}
            onClose={() => setSettingsOpen(false)}
            onChanged={refreshPrefs}
          />
        </Suspense>
      ) : null}
      <Toasts />
    </div>
  );
}
