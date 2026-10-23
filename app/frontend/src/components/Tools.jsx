import React, { useMemo, useState } from 'react';
import { useResource } from '../lib/swr.js';
import { IconTools } from './icons.jsx';

// Map a tool name to a human category so the catalog reads like a
// capability sheet, not a flat dump.
const CATEGORIES = [
  { id: 'memory', label: 'Memory', match: (n) => /remember|recall|forget/.test(n) },
  { id: 'graph', label: 'Knowledge graph', match: (n) => /graph|edge/.test(n) },
  { id: 'email', label: 'Email', match: (n) => /email|inbox|mail|summarize/.test(n) },
  { id: 'tasks', label: 'Tasks & workspace', match: (n) => /task|routine|present|view|file/.test(n) },
  { id: 'compute', label: 'Python & compute', match: (n) => /python|bash|run_|compute|exec/.test(n) },
  { id: 'web', label: 'Web & egress', match: (n) => /browser|web|navigate|extract|api_/.test(n) },
];

function categorize(name) {
  const hit = CATEGORIES.find((c) => c.match(name.toLowerCase()));
  return hit ? hit.id : 'other';
}

export default function Tools({ session }) {
  const authHeaders = () => session?.authHeaders?.() || {};
  const { data, loading, error, retry } = useResource(
    session?.email ? `tools:${session.email}` : null,
    (signal) =>
      fetch('/api/tools', { signal, credentials: 'include', headers: authHeaders() })
        .then((r) => (r.ok ? r.json() : { tools: [] }))
  );
  const [q, setQ] = useState('');

  const tools = data?.tools || data || [];
  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return tools.filter(
      (t) =>
        !needle ||
        (t.name || '').toLowerCase().includes(needle) ||
        (t.description || '').toLowerCase().includes(needle)
    );
  }, [tools, q]);

  const groups = useMemo(() => {
    const g = {};
    for (const t of filtered) {
      const cat = categorize(t.name || '');
      (g[cat] ||= []).push(t);
    }
    return g;
  }, [filtered]);

  const ordered = [...CATEGORIES, { id: 'other', label: 'Other' }].filter(
    (c) => groups[c.id]?.length
  );

  return (
    <div className="page">
      <div className="page-head">
        <h2>Tools</h2>
        <p>
          The governed capabilities Alex can use on your behalf — every call is
          checked against the constitution and your approved permissions.
        </p>
      </div>
      <div className="tools-note">
        Model Context Protocol compatible — external MCP clients can consume this
        same manifest from <code>GET /api/tools</code>.
      </div>
      <input
        className="tools-search"
        placeholder="Search capabilities…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        aria-label="Search tools"
      />

      {loading && !tools.length && (
        <div className="tools-grid">
          {Array.from({ length: 6 }).map((_, i) => (
            <div className="card tool-card skeleton" key={i} style={{ height: 96 }} />
          ))}
        </div>
      )}
      {error && (
        <div className="error-inline">
          Couldn’t load the tool manifest.{' '}
          <button className="btn ghost" onClick={retry}>Retry</button>
        </div>
      )}
      {!loading && !tools.length && !error && (
        <div className="empty-state">
          <span className="glyph"><IconTools /></span>
          <h3>No tools available</h3>
          <p>Your account has no approved capabilities yet.</p>
        </div>
      )}

      {ordered.map((cat) => (
        <div className="tools-cat" key={cat.id}>
          <span className="section-label">{cat.label}</span>
          <div className="tools-grid">
            {groups[cat.id].map((t) => (
              <ToolCard key={t.name} tool={t} session={session} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function ToolCard({ tool, session }) {
  const schema = tool.inputSchema || tool.input_schema || {};
  const props = schema.properties || {};
  const required = new Set(schema.required || []);
  const keys = Object.keys(props);
  const runnable = required.size === 0; // safe to invoke with no input
  const [state, setState] = useState('idle'); // idle | running | done | error
  const [result, setResult] = useState(null);

  const run = async () => {
    setState('running');
    setResult(null);
    try {
      const r = await fetch(`/api/tools/${tool.name}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(session?.authHeaders?.() || {}) },
        body: JSON.stringify({ args: {} }),
      });
      const d = await r.json();
      setResult(d.result ?? d);
      setState(r.ok ? 'done' : 'error');
    } catch (e) {
      setResult({ error: String(e.message || e) });
      setState('error');
    }
  };

  return (
    <div className="card tool-card">
      <div className="tool-card-head">
        <h4>{tool.name}</h4>
        {runnable && (
          <button className="btn ghost tool-run" onClick={run} disabled={state === 'running'}>
            {state === 'running' ? 'Running…' : 'Run'}
          </button>
        )}
      </div>
      <p>{tool.description || 'No description.'}</p>
      {keys.length > 0 && (
        <details className="tool-schema">
          <summary>{keys.length} input{keys.length === 1 ? '' : 's'}</summary>
          {keys.map((k) => (
            <div className="tool-arg" key={k}>
              <code>{k}{required.has(k) && <span className="req-star">*</span>}</code>
              <span className="type">{props[k]?.type || 'any'}</span>
            </div>
          ))}
        </details>
      )}
      {result != null && (
        <pre className={'tool-result ' + state}>
          {JSON.stringify(result, null, 2).slice(0, 1200)}
        </pre>
      )}
    </div>
  );
}
