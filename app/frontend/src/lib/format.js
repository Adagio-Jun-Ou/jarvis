// format.js — tiny display helpers shared across views.

export function relTime(input) {
  if (!input) return '';
  const then = typeof input === 'number' ? input : Date.parse(input);
  if (Number.isNaN(then)) return String(input);
  const s = Math.round((Date.now() - then) / 1000);
  if (s < 45) return 'just now';
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  if (d < 7) return `${d}d ago`;
  return new Date(then).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function formatBytes(n) {
  if (n == null || Number.isNaN(n)) return '';
  if (n < 1024) return `${n} B`;
  const kb = n / 1024;
  if (kb < 1024) return `${kb.toFixed(kb < 10 ? 1 : 0)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

export function monogram(email) {
  if (!email) return '·';
  const name = email.split('@')[0] || email;
  const parts = name.split(/[.\-_]/).filter(Boolean);
  const a = parts[0]?.[0] || name[0] || '';
  const b = parts[1]?.[0] || name[1] || '';
  return (a + b).toUpperCase();
}
