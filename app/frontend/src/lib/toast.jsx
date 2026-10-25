// toast.js — a featherweight toast bus + renderer.  No portals, no
// deps: `toast("message")` from anywhere; <Toasts/> (mounted once in
// Desktop) renders and auto-dismisses them.

import React, { useEffect, useState } from "react";

const listeners = new Set();
let seq = 0;

export function toast(message, kind = "error", ttl = 5000) {
  const t = { id: ++seq, message: String(message), kind, ttl };
  listeners.forEach((fn) => fn(t));
  return t.id;
}

export function Toasts() {
  const [items, setItems] = useState([]);
  useEffect(() => {
    const onToast = (t) => {
      setItems((xs) => [...xs.slice(-3), t]);
      setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== t.id)), t.ttl);
    };
    listeners.add(onToast);
    return () => listeners.delete(onToast);
  }, []);
  if (!items.length) return null;
  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {items.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind}`}>
          {t.kind === "error" ? "⚠ " : ""}{t.message}
          <button
            className="toast-close"
            aria-label="Dismiss"
            onClick={() => setItems((xs) => xs.filter((x) => x.id !== t.id))}
          >×</button>
        </div>
      ))}
    </div>
  );
}
