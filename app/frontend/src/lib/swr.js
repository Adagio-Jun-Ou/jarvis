// swr.js — a tiny per-session stale-while-revalidate cache for the
// read-only windows (inbox / tasks / briefs / files / skills).
//
// Semantics: the first fetch for a key populates the cache; later
// mounts render the cached value instantly and revalidate in the
// background.  Entries older than `maxAge` are refetched before
// being trusted.  In-flight requests are shared (dedup) and aborted
// via the caller's AbortSignal only when no one else is waiting.

import { useCallback, useEffect, useRef, useState } from "react";

const cache = new Map(); // key -> { data, ts }
const inflight = new Map(); // key -> Promise

export const DEFAULT_MAX_AGE = 30_000;

export function clearCache() {
  cache.clear();
  inflight.clear();
}

export function getCached(key) {
  return cache.get(key)?.data;
}

export function setCached(key, data) {
  cache.set(key, { data, ts: Date.now() });
}

export function isFresh(key, maxAge = DEFAULT_MAX_AGE) {
  const e = cache.get(key);
  return !!e && Date.now() - e.ts < maxAge;
}

/**
 * fetchSWR(key, fetcher, { maxAge }) -> Promise<data>
 * Returns fresh cache immediately; otherwise dedups concurrent
 * fetchers for the same key and caches the result.
 */
export function fetchSWR(key, fetcher, { maxAge = DEFAULT_MAX_AGE } = {}) {
  if (isFresh(key, maxAge)) return Promise.resolve(cache.get(key).data);
  if (inflight.has(key)) return inflight.get(key);
  const p = Promise.resolve()
    .then(fetcher)
    .then((data) => {
      setCached(key, data);
      inflight.delete(key);
      return data;
    })
    .catch((e) => {
      inflight.delete(key);
      throw e;
    });
  inflight.set(key, p);
  return p;
}

/**
 * useResource(key, fetcher) — React hook.
 *
 * `fetcher(signal)` must return a Promise of data.  The hook:
 *   - serves stale cache instantly (SWR) while revalidating,
 *   - exposes { data, error, loading, retry },
 *   - aborts the in-flight fetch when the component unmounts
 *     (window closed) via AbortController.
 */
export function useResource(key, fetcher, { maxAge = DEFAULT_MAX_AGE } = {}) {
  const stale = getCached(key);
  const [data, setData] = useState(stale);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(stale === undefined);
  const [nonce, setNonce] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    // A null/undefined key means "not ready to fetch" (e.g. the session
    // token hasn't arrived yet) — render loading and do nothing.
    if (key == null) return undefined;
    const ctrl = new AbortController();
    let alive = true;
    const cached = getCached(key);
    if (cached !== undefined) {
      setData(cached);
      setLoading(false);
      if (isFresh(key, maxAge) && nonce === 0) return () => ctrl.abort();
    } else {
      setLoading(true);
    }
    setError(null);
    fetchSWR(key, () => fetcherRef.current(ctrl.signal), { maxAge })
      .then((d) => { if (alive) { setData(d); setLoading(false); } })
      .catch((e) => {
        if (!alive || e?.name === "AbortError") return;
        setError(e);
        setLoading(false);
      });
    return () => {
      alive = false;
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, nonce]);

  const retry = useCallback(() => {
    cache.delete(key);
    setNonce((n) => n + 1);
  }, [key]);

  return { data, error, loading, retry };
}
