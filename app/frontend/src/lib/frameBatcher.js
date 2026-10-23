// frameBatcher.js — coalesce high-frequency updates into one flush
// per animation frame.  Used for SSE token appends (local models can
// emit hundreds of tokens/sec — one setState per token melts the
// main thread) and for pointermove-driven drag/resize.

const raf =
  typeof requestAnimationFrame === "function"
    ? requestAnimationFrame
    : (cb) => setTimeout(cb, 16);
const caf =
  typeof cancelAnimationFrame === "function"
    ? cancelAnimationFrame
    : clearTimeout;

/**
 * createFrameBatcher(flush) -> { push, flushNow, cancel, pending }
 *
 * `push(item)` queues an item; `flush(items[])` is called at most
 * once per animation frame with everything queued since the last
 * flush.  `flushNow()` flushes synchronously (used on stream end so
 * the final tokens are never dropped).
 */
export function createFrameBatcher(flush) {
  let queue = [];
  let handle = null;

  const run = () => {
    handle = null;
    if (!queue.length) return;
    const items = queue;
    queue = [];
    flush(items);
  };

  return {
    push(item) {
      queue.push(item);
      if (handle == null) handle = raf(run);
    },
    flushNow() {
      if (handle != null) { caf(handle); handle = null; }
      run();
    },
    cancel() {
      if (handle != null) { caf(handle); handle = null; }
      queue = [];
    },
    pending: () => queue.length,
  };
}

/**
 * rafThrottle(fn) -> throttled fn that runs at most once per frame
 * with the latest arguments.  `.cancel()` drops any pending call.
 */
export function rafThrottle(fn) {
  let handle = null;
  let lastArgs = null;
  const throttled = (...args) => {
    lastArgs = args;
    if (handle == null) {
      handle = raf(() => {
        handle = null;
        fn(...lastArgs);
      });
    }
  };
  throttled.cancel = () => {
    if (handle != null) { caf(handle); handle = null; }
  };
  return throttled;
}
