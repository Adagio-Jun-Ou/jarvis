/**
 * Tests for the SSE event reducer used by Chat.jsx.
 * `reduceTurn` is exported from the component so these tests exercise
 * the REAL production reducer (no local copy to drift).
 */
import { describe, it, expect } from 'vitest';
import { reduceTurn as applyEvent } from '../components/Chat.jsx';

const baseTurn = () => ({
  content: '', activity: [], reasoning: [], lawBlocks: [], artifacts: [],
  timeline: [], card: null, streaming: true, gotFirstEvent: false,
});

describe('applyEvent', () => {
  it('appends text', () => {
    const t = applyEvent(baseTurn(), { type: 'text', text: 'Hello' });
    expect(t.content).toBe('Hello');
  });

  it('appends reasoning', () => {
    const t = applyEvent(baseTurn(), { type: 'reasoning', phase: 'plan', text: 'x' });
    expect(t.reasoning).toEqual([{ type: 'reasoning', phase: 'plan', text: 'x' }]);
  });

  it('records tool then resolves the same tool (activity + timeline)', () => {
    // The same args object reference is used so the parser can
    // match the pre and post events for the same tool call.
    const args = { a: 1 };
    let t = applyEvent(baseTurn(), { type: 'tool', tool: 'x', args, result: null });
    t = applyEvent(t, { type: 'tool', tool: 'x', args, result: 'ok' });
    expect(t.activity[0].result).toBe('ok');
    expect(t.timeline).toHaveLength(1);
    expect(t.timeline[0]).toMatchObject({
      kind: 'tool', tool: 'x', args, result: 'ok',
    });
  });

  it('times each tool step so the execution log can show duration', () => {
    const args = { a: 1 };
    let t = applyEvent(baseTurn(), { type: 'tool', tool: 'x', args, result: null });
    const step = t.timeline[0];
    expect(step.startedAt).toBeGreaterThan(0);
    expect(step.endedAt).toBeUndefined();
    t = applyEvent(t, { type: 'tool', tool: 'x', args, result: 'ok' });
    expect(t.timeline[0].endedAt).toBeGreaterThanOrEqual(step.startedAt);
  });

  it('appends law_block and sets card from confirmation_required', () => {
    let t = applyEvent(baseTurn(), { type: 'law_block', article: 'A4', detail: 'no' });
    t = applyEvent(t, { type: 'confirmation_required', token: 'tok', summary: 'go' });
    expect(t.lawBlocks[0].article).toBe('A4');
    expect(t.card.token).toBe('tok');
    expect(t.timeline).toEqual([{ kind: 'law_block', article: 'A4', detail: 'no' }]);
  });

  it('records memory events on the timeline', () => {
    const t = applyEvent(baseTurn(), { type: 'memory', fact: 'user prefers dark mode' });
    expect(t.timeline).toEqual([{ kind: 'memory', fact: 'user prefers dark mode' }]);
  });

  it('records artifacts on the timeline', () => {
    const t = applyEvent(baseTurn(), { type: 'artifact', path: 'out/report.pdf', kind: 'pdf' });
    expect(t.artifacts).toHaveLength(1);
    expect(t.timeline).toEqual([{ kind: 'artifact', path: 'out/report.pdf', artifactKind: 'pdf' }]);
  });

  it('emits done', () => {
    const t = applyEvent(baseTurn(), { type: 'done' });
    expect(t.streaming).toBe(false);
  });

  it('records errors as timeline cards (not content)', () => {
    const t = applyEvent(baseTurn(), { type: 'error', code: 'E1', message: 'oops' });
    expect(t.content).toBe('');
    expect(t.timeline).toEqual([{ kind: 'error', message: 'oops', code: 'E1' }]);
  });

  it('flips gotFirstEvent on any event so the waiting indicator clears', () => {
    const t = applyEvent(baseTurn(), { type: 'reasoning', text: 'x' });
    expect(t.gotFirstEvent).toBe(true);
  });
});
