// The design system's structural invariants.
//
// These are not pixel tests — they hold the two properties that rot
// silently: the light theme actually existing (the toggle sets
// data-theme on the root; for a long time nothing read it, so the
// switch flipped an attribute into the void), and the type scale being
// consumed by token rather than re-accumulating magic sizes.

import { it, expect, describe } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const css = readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), '..', 'styles.css'),
  'utf8'
);

describe('the light theme is real', () => {
  it('defines a [data-theme="light"] block', () => {
    expect(css).toMatch(/\[data-theme="light"\]/);
  });

  it('redefines every core surface and text token for light', () => {
    const block = css.split('[data-theme="light"]')[1].split('\n}')[0];
    for (const token of [
      '--bg:', '--panel:', '--card:', '--card-2:', '--hairline:',
      '--text:', '--text-2:', '--text-faint:', '--accent:',
      '--hover-wash:', '--scrim:', '--shadow-soft:',
    ]) {
      expect(block, `light theme is missing ${token}`).toContain(token);
    }
  });

  it('flips color-scheme so form controls follow', () => {
    const block = css.split('[data-theme="light"]')[1].split('\n}')[0];
    expect(block).toContain('color-scheme: light');
  });

  it('the prefs module actually sets data-theme on the root', () => {
    const prefs = readFileSync(
      join(dirname(fileURLToPath(import.meta.url)), '..', 'lib', 'prefs.js'),
      'utf8'
    );
    expect(prefs).toMatch(/setAttribute\(['"]data-theme['"]/);
  });
});

describe('interaction layers are theme-aware', () => {
  it('no literal white/black alpha outside the token blocks', () => {
    // A hover of rgba(255,255,255,.05) is invisible on a white surface —
    // every wash, scrim and micro-hairline must go through an alias the
    // light theme redefines.
    const lines = css.split('\n');
    let inTokens = false;
    const offenders = [];
    lines.forEach((ln, i) => {
      if (/^:root|^\[data-theme/.test(ln)) inTokens = true;
      else if (inTokens && ln.startsWith('}')) inTokens = false;
      else if (!inTokens && /rgba\(\s*(255,\s*255,\s*255|0,\s*0,\s*0)/.test(ln)) {
        offenders.push(`${i + 1}: ${ln.trim()}`);
      }
    });
    expect(offenders, offenders.join('\n')).toEqual([]);
  });
});

describe('the type scale is the spine', () => {
  it('defines the seven named steps', () => {
    for (const step of ['--fs-3xs', '--fs-2xs', '--fs-xs', '--fs-sm',
                        '--fs-base', '--fs-md', '--fs-lg', '--fs-xl']) {
      expect(css).toContain(`${step}:`);
    }
  });

  it('components consume the scale, not magic pixel sizes', () => {
    // The two grandfathered display sizes are component-level; anything
    // beyond that count means sizes are re-accumulating.
    const literals = css.match(/font-size:\s*[0-9.]+px/g) || [];
    expect(
      literals.length,
      `literal font-sizes creeping back in: ${literals.join(', ')}`
    ).toBeLessThanOrEqual(2);
  });
});
