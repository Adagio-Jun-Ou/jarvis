// Tests for lib/markdown.jsx — the safe chat markdown renderer.
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import React from 'react';
import { renderMarkdown } from '../lib/markdown.jsx';

const mount = (text) => render(<div>{renderMarkdown(text)}</div>).container;

describe('renderMarkdown', () => {
  it('renders bold and inline code', () => {
    const c = mount('**Insight:** run `pytest` now');
    expect(c.querySelector('strong')?.textContent).toBe('Insight:');
    expect(c.querySelector('code')?.textContent).toBe('pytest');
  });

  it('renders bullet lists', () => {
    const c = mount('- one\n- two\n- three');
    expect(c.querySelectorAll('ul li')).toHaveLength(3);
  });

  it('renders numbered lists', () => {
    const c = mount('1. first\n2. second');
    expect(c.querySelectorAll('ol li')).toHaveLength(2);
  });

  it('renders fenced code blocks verbatim', () => {
    const c = mount('```\nprint("hi")\n```');
    expect(c.querySelector('pre code')?.textContent).toBe('print("hi")');
  });

  it('never injects HTML from model output', () => {
    const c = mount('<img src=x onerror=alert(1)> **bold**');
    expect(c.querySelector('img')).toBeNull();
    expect(c.textContent).toContain('<img src=x onerror=alert(1)>');
  });

  it('renders headings at chat scale', () => {
    const c = mount('# Title\nbody');
    expect(c.querySelector('h3')?.textContent).toBe('Title');
  });

  it('handles empty input', () => {
    expect(renderMarkdown('')).toEqual([]);
  });
});
