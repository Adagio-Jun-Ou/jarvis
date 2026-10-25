// markdown.jsx — a tiny, dependency-free markdown renderer for chat.
//
// Supports the subset LLM replies actually use: **bold**, *italic*,
// `inline code`, ``` fenced code blocks, # headings, - / * bullet
// lists, 1. numbered lists. Everything is emitted as React elements
// (never innerHTML), so untrusted model output cannot inject markup.

import React from 'react';

const INLINE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;

export function renderInline(text) {
  const parts = String(text ?? '').split(INLINE);
  return parts.map((p, i) => {
    if (p.startsWith('**') && p.endsWith('**') && p.length > 4) {
      return <strong key={i}>{p.slice(2, -2)}</strong>;
    }
    if (p.startsWith('`') && p.endsWith('`') && p.length > 2) {
      return <code key={i}>{p.slice(1, -1)}</code>;
    }
    if (p.startsWith('*') && p.endsWith('*') && p.length > 2) {
      return <em key={i}>{p.slice(1, -1)}</em>;
    }
    return p;
  });
}

export function renderMarkdown(text) {
  const lines = String(text ?? '').split('\n');
  const out = [];
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trimStart().startsWith('```')) {
      const code = [];
      i += 1;
      while (i < lines.length && !lines[i].trimStart().startsWith('```')) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1; // closing fence
      out.push(<pre key={key++} className="md-code"><code>{code.join('\n')}</code></pre>);
      continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      const Tag = `h${Math.min(h[1].length + 2, 6)}`; // # -> h3 in chat scale
      out.push(<Tag key={key++} className="md-h">{renderInline(h[2])}</Tag>);
      i += 1;
      continue;
    }
    const isBullet = (l) => /^\s*[-*]\s+/.test(l);
    const isNum = (l) => /^\s*\d+[.)]\s+/.test(l);
    if (isBullet(line) || isNum(line)) {
      const ordered = isNum(line);
      const match = ordered ? isNum : isBullet;
      const items = [];
      while (i < lines.length && match(lines[i])) {
        items.push(lines[i].replace(/^\s*([-*]|\d+[.)])\s+/, ''));
        i += 1;
      }
      const L = ordered ? 'ol' : 'ul';
      out.push(
        React.createElement(
          L,
          { key: key++, className: 'md-list' },
          items.map((it, j) => <li key={j}>{renderInline(it)}</li>)
        )
      );
      continue;
    }
    if (line.trim() === '') {
      i += 1;
      continue;
    }
    // Paragraph: greedily absorb consecutive plain lines.
    const para = [line];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !lines[i].trimStart().startsWith('```') &&
      !/^(#{1,4})\s+/.test(lines[i]) &&
      !isBullet(lines[i]) &&
      !isNum(lines[i])
    ) {
      para.push(lines[i]);
      i += 1;
    }
    out.push(<p key={key++}>{renderInline(para.join(' '))}</p>);
  }
  return out;
}
