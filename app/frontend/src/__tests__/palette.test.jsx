// Tests for the command palette and the preference cache.
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import React from 'react';
import CommandPalette, { fuzzyScore, rankCommands } from '../components/CommandPalette.jsx';
import { cached, applyPrefs, FALLBACK } from '../lib/prefs.js';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
});

const CMDS = [
  { id: 'nav.chat', label: 'Go to Chat', group: 'Navigate', run: vi.fn() },
  { id: 'nav.board', label: 'Go to Board', group: 'Navigate', run: vi.fn() },
  { id: 'settings.open', label: 'Open settings', group: 'Workspace', run: vi.fn() },
];

describe('fuzzyScore', () => {
  it('prefers a direct substring', () => {
    expect(fuzzyScore('board', 'Go to Board')).toBeGreaterThan(0);
  });

  it('matches a subsequence', () => {
    expect(fuzzyScore('gob', 'Go to Board')).toBeGreaterThanOrEqual(0);
  });

  it('rejects a non-match', () => {
    expect(fuzzyScore('zzz', 'Go to Board')).toBe(-1);
  });

  it('an empty needle is neutral', () => {
    expect(fuzzyScore('', 'anything')).toBe(0);
  });
});

describe('rankCommands', () => {
  it('filters to matches', () => {
    const out = rankCommands(CMDS, 'board');
    expect(out.map((c) => c.id)).toEqual(['nav.board']);
  });

  it('surfaces recents first when there is no query', () => {
    const out = rankCommands(CMDS, '', ['settings.open']);
    expect(out[0].id).toBe('settings.open');
  });

  it('returns everything for an empty query', () => {
    expect(rankCommands(CMDS, '')).toHaveLength(3);
  });

  it('matches on keywords too', () => {
    const cmds = [{ id: 'x', label: 'Switch to light mode', keywords: ['theme'] }];
    expect(rankCommands(cmds, 'theme')).toHaveLength(1);
  });
});

describe('CommandPalette', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <CommandPalette open={false} commands={CMDS} onClose={() => {}} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('is a combobox driving a listbox', () => {
    render(<CommandPalette open commands={CMDS} onClose={() => {}} />);
    const input = screen.getByRole('combobox');
    expect(input).toHaveAttribute('aria-controls', 'cp-list');
    // Focus stays in the input; the highlight is virtual.
    expect(input).toHaveAttribute('aria-activedescendant', 'cp-opt-nav.chat');
    expect(screen.getByRole('listbox')).toBeInTheDocument();
  });

  it('filters as you type', () => {
    render(<CommandPalette open commands={CMDS} onClose={() => {}} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'board' } });
    expect(screen.getAllByRole('option')).toHaveLength(1);
    expect(screen.getByText('Go to Board')).toBeInTheDocument();
  });

  it('moves the virtual highlight with the arrow keys', () => {
    render(<CommandPalette open commands={CMDS} onClose={() => {}} />);
    const input = screen.getByRole('combobox');
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    expect(input).toHaveAttribute('aria-activedescendant', 'cp-opt-nav.board');
    fireEvent.keyDown(input, { key: 'ArrowUp' });
    expect(input).toHaveAttribute('aria-activedescendant', 'cp-opt-nav.chat');
  });

  it('wraps around at the ends', () => {
    render(<CommandPalette open commands={CMDS} onClose={() => {}} />);
    const input = screen.getByRole('combobox');
    fireEvent.keyDown(input, { key: 'ArrowUp' });
    expect(input).toHaveAttribute('aria-activedescendant', 'cp-opt-settings.open');
  });

  it('runs the highlighted command on Enter and closes', async () => {
    const onClose = vi.fn();
    const run = vi.fn();
    render(
      <CommandPalette
        open
        commands={[{ id: 'x', label: 'Do it', run }]}
        onClose={onClose}
      />
    );
    fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' });
    expect(onClose).toHaveBeenCalled();
    await vi.waitFor(() => expect(run).toHaveBeenCalled());
  });

  it('closes on Escape without running anything', () => {
    const onClose = vi.fn();
    const run = vi.fn();
    render(
      <CommandPalette open commands={[{ id: 'x', label: 'Do it', run }]} onClose={onClose} />
    );
    fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
    expect(run).not.toHaveBeenCalled();
  });

  it('says so when nothing matches', () => {
    render(<CommandPalette open commands={CMDS} onClose={() => {}} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'zzzz' } });
    expect(screen.getByText(/Nothing matches/)).toBeInTheDocument();
  });

  it('remembers what you ran', async () => {
    const onClose = vi.fn();
    render(<CommandPalette open commands={CMDS} onClose={onClose} />);
    fireEvent.keyDown(screen.getByRole('combobox'), { key: 'ArrowDown' });
    fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' });
    expect(JSON.parse(localStorage.getItem('alex-palette-recents'))).toContain(
      'nav.board'
    );
  });
});

describe('preference cache', () => {
  beforeEach(() => localStorage.clear());

  it('falls back to defaults with no cache', () => {
    expect(cached().theme).toBe(FALLBACK.theme);
  });

  it('reads a cached value synchronously', () => {
    localStorage.setItem('alex-prefs', JSON.stringify({ density: 'compact' }));
    expect(cached().density).toBe('compact');
    // Unset keys still fall back.
    expect(cached().theme).toBe('dark');
  });

  it('survives a corrupt cache', () => {
    localStorage.setItem('alex-prefs', '{not json');
    expect(cached()).toEqual(FALLBACK);
  });

  it('applies presentation to the document root', () => {
    applyPrefs({ theme: 'light', density: 'compact', reduce_motion: true });
    const root = document.documentElement;
    expect(root.getAttribute('data-theme')).toBe('light');
    expect(root.getAttribute('data-density')).toBe('compact');
    expect(root.getAttribute('data-reduce-motion')).toBe('true');
  });

  it('clears reduce-motion when turned off', () => {
    applyPrefs({ theme: 'dark', density: 'comfortable', reduce_motion: true });
    applyPrefs({ theme: 'dark', density: 'comfortable', reduce_motion: false });
    expect(document.documentElement.hasAttribute('data-reduce-motion')).toBe(false);
  });
});
