// Admin surfaces are hidden as well as refused. Hiding the nav entry is
// only half of it — the initial view is read from ?view= in the URL
// before the role is known, so a non-admin can type their way onto an
// admin surface and sit there reading a permission notice.

import { it, expect } from 'vitest';
import { allowedView } from '../App.jsx';

it('sends a non-admin off an admin-only view', () => {
  expect(allowedView('oversight', { isAdmin: false, roleLoaded: true })).toBe('chat');
});

it('leaves an admin on the admin-only view', () => {
  expect(allowedView('oversight', { isAdmin: true, roleLoaded: true })).toBe('oversight');
});

it('does not bounce anyone while the role is still loading', () => {
  // An admin on a slow connection must not be thrown off their own
  // surface just because /api/security/me has not answered yet.
  expect(allowedView('oversight', { isAdmin: false, roleLoaded: false })).toBe('oversight');
});

it('leaves ordinary views alone for everyone', () => {
  for (const view of ['chat', 'board', 'files', 'transparency']) {
    expect(allowedView(view, { isAdmin: false, roleLoaded: true })).toBe(view);
    expect(allowedView(view, { isAdmin: true, roleLoaded: true })).toBe(view);
  }
});

it('leaves an unknown view id alone rather than guessing', () => {
  expect(allowedView('nonsense', { isAdmin: false, roleLoaded: true })).toBe('nonsense');
});
