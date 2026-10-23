// icons.jsx — inline SVG icon set. Stroke icons, 24-box, currentColor.
// Keeps the bundle free of icon-font deps.
import React from 'react';

const base = {
  width: 24, height: 24, viewBox: '0 0 24 24', fill: 'none',
  stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round',
};

export const IconChat = (p) => (
  <svg {...base} {...p}><path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.4 9 9 0 0 1-4-.9L3 21l1.9-5a8.38 8.38 0 0 1-.9-4A8.5 8.5 0 0 1 12.5 3a8.5 8.5 0 0 1 8.5 8.5z"/></svg>
);
export const IconEmail = (p) => (
  <svg {...base} {...p}><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></svg>
);
export const IconSessions = (p) => (
  <svg {...base} {...p}><rect x="3" y="4" width="18" height="6" rx="1.5"/><rect x="3" y="14" width="18" height="6" rx="1.5"/></svg>
);
export const IconFiles = (p) => (
  <svg {...base} {...p}><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/></svg>
);
export const IconBrowser = (p) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18z"/></svg>
);
export const IconTools = (p) => (
  <svg {...base} {...p}><path d="M14.7 6.3a4 4 0 0 0-5.4 5.3L3 18l3 3 6.4-6.4a4 4 0 0 0 5.3-5.4l-2.6 2.6-2.3-2.3z"/></svg>
);
export const IconBoard = (p) => (
  <svg {...base} {...p}><rect x="3" y="4" width="5" height="16" rx="1"/><rect x="10" y="4" width="5" height="10" rx="1"/><rect x="17" y="4" width="4" height="13" rx="1"/></svg>
);
export const IconGuidance = (p) => (
  <svg {...base} {...p}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/><path d="M9 7h7M9 11h5"/></svg>
);
export const IconOversight = (p) => (
  <svg {...base} {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/></svg>
);
export const IconTransparency = (p) => (
  <svg {...base} {...p}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>
);
export const IconSpark = (p) => (
  <svg {...base} width="16" height="16" viewBox="0 0 24 24" fill="#fff" stroke="none" {...p}><path d="M12 2l1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8z"/></svg>
);
export const IconSend = (p) => (
  <svg {...base} {...p}><path d="m22 2-7 20-4-9-9-4z"/><path d="M22 2 11 13"/></svg>
);
export const IconPaperclip = (p) => (
  <svg {...base} {...p}><path d="M21.4 11.05 12.25 20.2a5 5 0 0 1-7.07-7.07l9.19-9.19a3 3 0 0 1 4.24 4.24l-9.2 9.19a1 1 0 0 1-1.41-1.42l8.48-8.48"/></svg>
);
export const IconSearch = (p) => (
  <svg {...base} {...p}><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>
);
export const IconPlus = (p) => (
  <svg {...base} {...p}><path d="M12 5v14M5 12h14"/></svg>
);
export const IconClose = (p) => (
  <svg {...base} {...p}><path d="M18 6 6 18M6 6l12 12"/></svg>
);
export const IconDownload = (p) => (
  <svg {...base} {...p}><path d="M12 3v12m0 0 4-4m-4 4-4-4M4 21h16"/></svg>
);
export const IconChevron = (p) => (
  <svg {...base} {...p}><path d="m9 18 6-6-6-6"/></svg>
);
