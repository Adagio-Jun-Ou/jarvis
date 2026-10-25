/**
 * Theme Management System
 * 
 * Features:
 * - Dark/Light/System theme support
 * - CSS variable-based theming
 * - Smooth transitions
 * - Persistent storage
 * - OS-level preference detection
 */

import { useState, useEffect, useCallback } from 'react';
import type React from 'react';

export type Theme = 'dark' | 'light' | 'system';

interface ThemeColors {
  background: string;
  surface: string;
  primary: string;
  secondary: string;
  text: string;
  textSecondary: string;
  border: string;
  accent: string;
  success: string;
  warning: string;
  error: string;
  info: string;
}

const themes: Record<'dark' | 'light', ThemeColors> = {
  dark: {
    background: '#0f172a',
    surface: '#1e293b',
    primary: '#3b82f6',
    secondary: '#64748b',
    text: '#f1f5f9',
    textSecondary: '#94a3b8',
    border: '#334155',
    accent: '#06b6d4',
    success: '#22c55e',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
  },
  light: {
    background: '#ffffff',
    surface: '#f8fafc',
    primary: '#2563eb',
    secondary: '#64748b',
    text: '#1e293b',
    textSecondary: '#64748b',
    border: '#e2e8f0',
    accent: '#0891b2',
    success: '#16a34a',
    warning: '#d97706',
    error: '#dc2626',
    info: '#2563eb',
  },
};

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window === 'undefined') return 'system';
    return (localStorage.getItem('jarvis-theme') as Theme) || 'system';
  });

  const [resolvedTheme, setResolvedTheme] = useState<'dark' | 'light'>('dark');

  // Resolve system preference
  const getSystemTheme = useCallback((): 'dark' | 'light' => {
    if (typeof window === 'undefined') return 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }, []);

  // Apply theme
  const applyTheme = useCallback((newTheme: Theme) => {
    const resolved = newTheme === 'system' ? getSystemTheme() : newTheme;
    setResolvedTheme(resolved);

    // Apply CSS variables
    const colors = themes[resolved];
    const root = document.documentElement;
    
    Object.entries(colors).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value);
    });

    // Apply class for Tailwind
    if (resolved === 'dark') {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    } else {
      document.documentElement.classList.add('light');
      document.documentElement.classList.remove('dark');
    }

    // Store preference
    localStorage.setItem('jarvis-theme', newTheme);

    // Update meta theme-color
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.setAttribute('content', colors.background);
    }
  }, [getSystemTheme]);

  // Set theme
  const setThemeValue = useCallback((newTheme: Theme) => {
    setTheme(newTheme);
    applyTheme(newTheme);
  }, [applyTheme]);

  // Toggle theme
  const toggleTheme = useCallback(() => {
    const themes: Theme[] = ['light', 'dark', 'system'];
    const currentIndex = themes.indexOf(theme);
    const nextTheme = themes[(currentIndex + 1) % themes.length];
    setThemeValue(nextTheme);
  }, [theme, setThemeValue]);

  // Listen for system theme changes
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = () => {
      if (theme === 'system') {
        applyTheme('system');
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme, applyTheme]);

  // Initial apply
  useEffect(() => {
    applyTheme(theme);
  }, []);

  return {
    theme,
    resolvedTheme,
    setTheme: setThemeValue,
    toggleTheme,
    colors: themes[resolvedTheme],
  };
}

// ThemeProvider component
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const { resolvedTheme } = useTheme();

  return (
    <div className={`theme-${resolvedTheme} min-h-screen transition-colors duration-300`}>
      {children}
    </div>
  );
}

// CSS Variables to inject
export const themeVariables = `
  :root {
    --color-background: #0f172a;
    --color-surface: #1e293b;
    --color-primary: #3b82f6;
    --color-secondary: #64748b;
    --color-text: #f1f5f9;
    --color-text-secondary: #94a3b8;
    --color-border: #334155;
    --color-accent: #06b6d4;
    --color-success: #22c55e;
    --color-warning: #f59e0b;
    --color-error: #ef4444;
    --color-info: #3b82f6;
  }

  .light {
    --color-background: #ffffff;
    --color-surface: #f8fafc;
    --color-primary: #2563eb;
    --color-secondary: #64748b;
    --color-text: #1e293b;
    --color-text-secondary: #64748b;
    --color-border: #e2e8f0;
    --color-accent: #0891b2;
    --color-success: #16a34a;
    --color-warning: #d97706;
    --color-error: #dc2626;
    --color-info: #2563eb;
  }
`;