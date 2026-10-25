/**
 * Keyboard Shortcuts Hook
 * 
 * Features:
 * - Global keyboard shortcuts
 * - Scoped shortcuts for specific components
 * - Modifier key support (Ctrl, Alt, Shift, Meta)
 * - Prevent default behavior
 */

import { useState, useEffect, useCallback, useRef } from 'react';

export interface Shortcut {
  key: string;
  modifiers?: {
    ctrl?: boolean;
    alt?: boolean;
    shift?: boolean;
    meta?: boolean;
  };
  action: () => void;
  description?: string;
  preventDefault?: boolean;
  scope?: string;
}

export function useKeyboardShortcuts(shortcuts: Shortcut[]) {
  const shortcutsRef = useRef(shortcuts);
  
  useEffect(() => {
    shortcutsRef.current = shortcuts;
  }, [shortcuts]);
  
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    for (const shortcut of shortcutsRef.current) {
      const { key, modifiers = {}, action, preventDefault = true, scope } = shortcut;
      
      // Check if scope is active (if specified)
      if (scope) {
        const activeScope = document.activeElement?.getAttribute('data-shortcut-scope');
        if (activeScope !== scope) continue;
      }
      
      // Check key match
      if (event.key.toLowerCase() !== key.toLowerCase()) continue;
      
      // Check modifiers
      if (modifiers.ctrl && !event.ctrlKey) continue;
      if (modifiers.alt && !event.altKey) continue;
      if (modifiers.shift && !event.shiftKey) continue;
      if (modifiers.meta && !event.metaKey) continue;
      
      // Check if modifiers match exactly
      if (Object.keys(modifiers).length > 0) {
        const hasModifier = event.ctrlKey || event.altKey || event.shiftKey || event.metaKey;
        if (!hasModifier && Object.keys(modifiers).some(k => modifiers[k as keyof typeof modifiers])) continue;
      }
      
      // Execute action
      if (preventDefault) {
        event.preventDefault();
      }
      action();
      break;
    }
  }, []);
  
  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
  
  // Return shortcut help
  const getShortcutHelp = useCallback(() => {
    return shortcuts.map(s => {
      const modifiers = [];
      if (s.modifiers?.ctrl) modifiers.push('Ctrl');
      if (s.modifiers?.alt) modifiers.push('Alt');
      if (s.modifiers?.shift) modifiers.push('Shift');
      if (s.modifiers?.meta) modifiers.push('Cmd');
      
      const keyCombo = [...modifiers, s.key.toUpperCase()].join('+');
      return { keyCombo, description: s.description || s.key };
    });
  }, [shortcuts]);
  
  return { getShortcutHelp };
}

// Common shortcuts for Jarvis
export const createDefaultShortcuts = (actions: {
  newChat: () => void;
  clearChat: () => void;
  toggleSidebar: () => void;
  focusInput: () => void;
  toggleTheme: () => void;
  sendMessage: () => void;
  escape: () => void;
}): Shortcut[] => [
  {
    key: 'k',
    modifiers: { ctrl: true },
    action: actions.newChat,
    description: 'New chat',
  },
  {
    key: 'l',
    modifiers: { ctrl: true },
    action: actions.clearChat,
    description: 'Clear chat',
  },
  {
    key: 'b',
    modifiers: { ctrl: true },
    action: actions.toggleSidebar,
    description: 'Toggle sidebar',
  },
  {
    key: '/',
    modifiers: { ctrl: true },
    action: actions.focusInput,
    description: 'Focus input',
  },
  {
    key: 'd',
    modifiers: { ctrl: true, shift: true },
    action: actions.toggleTheme,
    description: 'Toggle dark mode',
  },
  {
    key: 'Enter',
    modifiers: { meta: true },
    action: actions.sendMessage,
    description: 'Send message (Cmd+Enter)',
  },
  {
    key: 'Escape',
    action: actions.escape,
    description: 'Cancel / Close',
    preventDefault: false,
  },
];

/**
 * Touch Gestures Hook
 * 
 * Features:
 * - Swipe gestures (left, right, up, down)
 * - Pinch to zoom
 * - Tap and hold
 * - Pull to refresh
 */

interface TouchState {
  startX: number;
  startY: number;
  startTime: number;
}

interface SwipeConfig {
  threshold?: number;
  velocity?: number;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
}

export function useTouchGestures(elementRef: React.RefObject<HTMLElement>, config: SwipeConfig) {
  const { threshold = 50, velocity = 0.3 } = config;
  const [touchState, setTouchState] = useState<TouchState | null>(null);
  
  const handleTouchStart = useCallback((e: TouchEvent) => {
    const touch = e.touches[0];
    setTouchState({
      startX: touch.clientX,
      startY: touch.clientY,
      startTime: Date.now(),
    });
  }, []);
  
  const handleTouchEnd = useCallback((e: TouchEvent) => {
    if (!touchState) return;
    
    const touch = e.changedTouches[0];
    const deltaX = touch.clientX - touchState.startX;
    const deltaY = touch.clientY - touchState.startY;
    const deltaTime = Date.now() - touchState.startTime;
    
    const velocityX = Math.abs(deltaX) / deltaTime;
    const velocityY = Math.abs(deltaY) / deltaTime;
    
    // Check for swipe
    if (velocityX > velocity || velocityY > velocity) {
      if (Math.abs(deltaX) > Math.abs(deltaY)) {
        // Horizontal swipe
        if (Math.abs(deltaX) > threshold) {
          if (deltaX > 0) {
            config.onSwipeRight?.();
          } else {
            config.onSwipeLeft?.();
          }
        }
      } else {
        // Vertical swipe
        if (Math.abs(deltaY) > threshold) {
          if (deltaY > 0) {
            config.onSwipeDown?.();
          } else {
            config.onSwipeUp?.();
          }
        }
      }
    }
    
    setTouchState(null);
  }, [touchState, threshold, velocity, config]);
  
  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;
    
    element.addEventListener('touchstart', handleTouchStart);
    element.addEventListener('touchend', handleTouchEnd);
    
    return () => {
      element.removeEventListener('touchstart', handleTouchStart);
      element.removeEventListener('touchend', handleTouchEnd);
    };
  }, [elementRef, handleTouchStart, handleTouchEnd]);
}

/**
 * Pull to Refresh Hook
 */
export function usePullToRefresh(
  elementRef: React.RefObject<HTMLElement>,
  onRefresh: () => Promise<void>
) {
  const [pullDistance, setPullDistance] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const startYRef = useRef(0);
  const isPullingRef = useRef(false);
  
  const THRESHOLD = 80;
  
  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;
    
    const handleTouchStart = (e: TouchEvent) => {
      if (element.scrollTop === 0 && !isRefreshing) {
        startYRef.current = e.touches[0].clientY;
        isPullingRef.current = true;
      }
    };
    
    const handleTouchMove = (e: TouchEvent) => {
      if (!isPullingRef.current) return;
      
      const currentY = e.touches[0].clientY;
      const delta = currentY - startYRef.current;
      
      if (delta > 0) {
        e.preventDefault();
        setPullDistance(Math.min(delta * 0.5, THRESHOLD * 1.5));
      }
    };
    
    const handleTouchEnd = async () => {
      if (!isPullingRef.current) return;
      
      isPullingRef.current = false;
      
      if (pullDistance >= THRESHOLD && !isRefreshing) {
        setIsRefreshing(true);
        await onRefresh();
        setIsRefreshing(false);
      }
      
      setPullDistance(0);
    };
    
    element.addEventListener('touchstart', handleTouchStart);
    element.addEventListener('touchmove', handleTouchMove, { passive: false });
    element.addEventListener('touchend', handleTouchEnd);
    
    return () => {
      element.removeEventListener('touchstart', handleTouchStart);
      element.removeEventListener('touchmove', handleTouchMove);
      element.removeEventListener('touchend', handleTouchEnd);
    };
  }, [elementRef, onRefresh, pullDistance, isRefreshing]);
  
  return { pullDistance, isRefreshing };
}

/**
 * Long Press Hook
 */
export function useLongPress(
  callback: () => void,
  ms = 500
) {
  const [startLongPress, setStartLongPress] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  useEffect(() => {
    if (startLongPress) {
      timerRef.current = setTimeout(callback, ms);
    } else if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [startLongPress, callback, ms]);
  
  return {
    onMouseDown: () => setStartLongPress(true),
    onMouseUp: () => setStartLongPress(false),
    onMouseLeave: () => setStartLongPress(false),
    onTouchStart: () => setStartLongPress(true),
    onTouchEnd: () => setStartLongPress(false),
  };
}

/**
 * Double Tap Hook
 */
export function useDoubleTap(
  onDoubleTap: () => void,
  delay = 300
) {
  const [lastTap, setLastTap] = useState(0);
  
  return useCallback(() => {
    const now = Date.now();
    if (now - lastTap < delay) {
      onDoubleTap();
    }
    setLastTap(now);
  }, [lastTap, onDoubleTap, delay]);
}