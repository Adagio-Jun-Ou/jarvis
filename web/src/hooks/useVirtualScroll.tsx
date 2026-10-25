/**
 * Progressive Loading for Chat History
 * 
 * Features:
 * - Virtual scrolling for long conversations
 * - Progressive message loading
 * - Intersection Observer for lazy loading
 * - Smooth scroll position restoration
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import type React from 'react';

interface UseVirtualScrollOptions {
  itemHeight: number;
  overscan?: number;
  containerHeight?: number;
}

export function useVirtualScroll<T>(
  items: T[],
  options: UseVirtualScrollOptions
) {
  const { itemHeight, overscan = 5, containerHeight = 600 } = options;
  
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  
  // Calculate visible range
  const virtualItems = useMemo(() => {
    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
    const visibleCount = Math.ceil(containerHeight / itemHeight) + 2 * overscan;
    const endIndex = Math.min(items.length, startIndex + visibleCount);
    
    return items.slice(startIndex, endIndex).map((item, index) => ({
      ...item,
      index: startIndex + index,
      style: {
        position: 'absolute' as const,
        top: (startIndex + index) * itemHeight,
        height: itemHeight,
        left: 0,
        right: 0,
      },
    }));
  }, [items, scrollTop, itemHeight, overscan, containerHeight]);
  
  // Total height for scroll container
  const totalHeight = items.length * itemHeight;
  
  // Handle scroll
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const newScrollTop = (e.target as HTMLDivElement).scrollTop;
    setScrollTop(newScrollTop);
    
    // Check if near top (for loading older messages)
    if (newScrollTop < itemHeight * 3 && !isLoading) {
      const event = new CustomEvent('virtual-scroll-near-top', {
        detail: { scrollTop: newScrollTop },
      });
      containerRef.current?.dispatchEvent(event);
    }
  }, [itemHeight, isLoading]);
  
  // Load more items (for progressive loading)
  const loadMore = useCallback(async (direction: 'before' | 'after') => {
    setIsLoading(true);
    
    const event = new CustomEvent('virtual-scroll-load-more', {
      detail: { direction },
    });
    containerRef.current?.dispatchEvent(event);
    
    // Simulate loading delay
    await new Promise(resolve => setTimeout(resolve, 300));
    setIsLoading(false);
  }, []);
  
  // Scroll to item
  const scrollToItem = useCallback((index: number) => {
    if (containerRef.current) {
      containerRef.current.scrollTop = index * itemHeight;
    }
  }, [itemHeight]);
  
  // Scroll to bottom
  const scrollToBottom = useCallback(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = totalHeight;
    }
  }, [totalHeight]);
  
  return {
    containerRef,
    virtualItems,
    totalHeight,
    handleScroll,
    scrollToItem,
    scrollToBottom,
    loadMore,
    isLoading,
  };
}

/**
 * Intersection Observer for lazy loading
 */
export function useIntersectionObserver(
  options: IntersectionObserverInit = {},
  callback: (entries: IntersectionObserverEntry[]) => void
) {
  const { threshold = 0.1, root = null, rootMargin = '0px' } = options;
  const targetRef = useRef<HTMLElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);
  
  useEffect(() => {
    const element = targetRef.current;
    if (!element) return;
    
    observerRef.current = new IntersectionObserver(callback, {
      threshold,
      root,
      rootMargin,
    });
    
    observerRef.current.observe(element);
    
    return () => {
      observerRef.current?.disconnect();
    };
  }, [threshold, root, rootMargin, callback]);
  
  return targetRef;
}

/**
 * Progressive Image Loading
 */
export function useProgressiveImage(src: string, placeholderSrc?: string) {
  const [imageSrc, setImageSrc] = useState(placeholderSrc || '');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    if (!src) return;
    
    setIsLoading(true);
    setError(null);
    
    const img = new Image();
    
    img.onload = () => {
      setImageSrc(src);
      setIsLoading(false);
    };
    
    img.onerror = () => {
      setError(new Error(`Failed to load image: ${src}`));
      setIsLoading(false);
    };
    
    img.src = src;
    
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [src]);
  
  return { src: imageSrc, isLoading, error };
}

/**
 * Progressive Content Loading Hook
 */
export function useProgressiveLoading<T>(
  loadItems: (page: number) => Promise<T[]>,
  options: { threshold?: number } = {}
) {
  const { threshold = 100 } = options;
  
  const [items, setItems] = useState<T[]>([]);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const loaderRef = useRef<HTMLDivElement>(null);
  
  const loadMore = useCallback(async () => {
    if (isLoading || !hasMore) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const newItems = await loadItems(page);
      
      if (newItems.length === 0) {
        setHasMore(false);
      } else {
        setItems(prev => [...prev, ...newItems]);
        setPage(p => p + 1);
      }
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  }, [loadItems, page, isLoading, hasMore]);
  
  // Intersection Observer for infinite scroll
  useEffect(() => {
    const element = loaderRef.current;
    if (!element) return;
    
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !isLoading) {
          loadMore();
        }
      },
      { threshold: threshold / 100 }
    );
    
    observer.observe(element);
    
    return () => observer.disconnect();
  }, [loadMore, hasMore, isLoading, threshold]);
  
  const reset = useCallback(() => {
    setItems([]);
    setPage(0);
    setHasMore(true);
    setError(null);
  }, []);
  
  return {
    items,
    isLoading,
    hasMore,
    error,
    loadMore,
    reset,
    loaderRef,
  };
}

/**
 * Chat History Virtual List Component
 */
export function VirtualMessageList({
  messages,
  renderMessage,
  itemHeight = 80,
  containerHeight = 600,
  onLoadMore,
}: {
  messages: any[];
  renderMessage: (message: any, index: number) => React.ReactNode;
  itemHeight?: number;
  containerHeight?: number;
  onLoadMore?: () => void;
}) {
  const {
    containerRef,
    virtualItems,
    totalHeight,
    handleScroll,
    isLoading,
  } = useVirtualScroll(messages, { itemHeight, containerHeight });
  
  // Listen for load more events
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !onLoadMore) return;
    
    const handleLoadMore = () => {
      onLoadMore();
    };
    
    container.addEventListener('virtual-scroll-load-more', handleLoadMore);
    return () => {
      container.removeEventListener('virtual-scroll-load-more', handleLoadMore);
    };
  }, [containerRef, onLoadMore]);
  
  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      style={{
        height: containerHeight,
        overflow: 'auto',
        position: 'relative',
      }}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        {virtualItems.map((item: any) => (
          <div key={item.id || item.index} style={item.style}>
            {renderMessage(item, item.index)}
          </div>
        ))}
      </div>
      
      {isLoading && (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          Loading more messages...
        </div>
      )}
    </div>
  );
}