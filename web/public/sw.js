/**
 * Advanced Service Worker for Jarvis PWA
 * 
 * Features:
 * - Offline functionality with cache-first strategy
 * - Background sync for queued messages
 * - Push notifications for background tasks
 * - Periodic background sync for updates
 * - Precaching of critical assets
 */

const CACHE_NAME = 'jarvis-v1';
const STATIC_CACHE = 'jarvis-static-v1';
const DYNAMIC_CACHE = 'jarvis-dynamic-v1';
const IMAGE_CACHE = 'jarvis-images-v1';

// Critical assets to precache
const PRECACHE_ASSETS = [
  '/',
  '/index.html',
  '/mobile',
  '/manifest.json',
  '/icons/icon-72.png',
  '/icons/icon-96.png',
  '/icons/icon-128.png',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
];

// Install event - precache critical assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Precaching assets');
        return cache.addAll(PRECACHE_ASSETS);
      })
      .then(() => {
        console.log('[SW] Precache complete');
        return self.skipWaiting();
      })
      .catch((err) => {
        console.error('[SW] Precache failed:', err);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            // Delete old caches
            if (cacheName !== STATIC_CACHE && 
                cacheName !== DYNAMIC_CACHE && 
                cacheName !== IMAGE_CACHE) {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('[SW] Activation complete');
        return self.clients.claim();
      })
  );
});

// Fetch event - cache strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip Chrome extensions
  if (url.protocol === 'chrome-extension:') {
    return;
  }
  
  // API requests - network first with cache fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request));
    return;
  }
  
  // WebSocket requests - don't cache
  if (url.protocol === 'wss:' || url.protocol === 'ws:') {
    return;
  }
  
  // Images - cache first with network fallback
  if (request.destination === 'image') {
    event.respondWith(cacheFirst(request, IMAGE_CACHE));
    return;
  }
  
  // Static assets - cache first
  if (request.destination === 'script' || 
      request.destination === 'style' ||
      request.destination === 'font') {
    event.respondWith(cacheFirst(request, STATIC_CACHE));
    return;
  }
  
  // HTML pages - stale-while-revalidate
  if (request.destination === 'document') {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }
  
  // Default - network with cache fallback
  event.respondWith(networkWithCacheFallback(request));
});

// Cache strategies

/**
 * Cache First Strategy
 * Try cache first, fallback to network
 */
async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  
  if (cached) {
    // Return cached version and update in background
    fetch(request)
      .then((response) => {
        if (response.ok) {
          cache.put(request, response.clone());
        }
      })
      .catch(() => {}); // Ignore background update errors
    
    return cached;
  }
  
  // Not in cache, fetch from network
  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    // Network failed, no cache
    return new Response('Network error', { 
      status: 503, 
      statusText: 'Service Unavailable' 
    });
  }
}

/**
 * Network First Strategy
 * Try network first, fallback to cache
 */
async function networkFirst(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      // Update cache
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    // Network failed, try cache
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }
    
    // No cache either
    return new Response(
      JSON.stringify({ error: 'Network unavailable' }), 
      { 
        status: 503, 
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

/**
 * Stale While Revalidate Strategy
 * Return cached immediately, update in background
 */
async function staleWhileRevalidate(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const cached = await cache.match(request);
  
  // Fetch from network in background
  const networkPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => cached); // Fallback to cache on error
  
  // Return cached immediately if available
  if (cached) {
    networkPromise; // Trigger background update
    return cached;
  }
  
  // No cache, wait for network
  return networkPromise;
}

/**
 * Network with Cache Fallback
 * Try network, fallback to cache
 */
async function networkWithCacheFallback(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }
    throw error;
  }
}

// Background Sync - queue messages when offline
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-messages') {
    event.waitUntil(syncMessages());
  }
  
  if (event.tag === 'sync-insights') {
    event.waitUntil(syncInsights());
  }
});

/**
 * Sync queued messages when back online
 */
async function syncMessages() {
  const db = await openDB('jarvis-sync', 1);
  const messages = await db.getAll('queued-messages');
  
  for (const message of messages) {
    try {
      await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(message),
      });
      
      // Remove from queue on success
      await db.delete('queued-messages', message.id);
    } catch (error) {
      console.error('[SW] Failed to sync message:', error);
    }
  }
}

/**
 * Sync insights when back online
 */
async function syncInsights() {
  // Fetch latest insights from server
  try {
    const response = await fetch('/api/insights/latest');
    if (response.ok) {
      const insights = await response.json();
      
      // Show notification for new insights
      insights.forEach((insight) => {
        self.registration.showNotification('New Insight from Jarvis', {
          body: insight.summary || 'A new insight has been discovered',
          icon: '/icons/icon-192.png',
          badge: '/icons/icon-72.png',
          tag: `insight-${insight.id}`,
          requireInteraction: true,
          data: { insight },
          actions: [
            { action: 'view', title: 'View' },
            { action: 'dismiss', title: 'Dismiss' },
          ],
        });
      });
    }
  } catch (error) {
    console.error('[SW] Failed to sync insights:', error);
  }
}

// Push Notifications
self.addEventListener('push', (event) => {
  console.log('[SW] Push received:', event);
  
  let data = {};
  try {
    data = event.data.json();
  } catch (e) {
    data = { title: 'Jarvis', body: event.data.text() };
  }
  
  const title = data.title || 'Jarvis';
  const options = {
    body: data.body || 'You have a new notification',
    icon: data.icon || '/icons/icon-192.png',
    badge: data.badge || '/icons/icon-72.png',
    tag: data.tag || 'default',
    requireInteraction: data.requireInteraction || false,
    data: data.data || {},
    actions: data.actions || [
      { action: 'open', title: 'Open' },
      { action: 'close', title: 'Close' },
    ],
    vibrate: data.vibrate || [200, 100, 200],
  };
  
  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  const action = event.action;
  const data = event.notification.data;
  
  if (action === 'close' || action === 'dismiss') {
    return;
  }
  
  // Open or focus app window
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then((windowClients) => {
        // Check if there's an existing window
        for (const client of windowClients) {
          if (client.url.startsWith(self.location.origin) && 'focus' in client) {
            client.focus();
            
            // Post message to client
            client.postMessage({
              type: 'notification-click',
              action: action,
              data: data,
            });
            
            return;
          }
        }
        
        // No existing window, open new one
        if (clients.openWindow) {
          const url = data?.url || '/';
          return clients.openWindow(url);
        }
      })
  );
});

// Message handling from client
self.addEventListener('message', (event) => {
  const { data } = event;
  
  if (data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (data.type === 'CACHE_ASSETS') {
    caches.open(DYNAMIC_CACHE)
      .then((cache) => cache.addAll(data.assets))
      .then(() => {
        event.source.postMessage({ type: 'ASSETS_CACHED' });
      });
  }
  
  if (data.type === 'CLEAR_CACHE') {
    caches.delete(DYNAMIC_CACHE)
      .then(() => caches.delete(IMAGE_CACHE))
      .then(() => {
        event.source.postMessage({ type: 'CACHE_CLEARED' });
      });
  }
});

// IndexedDB helper
function openDB(name, version) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, version);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('queued-messages')) {
        db.createObjectStore('queued-messages', { keyPath: 'id' });
      }
    };
  });
}

// Periodic background sync (if supported)
if ('periodicSync' in self.registration) {
  self.addEventListener('periodicsync', (event) => {
    if (event.tag === 'check-insights') {
      event.waitUntil(syncInsights());
    }
    
    if (event.tag === 'check-messages') {
      event.waitUntil(syncMessages());
    }
  });
}