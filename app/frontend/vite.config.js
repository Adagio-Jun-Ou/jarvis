/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite config for the Alex workspace SPA.
//
// Production: `npm run build` produces frontend/dist/ with hashed
// asset names; the FastAPI backend mounts /assets/ and serves
// index.html for every other route. The build is done in the bundle's
// app.yaml `build:` block.
//
// Dev: `npm run dev` runs the SPA on :5173 and proxies /api to the
// FastAPI backend on :8000.
//
// Test: jsdom is the environment for component tests (the Browser
// OS shell pieces are pure React with no native deps).
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,                 // never ship source maps to prod
    minify: 'esbuild',
    rollupOptions: {
      output: {
        manualChunks: {
          // Keep vendor split so first paint is fast
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: false,
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.js'],
  },
});
