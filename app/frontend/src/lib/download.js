import { toast } from './toast.jsx';

// Download an artifact through an authenticated fetch. A plain <a href>
// navigation cannot carry the Authorization bearer header used in
// token-auth (dev) mode, so it 401s. Fetch the blob with auth headers and
// trigger a client-side download instead.
export async function downloadArtifact(path, authHeaders) {
  const name = String(path || '').split('/').pop() || 'download';
  try {
    const r = await fetch(`/api/files/download?path=${encodeURIComponent(path)}`, {
      credentials: 'include',
      headers: (authHeaders && authHeaders()) || {},
    });
    if (!r.ok) throw new Error(`${r.status}`);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch {
    toast(`Download failed: ${name}`);
  }
}
