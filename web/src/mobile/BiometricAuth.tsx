import React, { useState, useEffect, useCallback } from 'react';

interface Props {
  onAuthenticated: (token: string) => void;
  hasCredential: boolean;
  deviceId: string;
}

export const BiometricAuth: React.FC<Props> = ({ onAuthenticated, hasCredential, deviceId }) => {
  const [status, setStatus] = useState<'idle' | 'authenticating' | 'registering' | 'error'>('idle');
  const [error, setError] = useState<string>('');
  const [started, setStarted] = useState(false);

  const arrayToBase64 = (buffer: ArrayBuffer): string => {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    bytes.forEach(byte => binary += String.fromCharCode(byte));
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  };

  const base64ToArray = (base64: string | undefined | null): ArrayBuffer => {
    if (!base64) {
      throw new Error('Invalid base64 data: empty or undefined');
    }
    const binary = atob(base64.replace(/-/g, '+').replace(/_/g, '/'));
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  };

  const startFaceID = useCallback(async () => {
    // Check if WebAuthn is available
    if (!window.PublicKeyCredential) {
      setError('FaceID/TouchID not supported on this device. Use password instead.');
      setStatus('error');
      return;
    }

    // Check if we're in a secure context (HTTPS or localhost)
    const isSecureContext = window.isSecureContext || 
                             window.location.protocol === 'https:' || 
                             window.location.hostname === 'localhost' ||
                             window.location.hostname === '127.0.0.1';
    
    if (!isSecureContext) {
      setError('FaceID requires HTTPS. Use password instead, or access Jarvis via HTTPS/Tailscale domain.');
      setStatus('error');
      return;
    }

    if (!deviceId) {
      setError('Device not ready. Please wait or use password.');
      setStatus('error');
      return;
    }

    setError('');
    setStatus(hasCredential ? 'authenticating' : 'registering');
    setStarted(true);

    try {
      if (hasCredential) {
        const challengeResp = await fetch('/api/mobile/auth/challenge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device_id: deviceId }),
        });

        const challengeData = await challengeResp.json();

        if (!challengeResp.ok) {
          throw new Error(challengeData.error || 'Failed to get challenge');
        }

        if (!challengeData.challenge || !challengeData.allowCredentials) {
          throw new Error('Invalid challenge response from server');
        }

        const assertion = await navigator.credentials.get({
          publicKey: {
            challenge: base64ToArray(challengeData.challenge),
            allowCredentials: challengeData.allowCredentials.map((c: any) => ({
              type: 'public-key',
              id: base64ToArray(c.id),
            })),
            userVerification: 'required',
            timeout: challengeData.timeout || 60000,
            rpId: challengeData.rpId || window.location.hostname,
          },
        }) as PublicKeyCredential;

        const response = assertion.response as AuthenticatorAssertionResponse;

        const verifyResp = await fetch('/api/mobile/auth/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            challenge_id: challengeData.challenge_id,
            credential_id: arrayToBase64(assertion.rawId),
            client_data_json: arrayToBase64(response.clientDataJSON),
            authenticator_data: arrayToBase64(response.authenticatorData),
            signature: arrayToBase64(response.signature),
            user_handle: response.userHandle ? arrayToBase64(response.userHandle) : null,
          }),
        });

        const verifyData = await verifyResp.json();

        if (verifyResp.ok && verifyData.token) {
          onAuthenticated(verifyData.token);
        } else {
          throw new Error(verifyData.error || 'Authentication failed');
        }
      } else {
        const registerResp = await fetch('/api/mobile/register/challenge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device_id: deviceId }),
        });

        const registerData = await registerResp.json();

        if (!registerResp.ok) {
          throw new Error(registerData.error || 'Failed to start registration');
        }

        if (!registerData.challenge || !registerData.rp || !registerData.user) {
          throw new Error('Invalid registration response from server');
        }

        const credential = await navigator.credentials.create({
          publicKey: {
            challenge: base64ToArray(registerData.challenge),
            rp: registerData.rp,
            user: {
              id: base64ToArray(registerData.user.id),
              name: registerData.user.name || 'jarvis-user',
              displayName: registerData.user.displayName || 'Jarvis User',
            },
            pubKeyCredParams: registerData.pubKeyCredParams || [{ type: 'public-key', alg: -7 }, { type: 'public-key', alg: -257 }],
            authenticatorSelection: registerData.authenticatorSelection || { authenticatorAttachment: 'platform', userVerification: 'required' },
            timeout: registerData.timeout || 60000,
            attestation: registerData.attestation || 'direct',
          },
        }) as PublicKeyCredential;

        const response = credential.response as AuthenticatorAttestationResponse;

        const verifyResp = await fetch('/api/mobile/register/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            challenge_id: registerData.challenge_id,
            credential_id: arrayToBase64(credential.rawId),
            client_data_json: arrayToBase64(response.clientDataJSON),
            attestation_object: arrayToBase64(response.attestationObject),
          }),
        });

        const verifyData = await verifyResp.json();

        if (verifyResp.ok && verifyData.token) {
          onAuthenticated(verifyData.token);
        } else {
          throw new Error(verifyData.error || 'Registration failed');
        }
      }
    } catch (err: any) {
      console.error('Biometric auth error:', err);
      
      // Don't show error for user cancellation
      if (err.name === 'NotAllowedError') {
        setStatus('idle');
        setError('FaceID was cancelled. Tap the button to try again.');
      } else {
        setError(err.message || 'Authentication failed. Use password instead.');
        setStatus('error');
      }
    }
  }, [hasCredential, deviceId, onAuthenticated]);

  // Auto-start once when ready
  useEffect(() => {
    if (!started && deviceId && window.PublicKeyCredential) {
      // Small delay to ensure UI is ready
      const timer = setTimeout(() => {
        startFaceID();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [deviceId, started, startFaceID]);

  return (
    <div className="biometric-auth">
      <button 
        className="face-id-btn" 
        onClick={startFaceID}
        disabled={status === 'authenticating' || status === 'registering'}
      >
        🔐
      </button>

      <div className="auth-status">
        {status === 'registering' && <p>Setting up Face ID...</p>}
        {status === 'authenticating' && <p>Look at the camera to authenticate</p>}
        {status === 'idle' && !started && <p>Tap to use Face ID</p>}
        {status === 'error' && <p className="error">{error}</p>}
      </div>
    </div>
  );
};


interface PasswordLoginProps {
  onAuthenticated: (token: string) => void;
  deviceId: string;
}

export const PasswordLogin: React.FC<PasswordLoginProps> = ({ onAuthenticated, deviceId }) => {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const resp = await fetch('/api/mobile/auth/password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          device_id: deviceId,
          password,
        }),
      });

      const data = await resp.json();

      if (resp.ok && data.token) {
        onAuthenticated(data.token);
      } else {
        setError(data.error || 'Invalid password');
      }
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="password-login" onSubmit={handleSubmit}>
      <div className="input-group">
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter your password"
          autoComplete="current-password"
          disabled={loading}
        />
      </div>

      {error && <p className="error">{error}</p>}

      <button type="submit" disabled={!password || loading}>
        {loading ? 'Authenticating...' : 'Sign In'}
      </button>
    </form>
  );
};