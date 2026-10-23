import React, { useState, useEffect, useCallback } from 'react';
import { BiometricAuth, PasswordLogin } from './BiometricAuth';
import { MobileApp } from './MobileApp';
import './Mobile.css';

const DEVICE_ID_KEY = 'jarvis_device_id';
const TOKEN_KEY = 'jarvis_token';

export const MobileLogin: React.FC = () => {
  const [token, setToken] = useState<string | null>(null);
  const [deviceId, setDeviceId] = useState<string>('');
  const [hasCredential, setHasCredential] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let storedDeviceId = localStorage.getItem(DEVICE_ID_KEY);
    
    if (!storedDeviceId) {
      storedDeviceId = `device-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem(DEVICE_ID_KEY, storedDeviceId);
    }
    
    setDeviceId(storedDeviceId);
    
    const storedToken = localStorage.getItem(TOKEN_KEY);
    
    if (storedToken) {
      fetch('/api/mobile/auth/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: storedToken }),
      })
        .then(r => r.json())
        .then(data => {
          if (data.valid) {
            setToken(storedToken);
          } else {
            localStorage.removeItem(TOKEN_KEY);
          }
          setLoading(false);
        })
        .catch(() => {
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!deviceId) return;
    
    fetch('/api/mobile/auth/status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device_id: deviceId }),
    })
      .then(r => r.json())
      .then(data => {
        setHasCredential(data.has_credential || false);
      })
      .catch(console.error);
  }, [deviceId]);

  const handleAuthenticated = useCallback((newToken: string) => {
    localStorage.setItem(TOKEN_KEY, newToken);
    setToken(newToken);
  }, []);

  const handleLogout = useCallback(() => {
    const storedToken = localStorage.getItem(TOKEN_KEY);
    if (storedToken) {
      fetch('/api/mobile/auth/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: storedToken }),
      }).catch(console.error);
    }
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
  }, []);

  if (loading) {
    return (
      <div className="login-screen">
        <div className="loading-spinner" />
      </div>
    );
  }

  if (token) {
    return <MobileApp token={token} onLogout={handleLogout} />;
  }

  return (
    <div className="login-screen">
      <div className="login-logo">
        <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="50" cy="50" r="45" stroke="currentColor" strokeWidth="2" />
          <circle cx="35" cy="40" r="5" fill="currentColor" />
          <circle cx="65" cy="40" r="5" fill="currentColor" />
          <path d="M35 65 Q50 75 65 65" stroke="currentColor" strokeWidth="3" fill="none" strokeLinecap="round" />
        </svg>
      </div>
      
      <h1 className="login-title">Jarvis</h1>
      <p className="login-subtitle">Your AI Assistant</p>

      {!showPassword ? (
        <BiometricAuth
          onAuthenticated={handleAuthenticated}
          hasCredential={hasCredential}
          deviceId={deviceId}
        />
      ) : (
        <PasswordLogin
          onAuthenticated={handleAuthenticated}
          deviceId={deviceId}
        />
      )}
      
      {!showPassword && (
        <button 
          className="fallback-button" 
          onClick={() => setShowPassword(true)}
          style={{ marginTop: 20 }}
        >
          Use Password Instead
        </button>
      )}
    </div>
  );
};

export default MobileLogin;