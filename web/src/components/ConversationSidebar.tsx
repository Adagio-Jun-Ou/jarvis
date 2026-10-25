import { useState, useEffect } from 'react';
import './ConversationSidebar.css';

interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

interface ConversationSidebarProps {
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  isOpen: boolean;
  onToggle: () => void;
}

export function ConversationSidebar({
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  isOpen,
  onToggle,
}: ConversationSidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const loadConversations = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/conversations');
      const data = await response.json();
      if (data.success) {
        setConversations(data.conversations);
      }
    } catch (error) {
      console.error('Failed to load conversations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadConversations();
  }, []);

  useEffect(() => {
    if (isOpen) {
      loadConversations();
    }
  }, [isOpen]);

  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    if (!query.trim()) {
      loadConversations();
      return;
    }

    try {
      const response = await fetch(`/api/conversations/search?query=${encodeURIComponent(query)}`);
      const data = await response.json();
      if (data.success) {
        setConversations(data.results);
      }
    } catch (error) {
      console.error('Search failed:', error);
    }
  };

  const formatDate = (dateStr: string) => {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (window.confirm('Delete this conversation?')) {
      onDeleteConversation(id);
      setConversations(prev => prev.filter(c => c.id !== id));
    }
  };

  return (
    <>
      {isOpen && <div className="sidebar-overlay" onClick={onToggle} />}
      <div className={`conversation-sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h3>History</h3>
          <button className="close-sidebar" onClick={onToggle}>×</button>
        </div>

        <button className="new-conversation-btn" onClick={onNewConversation}>
          + New Chat
        </button>

        <div className="search-box">
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
          />
        </div>

        <div className="conversations-list">
          {isLoading ? (
            <div className="loading">Loading...</div>
          ) : conversations.length === 0 ? (
            <div className="no-conversations">
              <div className="no-conv-icon">💬</div>
              <div>No conversations yet</div>
              <div className="no-conv-hint">Start chatting to see your history here</div>
            </div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                className={`conversation-item ${currentConversationId === conv.id ? 'active' : ''}`}
                onClick={() => onSelectConversation(conv.id)}
              >
                <div className="conversation-title">{conv.title || 'Untitled'}</div>
                <div className="conversation-meta">
                  <span className="conversation-date">{formatDate(conv.updated_at)}</span>
                  <span className="conversation-count">{conv.message_count} msgs</span>
                </div>
                <button
                  className="delete-btn"
                  onClick={(e) => handleDelete(e, conv.id)}
                  title="Delete conversation"
                >
                  🗑️
                </button>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <button className="refresh-btn" onClick={loadConversations} disabled={isLoading}>
            ↻ Refresh
          </button>
          <div className="conversation-count-label">
            {conversations.length} conversation{conversations.length !== 1 ? 's' : ''}
          </div>
        </div>
      </div>
    </>
  );
}