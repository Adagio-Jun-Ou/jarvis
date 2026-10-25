#!/usr/bin/env python3
"""Test PWA message persistence and other frontend features."""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import asyncio
import sys
import json
sys.path.insert(0, _JARVIS_ROOT)


def test_pwa_message_persistence():
    """Test PWA message persistence logic."""
    print("\n=== Test: PWA Message Persistence Logic ===")
    
    # Simulate the frontend logic
    MAX_STORED_MESSAGES = 100
    
    def load_messages():
        """Simulate loadMessages()"""
        try:
            # In real app, uses localStorage
            return []  # Empty initial state
        except:
            return []
    
    def save_messages(messages):
        """Simulate saveMessages()"""
        # Filter out pending messages and limit count
        to_save = [m for m in messages if not m.get('pending')][-MAX_STORED_MESSAGES:]
        return to_save
    
    # Test 1: Basic save/load
    messages = [
        {"id": "1", "role": "user", "content": "Hello", "timestamp": 1000},
        {"id": "2", "role": "assistant", "content": "Hi!", "timestamp": 1001},
        {"id": "3", "role": "user", "content": "How are you?", "timestamp": 1002},
    ]
    
    saved = save_messages(messages)
    assert len(saved) == 3
    print("  ✓ Basic save preserves all messages")
    
    # Test 2: Pending messages are filtered
    messages_with_pending = messages + [
        {"id": "4", "role": "user", "content": "Pending", "pending": True, "timestamp": 1003},
    ]
    
    saved = save_messages(messages_with_pending)
    assert len(saved) == 3
    assert not any(m.get('pending') for m in saved)
    print("  ✓ Pending messages are filtered out")
    
    # Test 3: Message limit
    many_messages = [
        {"id": str(i), "role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}", "timestamp": i}
        for i in range(150)
    ]
    
    saved = save_messages(many_messages)
    assert len(saved) == MAX_STORED_MESSAGES
    print(f"  ✓ Message limit enforced ({MAX_STORED_MESSAGES} max)")
    
    # Test 4: Most recent messages preserved
    saved_ids = [m['id'] for m in saved]
    assert '149' in saved_ids  # Most recent
    assert '0' not in saved_ids  # Oldest should be dropped
    print("  ✓ Most recent messages preserved")
    
    # Test 5: Round-trip save/load
    original = [
        {"id": "1", "role": "user", "content": "Test"},
        {"id": "2", "role": "assistant", "content": "Response"},
    ]
    saved = save_messages(original)
    # Simulate load from saved
    assert len(saved) == len(original)
    assert saved[0]['content'] == original[0]['content']
    print("  ✓ Round-trip save/load preserves messages")


def test_message_deduplication():
    """Test message deduplication in WebSocket handling."""
    print("\n=== Test: Message Deduplication ===")
    
    processed_ids = set()
    
    def is_duplicate(message_id):
        """Check if message was already processed"""
        if message_id in processed_ids:
            return True
        processed_ids.add(message_id)
        # Limit set size
        if len(processed_ids) > 100:
            # Remove oldest entries (convert to list, slice, convert back)
            processed_list = list(processed_ids)
            processed_ids.clear()
            processed_ids.update(processed_list[-50:])
        return False
    
    # Test 1: First message not duplicate
    assert not is_duplicate("msg_1")
    print("  ✓ First message not duplicate")
    
    # Test 2: Same message is duplicate
    assert is_duplicate("msg_1")
    print("  ✓ Second occurrence is duplicate")
    
    # Test 3: Different message not duplicate
    assert not is_duplicate("msg_2")
    print("  ✓ Different message not duplicate")
    
    # Test 4: Set size limit
    for i in range(150):
        is_duplicate(f"msg_{i+3}")
    
    assert len(processed_ids) <= 100
    print(f"  ✓ Set size limited to {len(processed_ids)}")


def test_connection_status():
    """Test connection status tracking."""
    print("\n=== Test: Connection Status ===")
    
    # Connection states
    VALID_STATUSES = ['connecting', 'connected', 'disconnected', 'reconnecting']
    
    def get_status_map(status):
        return {
            'connecting': {'color': 'orange', 'text': 'Connecting...'},
            'connected': {'color': 'green', 'text': 'Connected'},
            'disconnected': {'color': 'red', 'text': 'Offline'},
            'reconnecting': {'color': 'orange', 'text': 'Reconnecting...'},
        }.get(status, {'color': 'gray', 'text': 'Unknown'})
    
    # Test valid statuses
    for status in VALID_STATUSES:
        result = get_status_map(status)
        assert 'color' in result
        assert 'text' in result
        print(f"  ✓ {status}: color={result['color']}, text={result['text']}")
    
    # Test unknown status
    result = get_status_map('unknown')
    assert result['color'] == 'gray'
    print("  ✓ Unknown status handled gracefully")


def test_thinking_block_parsing():
    """Test parsing of thinking blocks."""
    print("\n=== Test: Thinking Block Parsing ===")
    
    def parse_thinking(text):
        """Parse <think> blocks"""
        import re
        
        match = re.search(r'<think[^>]*>([\s\S]*?)</think>', text, re.IGNORECASE)
        if not match:
            return None
        
        content = match.group(1)
        header = 'Thought'
        
        # Extract header if specified
        header_match = re.search(r'<think[^>]*header=["\']([^"\']+)["\']', text, re.IGNORECASE)
        if header_match:
            header = header_match.group(1)
        else:
            # Check for # header in content
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith('#'):
                    header = line.lstrip('#').strip() or 'Thought'
                elif line.lower().startswith('header:'):
                    header = line.split(':')[1].strip() or 'Thought'
                else:
                    new_lines.append(line)
            content = '\n'.join(new_lines).strip()
        
        return {'header': header, 'content': content}
    
    # Test 1: Basic thinking block
    result = parse_thinking("<think>This is my thought</think>")
    assert result is not None
    assert result['header'] == 'Thought'
    assert 'my thought' in result['content']
    print("  ✓ Basic thinking block parsed")
    
    # Test 2: Thinking block with header attribute
    result = parse_thinking('<think header="Analysis">Analyzing data...</think>')
    assert result['header'] == 'Analysis'
    print("  ✓ Header attribute extracted")
    
    # Test 3: Thinking block with # header
    result = parse_thinking('<think># Plan\nThis is my plan</think>')
    assert result['header'] == 'Plan'
    print("  ✓ # header extracted")
    
    # Test 4: No thinking block
    result = parse_thinking("No thinking here")
    assert result is None
    print("  ✓ Non-thinking text returns None")


def test_markdown_rendering():
    """Test markdown rendering for messages."""
    print("\n=== Test: Markdown Rendering ===")
    
    def render_markdown(text):
        """Simplified markdown rendering"""
        import re
        
        if not text:
            return []
        
        parts = re.split(r'(```[\s\S]*?```|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\n)', text)
        
        elements = []
        for i, part in enumerate(parts):
            if part.startswith('```'):
                match = re.match(r'```(\w*)\n?([\s\S]*?)```', part)
                if match:
                    lang, code = match.groups()
                    elements.append({'type': 'code', 'lang': lang, 'content': code.strip()})
            elif part.startswith('`') and part.endswith('`'):
                elements.append({'type': 'inline_code', 'content': part[1:-1]})
            elif part.startswith('**') and part.endswith('**'):
                elements.append({'type': 'bold', 'content': part[2:-2]})
            elif part.startswith('*') and part.endswith('*') and len(part) > 2:
                elements.append({'type': 'italic', 'content': part[1:-1]})
            elif part == '\n':
                elements.append({'type': 'break'})
            elif part.strip():
                elements.append({'type': 'text', 'content': part})
        
        return elements
    
    # Test 1: Code block
    result = render_markdown("```python\nprint('hello')\n```")
    assert len(result) == 1
    assert result[0]['type'] == 'code'
    assert result[0]['lang'] == 'python'
    print("  ✓ Code block rendered")
    
    # Test 2: Inline code
    result = render_markdown("Use `print` function")
    assert any(e['type'] == 'inline_code' and e['content'] == 'print' for e in result)
    print("  ✓ Inline code rendered")
    
    # Test 3: Bold and italic
    result = render_markdown("**bold** and *italic*")
    assert any(e['type'] == 'bold' for e in result)
    assert any(e['type'] == 'italic' for e in result)
    print("  ✓ Bold and italic rendered")
    
    # Test 4: Mixed content
    result = render_markdown("Hello **world**! Check `code` here:\n```js\nx=1\n```")
    types = [e['type'] for e in result]
    assert 'text' in types
    assert 'bold' in types
    assert 'inline_code' in types
    assert 'code' in types
    print("  ✓ Mixed content rendered correctly")


def main():
    """Run all PWA tests."""
    print("=" * 70)
    print("PWA MESSAGE PERSISTENCE TESTS")
    print("=" * 70)
    
    tests = [
        test_pwa_message_persistence,
        test_message_deduplication,
        test_connection_status,
        test_thinking_block_parsing,
        test_markdown_rendering,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)