"""Tests for Conversation Summarizer."""
import pytest


class TestConversationSummarizer:
    def test_import(self):
        from core.conversation_summarizer import ConversationSummarizer, SummaryStyle, get_conversation_summarizer
        assert ConversationSummarizer is not None
        assert SummaryStyle.CONCISE.value == "concise"

    def test_singleton(self):
        from core.conversation_summarizer import get_conversation_summarizer
        s1 = get_conversation_summarizer()
        s2 = get_conversation_summarizer()
        assert s1 is s2

    def test_empty_messages(self):
        from core.conversation_summarizer import ConversationSummarizer
        s = ConversationSummarizer()
        result = s.summarize_messages([])
        assert result.summary_text == ""
        assert result.source_messages == 0

    def test_concise_summary(self):
        from core.conversation_summarizer import ConversationSummarizer, SummaryStyle
        s = ConversationSummarizer()
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language known for readability and versatility."},
            {"role": "user", "content": "What can I do with Python?"},
            {"role": "assistant", "content": "Python is used for web development, data analysis, AI, automation, and more."},
        ]
        result = s.summarize_messages(messages, SummaryStyle.CONCISE)
        assert len(result.summary_text) > 0
        assert result.source_messages == 4
        assert result.source_turns == 2

    def test_detailed_summary(self):
        from core.conversation_summarizer import ConversationSummarizer, SummaryStyle
        s = ConversationSummarizer()
        messages = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Hello Alice! How can I help you?"},
        ]
        result = s.summarize_messages(messages, SummaryStyle.DETAILED)
        assert "Alice" in result.summary_text

    def test_key_facts_extraction(self):
        from core.conversation_summarizer import ConversationSummarizer
        s = ConversationSummarizer()
        messages = [
            {"role": "user", "content": "Remember: my favorite color is blue"},
            {"role": "assistant", "content": "I'll remember that your favorite color is blue."},
        ]
        result = s.summarize_messages(messages)
        assert len(result.key_facts) > 0
        assert "blue" in result.key_facts[0]

    def test_tools_extraction(self):
        from core.conversation_summarizer import ConversationSummarizer
        s = ConversationSummarizer()
        messages = [
            {"role": "user", "content": "List files"},
            {"role": "tool", "name": "shell", "content": "file1.txt\nfile2.txt"},
            {"role": "assistant", "content": "Found 2 files."},
        ]
        result = s.summarize_messages(messages)
        assert "shell" in result.tools_used

    def test_compression_ratio(self):
        from core.conversation_summarizer import ConversationSummarizer
        s = ConversationSummarizer()
        messages = [
            {"role": "user", "content": "Tell me about " + "AI " * 200},
            {"role": "assistant", "content": "AI is " + "fascinating " * 200},
        ]
        result = s.summarize_messages(messages)
        assert result.original_tokens > 0
        summary_tokens = len(result.summary_text) // 4
        assert summary_tokens < result.original_tokens

    def test_stats(self):
        from core.conversation_summarizer import ConversationSummarizer
        s = ConversationSummarizer()
        messages = [{"role": "user", "content": "Hello"}]
        s.summarize_messages(messages)
        stats = s.get_stats()
        assert stats["total_summaries"] >= 1
        assert "avg_compression_ratio" in stats

    def test_get_summary(self):
        from core.conversation_summarizer import ConversationSummarizer
        s = ConversationSummarizer()
        messages = [{"role": "user", "content": "Test message"}]
        result = s.summarize_messages(messages)
        retrieved = s.get_summary(result.id)
        assert retrieved is not None
        assert retrieved.id == result.id
        assert s.get_summary("nonexistent") is None

    def test_topics_extraction(self):
        from core.conversation_summarizer import ConversationSummarizer
        s = ConversationSummarizer()
        messages = [
            {"role": "user", "content": "Tell me about machine learning and neural networks"},
            {"role": "assistant", "content": "Machine learning uses neural networks to learn patterns from data"},
        ]
        result = s.summarize_messages(messages)
        assert len(result.topics) > 0