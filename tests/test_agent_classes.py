"""Unit tests for core agent classes."""
import pytest
import asyncio
from core.agent import TaskCompletionCache, RequestQueue


class TestTaskCompletionCache:
    """Tests for TaskCompletionCache - pattern-based task completion detection."""
    
    def test_simple_read_query_complete(self):
        """Single read queries should be marked complete."""
        assert TaskCompletionCache.is_definitely_complete(
            "read file.py",
            [("read_file", "content here with enough length")]
        )
    
    def test_simple_show_query_complete(self):
        """Single show queries should be marked complete."""
        assert TaskCompletionCache.is_definitely_complete(
            "show the config",
            [("read_file", "config data with enough characters")]
        )
    
    def test_simple_what_query_complete(self):
        """Single what queries should be marked complete."""
        assert TaskCompletionCache.is_definitely_complete(
            "what is python",
            [("search", "python definition with enough length")]
        )
    
    def test_compound_query_not_complete(self):
        """Compound queries should NOT be marked complete."""
        assert not TaskCompletionCache.is_definitely_complete(
            "find all python files and analyze them",
            [("shell", "file1.py\nfile2.py")]
        )
    
    def test_and_then_not_complete(self):
        """Queries with 'and then' should NOT be marked complete."""
        assert not TaskCompletionCache.is_definitely_complete(
            "read the file and then summarize it",
            [("read_file", "content with enough length")]
        )
    
    def test_first_word_pattern_complete(self):
        """First word matching simple pattern should be complete."""
        assert TaskCompletionCache.is_definitely_complete(
            "get the status",
            [("shell", "status output with length")]
        )
    
    def test_empty_results_not_complete(self):
        """Empty tool results should not be complete."""
        assert not TaskCompletionCache.is_definitely_complete(
            "read file.py",
            [("read_file", "")]
        )
    
    def test_short_results_not_complete(self):
        """Very short results should not be complete."""
        assert not TaskCompletionCache.is_definitely_complete(
            "read file.py",
            [("read_file", "x")]  # Too short
        )
    
    def test_multiple_tools_not_complete(self):
        """Multiple tool executions need LLM evaluation."""
        assert not TaskCompletionCache.is_definitely_complete(
            "read file1.py and file2.py",
            [
                ("read_file", "content1 with enough length"),
                ("read_file", "content2 with enough length")
            ]
        )
    
    def test_successful_result_complete(self):
        """Single tool with meaningful result should be complete."""
        result = TaskCompletionCache.is_definitely_complete(
            "list the files",
            [("shell", "file1.py\nfile2.py\nfile3.py")]
        )
        # Should be complete because "list" is in SINGLE_TOOL_COMPLETE_PATTERNS
        assert result


class TestRequestQueue:
    """Tests for RequestQueue - request prioritization and concurrency."""
    
    def test_queue_creation(self):
        """Test queue can be created."""
        queue = RequestQueue(max_concurrent=3)
        assert queue.max_concurrent == 3
        assert queue.max_queue_size == 100
    
    @pytest.mark.asyncio
    async def test_queue_enqueue(self):
        """Test queuing a request."""
        queue = RequestQueue(max_concurrent=3)
        
        async def simple_handler():
            return "result"
        
        # Start the queue
        await queue.start()
        
        try:
            result = await queue.enqueue("test_request", simple_handler, timeout=5.0)
            assert result == "result"
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_concurrent_limit(self):
        """Test that concurrent limit is respected."""
        queue = RequestQueue(max_concurrent=2)
        assert queue.max_concurrent == 2
        
        await queue.start()
        try:
            # Queue should accept requests
            async def handler():
                await asyncio.sleep(0.1)
                return "done"
            
            # Start multiple requests concurrently
            tasks = [
                queue.enqueue(f"req{i}", handler, timeout=5.0)
                for i in range(3)
            ]
            
            results = await asyncio.gather(*tasks)
            assert len(results) == 3
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_stop(self):
        """Test stopping the queue."""
        queue = RequestQueue(max_concurrent=3)
        
        await queue.start()
        assert queue._running is True
        
        await queue.stop()
        assert queue._running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])