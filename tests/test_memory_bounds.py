"""Tests for memory-bounded caches and leak prevention.

Validates that all bounded caches properly limit memory usage and
evict items when limits are reached.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataFrameCache:
    """Tests for DataFrameCache in data_tools.py."""

    def test_cache_limits_entries(self):
        """Cache should enforce max entries limit."""
        from tools.data_tools import DataFrameCache
        cache = DataFrameCache(max_memory_mb=100, max_entries=3)

        async def run():
            # Create mock dataframes (simple objects with memory_usage method)
            class MockDF:
                def __init__(self, size):
                    self._size = size
                def memory_usage(self, deep=True):
                    class MemResult:
                        def sum(self):
                            return self._size
                    mem = MemResult()
                    mem._size = self._size
                    return mem

            # Add 3 DataFrames
            for i in range(3):
                result = await cache.set(f'df_{i}', MockDF(1000))
                assert result is True

            # Adding 4th should trigger eviction
            result = await cache.set('df_3', MockDF(1000))
            assert result is True

            stats = cache.get_stats()
            assert stats['entries'] == 3
            assert stats['evictions'] >= 1

        asyncio.run(run())

    def test_cache_limits_memory(self):
        """Cache should enforce memory limit."""
        from tools.data_tools import DataFrameCache
        cache = DataFrameCache(max_memory_mb=1, max_entries=100)  # 1MB limit

        async def run():
            class MockDF:
                def __init__(self, size_mb):
                    self._size = size_mb * 1024 * 1024
                def memory_usage(self, deep=True):
                    class MemResult:
                        def sum(self):
                            return self._size
                    mem = MemResult()
                    mem._size = self._size
                    return mem

            # Try to add 2MB dataframe - should fail (exceeds 1MB limit)
            big_df = MockDF(2)
            result = await cache.set('big_df', big_df)
            assert result is False

            # Add small dataframe - should succeed
            small_df = MockDF(0.5)  # 0.5MB
            result = await cache.set('small_df', small_df)
            assert result is True

        asyncio.run(run())

    def test_cache_lru_eviction(self):
        """Cache should evict least recently used items."""
        from tools.data_tools import DataFrameCache
        cache = DataFrameCache(max_memory_mb=100, max_entries=3)

        async def run():
            class MockDF:
                def memory_usage(self, deep=True):
                    class MemResult:
                        def sum(self):
                            return 1000
                    return MemResult()

            # Add items: first, second, third
            await cache.set('first', MockDF())
            await cache.set('second', MockDF())
            await cache.set('third', MockDF())

            # Access 'first' to make it recently used
            await cache.get('first')

            # Add 'fourth' - should evict 'second' (oldest unused)
            await cache.set('fourth', MockDF())

            # 'first' should still exist
            result = await cache.get('first')
            assert result is not None

            # 'second' should be evicted
            result = await cache.get('second')
            assert result is None

        asyncio.run(run())

    def test_cache_clear(self):
        """Cache clear should remove all items."""
        from tools.data_tools import DataFrameCache
        cache = DataFrameCache()

        async def run():
            class MockDF:
                def memory_usage(self, deep=True):
                    class MemResult:
                        def sum(self):
                            return 1000
                    return MemResult()

            await cache.set('df_1', MockDF())
            await cache.set('df_2', MockDF())

            count = await cache.clear()
            assert count == 2

            stats = cache.get_stats()
            assert stats['entries'] == 0
            assert stats['current_memory_mb'] == 0.0

        asyncio.run(run())


class TestSQLConnectionPool:
    """Tests for ConnectionPool in data_tools.py."""

    def test_pool_limits_connections(self):
        """Pool should enforce max connections limit."""
        from tools.data_tools import ConnectionPool
        pool = ConnectionPool(max_connections=3)

        async def run():
            # Add 3 connections
            for i in range(3):
                await pool.set(f'conn_{i}', f'connection_{i}')

            # Adding 4th should evict oldest
            await pool.set('conn_3', 'connection_3')

            stats = pool.stats()
            assert stats['active_connections'] == 3

        asyncio.run(run())

    def test_pool_get_set(self):
        """Pool should store and retrieve connections."""
        from tools.data_tools import ConnectionPool
        pool = ConnectionPool()

        async def run():
            # Get non-existent should return None
            result = await pool.get('nonexistent')
            assert result is None

            # Set and get
            await pool.set('conn_1', 'connection_object')
            result = await pool.get('conn_1')
            assert result == 'connection_object'

        asyncio.run(run())

    def test_pool_remove(self):
        """Pool remove should delete connection."""
        from tools.data_tools import ConnectionPool
        pool = ConnectionPool()

        async def run():
            await pool.set('conn_1', 'connection_object')
            removed = await pool.remove('conn_1')
            assert removed is True

            result = await pool.get('conn_1')
            assert result is None

        asyncio.run(run())

    def test_pool_close_all(self):
        """Pool close_all should remove all connections."""
        from tools.data_tools import ConnectionPool
        pool = ConnectionPool()

        async def run():
            await pool.set('conn_1', 'conn_1')
            await pool.set('conn_2', 'conn_2')

            count = await pool.close_all()
            assert count == 2

            stats = pool.stats()
            assert stats['active_connections'] == 0

        asyncio.run(run())


class TestBoundedEmbeddingCache:
    """Tests for BoundedEmbeddingCache in semantic_memory.py."""

    def test_embedding_cache_limits_entries(self):
        """Cache should enforce max entries limit."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'semantic_memory',
            str(Path(__file__).parent.parent / 'core' / 'semantic_memory.py')
        )
        sm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sm)

        cache = sm.BoundedEmbeddingCache(max_entries=5)

        async def run():
            # Add 7 embeddings (exceeds limit)
            for i in range(7):
                await cache.set(f'key_{i}', [float(i)] * 128)

            stats = cache.stats()
            assert stats['entries'] == 5
            assert stats['evictions'] == 2  # 7 - 5 = 2 evicted

        asyncio.run(run())

    def test_embedding_cache_lru(self):
        """Cache should evict least recently used."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'semantic_memory',
            str(Path(__file__).parent.parent / 'core' / 'semantic_memory.py')
        )
        sm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sm)

        cache = sm.BoundedEmbeddingCache(max_entries=3)

        async def run():
            # Add items
            await cache.set('first', [1.0] * 128)
            await cache.set('second', [2.0] * 128)
            await cache.set('third', [3.0] * 128)

            # Access 'first' to make it recently used
            await cache.get('first')

            # Add 'fourth' - should evict 'second' (oldest unused)
            await cache.set('fourth', [4.0] * 128)

            # 'first' should exist
            result = await cache.get('first')
            assert result is not None

            # 'second' should be evicted
            result = await cache.get('second')
            assert result is None

        asyncio.run(run())

    def test_embedding_cache_stats(self):
        """Cache should track hits and misses."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'semantic_memory',
            str(Path(__file__).parent.parent / 'core' / 'semantic_memory.py')
        )
        sm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sm)

        cache = sm.BoundedEmbeddingCache(max_entries=10)

        async def run():
            await cache.set('key_1', [1.0] * 128)

            # Hit
            await cache.get('key_1')

            # Miss
            await cache.get('nonexistent')

            stats = cache.stats()
            assert stats['hits'] == 1
            assert stats['misses'] == 1

        asyncio.run(run())


class TestMemoryStress:
    """Stress tests for memory-bounded caches."""

    def test_dataframe_cache_stress(self):
        """Cache should handle many rapid operations."""
        from tools.data_tools import DataFrameCache
        cache = DataFrameCache(max_memory_mb=50, max_entries=100)

        async def run():
            class MockDF:
                def __init__(self, idx):
                    self.idx = idx
                def memory_usage(self, deep=True):
                    class MemResult:
                        def sum(self):
                            return 10000
                    return MemResult()

            # Rapid set operations
            tasks = [cache.set(f'df_{i}', MockDF(i)) for i in range(200)]
            await asyncio.gather(*tasks)

            stats = cache.get_stats()
            assert stats['entries'] <= 100  # Should not exceed limit
            assert stats['evictions'] > 0  # Some should be evicted

        asyncio.run(run())

    def test_concurrent_cache_operations(self):
        """Cache should be thread-safe for concurrent access."""
        from tools.data_tools import DataFrameCache
        cache = DataFrameCache()

        async def run():
            class MockDF:
                def memory_usage(self, deep=True):
                    class MemResult:
                        def sum(self):
                            return 100
                    return MemResult()

            async def writer(start, count):
                for i in range(start, start + count):
                    await cache.set(f'key_{i}', MockDF())

            async def reader(start, count):
                for i in range(start, start + count):
                    await cache.get(f'key_{i}')

            # Concurrent writers and readers
            await asyncio.gather(
                writer(0, 50),
                writer(50, 50),
                reader(0, 25),
                reader(25, 25),
            )

            # No exceptions means success

        asyncio.run(run())
