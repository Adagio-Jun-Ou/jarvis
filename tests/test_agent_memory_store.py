"""Tests for core/agent_memory_store.py — AgentMemoryStore."""
import time
import threading

import pytest

from core.agent_memory_store import AgentMemoryStore, get_agent_memory_store


class TestAgentMemoryStoreBasic:
    def test_put_and_get(self):
        store = AgentMemoryStore()
        assert store.put("ns1", "key1", "val1") is True
        assert store.get("ns1", "key1") == "val1"

    def test_get_missing_returns_default(self):
        store = AgentMemoryStore()
        assert store.get("ns1", "missing") is None
        assert store.get("ns1", "missing", "default") == "default"

    def test_put_overwrite(self):
        store = AgentMemoryStore()
        store.put("ns1", "key1", "v1")
        store.put("ns1", "key1", "v2")
        assert store.get("ns1", "key1") == "v2"

    def test_namespaced_isolation(self):
        store = AgentMemoryStore()
        store.put("ns1", "key1", "a")
        store.put("ns2", "key1", "b")
        assert store.get("ns1", "key1") == "a"
        assert store.get("ns2", "key1") == "b"

    def test_delete(self):
        store = AgentMemoryStore()
        store.put("ns1", "key1", "val")
        assert store.delete("ns1", "key1") is True
        assert store.get("ns1", "key1") is None

    def test_delete_missing(self):
        store = AgentMemoryStore()
        assert store.delete("ns1", "missing") is False

    def test_exists(self):
        store = AgentMemoryStore()
        store.put("ns1", "key1", "val")
        assert store.exists("ns1", "key1") is True
        assert store.exists("ns1", "missing") is False


class TestAgentMemoryStoreTTL:
    def test_ttl_expiry(self):
        store = AgentMemoryStore()
        store.put("ns1", "ephemeral", "data", ttl=0.05)
        assert store.get("ns1", "ephemeral") == "data"
        time.sleep(0.1)
        assert store.get("ns1", "ephemeral") is None

    def test_ttl_not_expired(self):
        store = AgentMemoryStore()
        store.put("ns1", "live", "data", ttl=60.0)
        assert store.get("ns1", "live") == "data"

    def test_default_ttl(self):
        store = AgentMemoryStore(default_ttl=0.05)
        store.put("ns1", "key1", "val")
        assert store.get("ns1", "key1") == "val"
        time.sleep(0.1)
        assert store.get("ns1", "key1") is None

    def test_exists_checks_expiry(self):
        store = AgentMemoryStore()
        store.put("ns1", "key1", "val", ttl=0.05)
        time.sleep(0.1)
        assert store.exists("ns1", "key1") is False


class TestAgentMemoryStoreLRU:
    def test_lru_eviction(self):
        store = AgentMemoryStore(max_keys=3)
        store.put("ns", "a", 1)
        store.put("ns", "b", 2)
        store.put("ns", "c", 3)
        store.put("ns", "d", 4)
        assert store.get("ns", "a") is None
        assert store.get("ns", "d") == 4

    def test_lru_access_moves_to_end(self):
        store = AgentMemoryStore(max_keys=3)
        store.put("ns", "a", 1)
        store.put("ns", "b", 2)
        store.put("ns", "c", 3)
        store.get("ns", "a")
        store.put("ns", "d", 4)
        assert store.get("ns", "a") == 1
        assert store.get("ns", "b") is None


class TestAgentMemoryStoreNamespaces:
    def test_list_keys(self):
        store = AgentMemoryStore()
        store.put("ns1", "a", 1)
        store.put("ns1", "b", 2)
        store.put("ns2", "c", 3)
        keys_ns1 = sorted(store.list_keys("ns1"))
        assert keys_ns1 == ["a", "b"]

    def test_list_namespaces(self):
        store = AgentMemoryStore()
        store.put("alpha", "k1", 1)
        store.put("beta", "k2", 2)
        nss = store.list_namespaces()
        assert "alpha" in nss
        assert "beta" in nss

    def test_namespace_stats(self):
        store = AgentMemoryStore()
        store.put("ns1", "a", 1)
        store.put("ns1", "b", 2)
        store.get("ns1", "a")
        stats = store.get_namespace_stats("ns1")
        assert stats["count"] == 2
        assert stats["total_accesses"] >= 1

    def test_clear_namespace(self):
        store = AgentMemoryStore()
        store.put("ns1", "a", 1)
        store.put("ns1", "b", 2)
        store.put("ns2", "c", 3)
        removed = store.clear_namespace("ns1")
        assert removed == 2
        assert store.get("ns1", "a") is None
        assert store.get("ns2", "c") == 3


class TestAgentMemoryStoreValidation:
    def test_key_too_long(self):
        store = AgentMemoryStore(max_key_size=10)
        assert store.put("ns", "x" * 20, "val") is False

    def test_value_too_large(self):
        store = AgentMemoryStore(max_value_size=100)
        assert store.put("ns", "k", "x" * 200) is False

    def test_various_value_types(self):
        store = AgentMemoryStore()
        store.put("ns", "int", 42)
        store.put("ns", "list", [1, 2, 3])
        store.put("ns", "dict", {"a": 1})
        store.put("ns", "bool", True)
        assert store.get("ns", "int") == 42
        assert store.get("ns", "list") == [1, 2, 3]
        assert store.get("ns", "dict") == {"a": 1}


class TestAgentMemoryStoreSnapshot:
    def test_snapshot_and_restore(self):
        store = AgentMemoryStore()
        store.put("ns1", "a", 1)
        store.put("ns1", "b", 2)
        snap = store.snapshot()
        assert "entries" in snap
        assert "stats" in snap

        store2 = AgentMemoryStore()
        count = store2.restore(snap)
        assert count == 2
        assert store2.get("ns1", "a") == 1
        assert store2.get("ns1", "b") == 2

    def test_snapshot_excludes_expired(self):
        store = AgentMemoryStore()
        store.put("ns", "live", "ok", ttl=60)
        store.put("ns", "dead", "gone", ttl=0.01)
        time.sleep(0.05)
        snap = store.snapshot()
        assert "ns:live" in snap["entries"]
        assert "ns:dead" not in snap["entries"]


class TestAgentMemoryStoreCleanup:
    def test_cleanup_expired(self):
        store = AgentMemoryStore()
        store.put("ns", "live", "ok", ttl=60)
        store.put("ns", "dead", "gone", ttl=0.01)
        time.sleep(0.05)
        removed = store.cleanup()
        assert removed == 1
        assert store.get("ns", "live") == "ok"


class TestAgentMemoryStoreThreadSafety:
    def test_concurrent_puts(self):
        store = AgentMemoryStore(max_keys=1000)
        errors = []

        def writer(tid):
            try:
                for i in range(50):
                    store.put(f"ns{tid}", f"k{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        total = len(store.list_keys())
        assert total > 0


class TestAgentMemoryStoreStatus:
    def test_get_status(self):
        store = AgentMemoryStore()
        store.put("ns1", "a", 1)
        status = store.get_status()
        assert "total_keys" in status
        assert "max_keys" in status
        assert "namespaces" in status
        assert "stats" in status
        assert status["total_keys"] >= 1


class TestGetAgentMemoryStore:
    def test_singleton(self):
        a = get_agent_memory_store()
        b = get_agent_memory_store()
        assert a is b