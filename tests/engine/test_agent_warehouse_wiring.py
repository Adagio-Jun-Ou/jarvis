"""Agent tools must hit the REAL warehouse in production, not the in-memory
fake. _warehouse() previously returned the fake unconditionally, so agent-
written tasks/memories would never persist to Unity Catalog in prod.
"""
import alex.warehouse as whmod
from alex.warehouse import InMemoryWarehouse


def _reset(monkeypatch, mode):
    monkeypatch.setenv("ALEX_ENV", mode)
    from agent.tools import impls
    from alex import config
    config._settings = None
    impls._prod_warehouse = None
    return impls


def test_dev_uses_in_memory_fake(monkeypatch):
    impls = _reset(monkeypatch, "dev")
    from alex import testing
    testing._fakes = None
    assert isinstance(impls._warehouse(), InMemoryWarehouse)


def test_production_uses_real_warehouse_client(monkeypatch):
    impls = _reset(monkeypatch, "production")

    class _Sentinel:
        def __init__(self, *a, **k):
            pass

    # Avoid constructing a real WorkspaceClient (no creds in tests).
    monkeypatch.setattr(whmod, "WarehouseClient", _Sentinel)
    wh = impls._warehouse()
    assert isinstance(wh, _Sentinel)
    assert not isinstance(wh, InMemoryWarehouse)
    # reset so other tests get a fresh singleton
    impls._prod_warehouse = None


def test_vm_warehouse_dev_uses_fake(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from agent import vm
    from alex import config, testing
    config._settings = None
    testing._fakes = None
    vm._prod_warehouse = None
    assert isinstance(vm._vm_warehouse(), InMemoryWarehouse)


def test_vm_warehouse_prod_uses_real_client(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "production")
    from agent import vm
    from alex import config
    config._settings = None
    vm._prod_warehouse = None

    class _Sentinel:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(whmod, "WarehouseClient", _Sentinel)
    assert isinstance(vm._vm_warehouse(), _Sentinel)
    vm._prod_warehouse = None
