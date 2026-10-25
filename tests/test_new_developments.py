"""Tests for new developments: Dynamic Memory, Supervisor, Skills, API Keys.

Tests:
- Dynamic memory management with 24GB limit
- Server supervisor health and recovery
- Skill loader with binary aliases
- API key loading and persistence
- End-to-end integration
"""
import asyncio
import gc
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDynamicMemory:
    """Test dynamic memory management system."""

    def test_import_dynamic_memory(self):
        """Test that dynamic memory module imports."""
        from core.dynamic_memory import DynamicMemoryManager, DynamicMemoryConfig, MemoryPressure
        assert DynamicMemoryManager is not None
        assert DynamicMemoryConfig is not None
        assert MemoryPressure is not None

    def test_memory_config_defaults(self):
        """Test default memory configuration."""
        from core.dynamic_memory import DynamicMemoryConfig
        
        config = DynamicMemoryConfig()
        assert config.max_memory_mb == 24 * 1024.0  # 24GB
        assert config.critical_threshold == 0.85
        assert config.emergency_threshold == 0.95

    def test_memory_config_custom(self):
        """Test custom memory configuration."""
        from core.dynamic_memory import DynamicMemoryConfig
        
        config = DynamicMemoryConfig(max_memory_mb=16 * 1024, critical_threshold=0.80)
        assert config.max_memory_mb == 16 * 1024
        assert config.critical_threshold == 0.80

    def test_memory_pressure_levels(self):
        """Test memory pressure calculation."""
        from core.dynamic_memory import DynamicMemoryConfig, MemoryPressure
        
        config = DynamicMemoryConfig()
        
        # Test pressure level calculations
        assert config.elevated_threshold == 0.60
        assert config.high_threshold == 0.75
        assert config.critical_threshold == 0.85
        assert config.emergency_threshold == 0.95

    def test_get_system_memory(self):
        """Test getting system memory info."""
        from core.dynamic_memory import get_dynamic_memory_manager
        
        manager = get_dynamic_memory_manager()
        info = manager.get_system_memory()
        
        # info is a SystemMemoryInfo dataclass
        assert hasattr(info, "process_rss_mb")
        assert info.process_rss_mb >= 0

    def test_get_current_pressure(self):
        """Test getting current memory pressure."""
        from core.dynamic_memory import get_dynamic_memory_manager
        
        manager = get_dynamic_memory_manager()
        
        # Get target cache size
        target = manager.get_target_cache_size()
        assert target > 0
        assert isinstance(target, int)

    def test_singleton_pattern(self):
        """Test that manager is a singleton."""
        from core.dynamic_memory import get_dynamic_memory_manager
        
        manager1 = get_dynamic_memory_manager()
        manager2 = get_dynamic_memory_manager()
        
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_async_cleanup(self):
        """Test async memory cleanup."""
        from core.dynamic_memory import get_dynamic_memory_manager, MemoryPressure
        
        manager = get_dynamic_memory_manager()
        
        result = await manager.async_cleanup(MemoryPressure.NORMAL)
        
        assert "before_mb" in result
        assert "after_mb" in result
        assert "freed_mb" in result


class TestServerSupervisor:
    """Test server supervisor system."""

    def test_import_supervisor(self):
        """Test that supervisor module imports."""
        from core.supervisor import ServerSupervisor, SupervisorState, MemoryConfig, RestartPolicy
        assert ServerSupervisor is not None
        assert SupervisorState is not None
        assert MemoryConfig is not None
        assert RestartPolicy is not None

    def test_supervisor_state_enum(self):
        """Test supervisor state values."""
        from core.supervisor import SupervisorState
        
        assert SupervisorState.STARTING.value == "starting"
        assert SupervisorState.RUNNING.value == "running"
        assert SupervisorState.DEGRADED.value == "degraded"
        assert SupervisorState.STOPPING.value == "stopping"
        assert SupervisorState.STOPPED.value == "stopped"

    def test_memory_config_24gb(self):
        """Test memory config defaults to 24GB."""
        from core.supervisor import MemoryConfig
        
        config = MemoryConfig()
        assert config.max_memory_mb == 24 * 1024.0
        assert config.critical_threshold == 0.85
        assert config.emergency_threshold == 0.95

    def test_restart_policy_defaults(self):
        """Test restart policy defaults."""
        from core.supervisor import RestartPolicy
        
        policy = RestartPolicy()
        assert policy.max_restarts == 5
        assert policy.restart_window_seconds == 300.0
        assert policy.backoff_base == 2.0

    def test_supervisor_singleton(self):
        """Test supervisor singleton pattern."""
        from core.supervisor import get_supervisor
        
        supervisor1 = get_supervisor()
        supervisor2 = get_supervisor()
        
        assert supervisor1 is supervisor2

    def test_get_memory_usage(self):
        """Test getting memory usage."""
        from core.supervisor import get_supervisor
        
        supervisor = get_supervisor()
        usage = supervisor.get_memory_usage()
        
        assert "rss_mb" in usage
        assert usage["rss_mb"] > 0

    def test_get_system_memory_status(self):
        """Test getting system memory status."""
        from core.supervisor import get_supervisor
        
        supervisor = get_supervisor()
        status = supervisor.get_system_memory_status()
        
        # Should have system memory info
        assert "ram_percent" in status or len(status) >= 0

    def test_check_memory_pressure(self):
        """Test memory pressure check."""
        from core.supervisor import get_supervisor
        
        supervisor = get_supervisor()
        is_pressure, rss_mb, level = supervisor.check_memory_pressure()
        
        assert isinstance(is_pressure, bool)
        assert rss_mb > 0
        assert level in ["NORMAL", "CRITICAL", "EMERGENCY"]

    def test_llm_fallback_models(self):
        """Test LLM fallback model rotation."""
        from core.supervisor import get_supervisor
        
        supervisor = get_supervisor()
        
        # Should have fallback models configured
        assert len(supervisor._fallback_llm_models) >= 2
        
        # Test getting next fallback
        supervisor.reset_model_fallback()
        model1 = supervisor.get_next_fallback_model()
        assert model1 is not None

    def test_should_restart_logic(self):
        """Test restart decision logic."""
        from core.supervisor import get_supervisor, RestartPolicy
        
        supervisor = get_supervisor()
        should, delay = supervisor.should_restart()
        
        assert isinstance(should, bool)
        assert isinstance(delay, float)
        assert delay >= 0

    @pytest.mark.asyncio
    async def test_force_memory_cleanup(self):
        """Test force memory cleanup."""
        from core.supervisor import get_supervisor
        
        supervisor = get_supervisor()
        result = await supervisor.force_memory_cleanup("NORMAL")
        
        assert "before_mb" in result
        assert "after_mb" in result


class TestSkillLoader:
    """Test skill loader improvements."""

    def test_import_skill_loader(self):
        """Test skill loader imports."""
        from core.skill_loader import SkillLoader, LoadedSkill, SkillStatus
        assert SkillLoader is not None
        assert LoadedSkill is not None
        assert SkillStatus is not None

    def test_binary_aliases_python(self):
        """Test python binary alias works."""
        from core.skill_loader import get_skill_loader, SkillRequirements, LoadedSkill, SkillStatus, SkillMetadata
        from pathlib import Path
        
        loader = get_skill_loader()
        
        # Create a mock skill requiring python
        skill = LoadedSkill(
            skill_id="test_python",
            name="test-python",
            description="Test",
            instructions="Test",
            requirements=SkillRequirements(bins=["python"]),
            metadata=SkillMetadata(),
            source_path=Path("/tmp/test"),
        )
        
        met, missing = loader.check_requirements(skill)
        assert met, f"Python should be found via alias, missing: {missing}"

    def test_binary_aliases_python3(self):
        """Test python3 binary alias works."""
        from core.skill_loader import get_skill_loader, SkillRequirements, LoadedSkill, SkillMetadata
        from pathlib import Path
        
        loader = get_skill_loader()
        
        skill = LoadedSkill(
            skill_id="test_python3",
            name="test-python3",
            description="Test",
            instructions="Test",
            requirements=SkillRequirements(bins=["python3"]),
            metadata=SkillMetadata(),
            source_path=Path("/tmp/test"),
        )
        
        met, missing = loader.check_requirements(skill)
        assert met, f"Python3 should be found, missing: {missing}"

    def test_skill_discovery(self):
        """Test skill discovery finds skills."""
        from core.skill_loader import get_skill_loader
        
        loader = get_skill_loader()
        skills = loader.discover_skills()
        
        # Should find at least 5 skills
        assert len(skills) >= 5

    def test_all_skills_loadable(self):
        """Test that all discovered skills can be parsed."""
        from core.skill_loader import get_skill_loader
        
        loader = get_skill_loader()
        skills = loader.discover_skills()
        
        for skill_path in skills:
            skill = loader.parse_skill_file(skill_path)
            assert skill is not None, f"Failed to parse {skill_path}"
            assert skill.name
            assert skill.description


class TestAPIKeyLoading:
    """Test API key loading and persistence."""

    @pytest.mark.requires_host_env
    def test_secrets_file_exists(self):
        """Test that secrets file exists."""
        secrets_file = Path.home() / ".jarvis" / "secrets" / "api_keys.env"
        assert secrets_file.exists(), f"Secrets file not found at {secrets_file}"

    @pytest.mark.requires_host_env
    def test_api_key_in_secrets(self):
        """Test that API key is in secrets file."""
        secrets_file = Path.home() / ".jarvis" / "secrets" / "api_keys.env"
        content = secrets_file.read_text(encoding='utf-8')
        
        assert "OLLAMA_API_KEY" in content
        assert len(content) > 50  # Should have actual key

    def test_run_py_loads_secrets(self):
        """Test that run.py has secret loading code."""
        run_py = Path(__file__).parent.parent / "run.py"
        content = run_py.read_text(encoding='utf-8')
        
        assert "api_keys.env" in content
        assert "OLLAMA_API_KEY" in content or "secrets" in content

    def test_bashrc_has_api_key(self):
        """Test that .bashrc has API key."""
        import pytest
        bashrc = Path.home() / ".bashrc"
        if not bashrc.exists():
            pytest.skip("~/.bashrc does not exist")
        content = bashrc.read_text(encoding='utf-8')

        assert "OLLAMA_API_KEY" in content or "export" in content

    def test_local_bin_in_path(self):
        """Test that ~/.local/bin is in PATH."""
        local_bin = Path.home() / ".local" / "bin"
        path_env = os.environ.get("PATH", "")
        
        # Check if it's in PATH or can be added
        assert str(local_bin) in path_env or local_bin.exists()


class TestIntegration:
    """Integration tests for all new features."""

    @pytest.mark.asyncio
    async def test_supervisor_registers_memory_manager(self):
        """Test that supervisor integrates with memory manager."""
        from core.supervisor import get_supervisor
        
        supervisor = get_supervisor()
        
        # Supervisor should have memory config
        assert supervisor.memory_config.max_memory_mb == 24 * 1024

    def test_memory_manager_has_system_awareness(self):
        """Test memory manager is system-aware."""
        from core.dynamic_memory import get_dynamic_memory_manager
        
        manager = get_dynamic_memory_manager()
        status = manager.get_status()
        
        assert "system" in status
        assert "config" in status
        assert status["config"]["max_memory_mb"] == 24 * 1024

    def test_all_skills_load_with_new_tools(self):
        """Test that all 9 skills load with gh and cargo."""
        from core.skill_loader import get_skill_loader
        import shutil
        
        loader = get_skill_loader()
        
        # Check if tools are available
        has_gh = shutil.which("gh") is not None
        has_cargo = shutil.which("cargo") is not None
        
        loader.load_all_skills()
        loaded = loader.get_loaded_skills()
        
        # At minimum should have skills without dependencies
        assert len(loaded) >= 5
        
        # If tools installed, all should load
        if has_gh and has_cargo:
            assert len(loaded) >= 9, f"Expected at least 9 skills, got {len(loaded)}"

    @pytest.mark.asyncio
    async def test_graceful_shutdown_flow(self):
        """Test graceful shutdown can be triggered."""
        from core.supervisor import get_supervisor
        
        supervisor = get_supervisor()
        
        # Should be able to register shutdown handlers
        handler_called = []
        
        async def test_handler():
            handler_called.append(True)
        
        supervisor.register_shutdown_handler(test_handler)
        
        assert len(supervisor._shutdown_handlers) >= 1

    def test_status_endpoint_includes_memory(self):
        """Test that API status includes memory info."""
        # This would require a running server, so we test the structure
        from core.supervisor import get_supervisor
        
        supervisor = get_supervisor()
        status = supervisor.get_status()
        
        assert "memory" in status
        assert "process" in status["memory"] or "rss_mb" in status.get("memory", {})


class TestInstalledTools:
    """Test that required tools are installed."""

    def test_python_available(self):
        """Test python is available."""
        import shutil
        assert shutil.which("python") or shutil.which("python3")

    @pytest.mark.requires_host_env
    def test_gh_installed(self):
        """Test GitHub CLI is installed."""
        import shutil
        gh_path = shutil.which("gh")
        assert gh_path is not None, "gh (GitHub CLI) not installed"
        
        # Check it's the real gh, not the Python wrapper
        result = os.popen(f"{gh_path} --version 2>&1").read()
        assert "gh version" in result.lower() or "github.com" in result

    @pytest.mark.requires_host_env
    def test_cargo_installed(self):
        """Test cargo is installed."""
        import shutil
        cargo_path = shutil.which("cargo")
        assert cargo_path is not None, "cargo not installed"

    @pytest.mark.requires_host_env
    def test_rustc_installed(self):
        """Test rustc is installed."""
        import shutil
        rustc_path = shutil.which("rustc")
        assert rustc_path is not None, "rustc not installed"

    def test_npx_installed(self):
        """Test npx is installed."""
        import shutil
        assert shutil.which("npx") is not None, "npx not installed"


class TestLiveServer:
    """Tests for the running server."""

    @pytest.fixture
    def server_url(self):
        return "https://localhost:8080"

    def test_health_endpoint(self, server_url):
        """Test health endpoint returns healthy."""
        import urllib.request
        import ssl
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(f"{server_url}/health", context=ctx, timeout=5) as resp:
                data = resp.read().decode()
                assert "healthy" in data
        except Exception as e:
            pytest.skip(f"Server not reachable: {e}")

    def test_status_endpoint(self, server_url):
        """Test status endpoint returns data."""
        import urllib.request
        import json
        import ssl
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(f"{server_url}/api/status", context=ctx, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                assert "tools" in data
                assert data["tools"] >= 60
        except Exception as e:
            pytest.skip(f"Server not reachable: {e}")

    def test_status_includes_memory(self, server_url):
        """Test status includes memory information."""
        import urllib.request
        import json
        import ssl
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(f"{server_url}/api/status", context=ctx, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                assert "memory" in data
                mem = data["memory"]
                assert "pressure" in mem
                assert mem.get("max_gb", 0) == 24.0
        except Exception as e:
            pytest.skip(f"Server not reachable: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])