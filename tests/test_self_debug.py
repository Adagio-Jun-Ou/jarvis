"""Tests for self-refine self-debugging capabilities."""
import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from core.self_refine import (
    SelfRefineAgent,
    RefinementTask,
    RefinementPriority,
    DebugSession,
    RefinementState,
)


class TestDebugSession:
    def test_debug_session_creation(self):
        session = DebugSession(
            id="debug_001",
            task_id="task_001",
            started_at=datetime.now(),
            error_type="import_error",
            error_message="No module named 'xyz'",
        )
        assert session.id == "debug_001"
        assert session.error_type == "import_error"
        assert session.attempts == 0
        assert session.success is False

    def test_debug_session_with_file_context(self):
        session = DebugSession(
            id="debug_002",
            task_id="task_002",
            started_at=datetime.now(),
            error_type="attribute_error",
            error_message="'NoneType' object has no attribute 'foo'",
            file_path="core/example.py",
            line_number=42,
        )
        assert session.file_path == "core/example.py"
        assert session.line_number == 42


class TestErrorIdentification:
    @pytest.fixture
    def agent(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "{}"}})
        return SelfRefineAgent(mock_llm, base_path=Path(__file__).parent.parent)

    def test_identify_import_error(self, agent):
        error_output = "ImportError: No module named 'missing_module'"
        error_type = agent._identify_error_type(error_output)
        assert error_type == "import_error"

    def test_identify_attribute_error(self, agent):
        error_output = "AttributeError: 'str' object has no attribute 'missing_attr'"
        error_type = agent._identify_error_type(error_output)
        assert error_type == "attribute_error"

    def test_identify_type_error(self, agent):
        error_output = "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
        error_type = agent._identify_error_type(error_output)
        assert error_type == "type_error"

    def test_identify_key_error(self, agent):
        error_output = "KeyError: 'missing_key'"
        error_type = agent._identify_error_type(error_output)
        assert error_type == "key_error"

    def test_identify_index_error(self, agent):
        error_output = "IndexError: list index out of range"
        error_type = agent._identify_error_type(error_output)
        assert error_type == "index_error"

    def test_identify_name_error(self, agent):
        error_output = "NameError: name 'undefined_var' is not defined"
        error_type = agent._identify_error_type(error_output)
        assert error_type == "name_error"

    def test_identify_syntax_error(self, agent):
        error_output = "SyntaxError: invalid syntax"
        error_type = agent._identify_error_type(error_output)
        assert error_type == "syntax_error"

    def test_identify_test_failure(self, agent):
        error_output = "FAILED test_example.py::test_something - AssertionError"
        error_type = agent._identify_error_type(error_output)
        assert error_type == "test_failure"

    def test_identify_unknown_error(self, agent):
        error_output = "Some random error message"
        error_type = agent._identify_error_type(error_output)
        assert error_type == "unknown"


class TestErrorContextExtraction:
    @pytest.fixture
    def agent(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "{}"}})
        return SelfRefineAgent(mock_llm, base_path=Path(__file__).parent.parent)

    @pytest.mark.asyncio
    async def test_extract_traceback_context(self, agent):
        session = DebugSession(
            id="debug_001",
            task_id="task_001",
            started_at=datetime.now(),
            error_type="attribute_error",
            error_message="Test error",
        )
        
        error_output = '''
Traceback (most recent call last):
  File "core/example.py", line 42, in <module>
    result = obj.missing_attr
AttributeError: 'NoneType' object has no attribute 'missing_attr'
'''
        
        await agent._extract_error_context(session, error_output)
        assert session.file_path == "core/example.py"
        assert session.line_number == 42

    @pytest.mark.asyncio
    async def test_extract_test_file_context(self, agent):
        session = DebugSession(
            id="debug_002",
            task_id="task_002",
            started_at=datetime.now(),
            error_type="test_failure",
            error_message="Test failed",
        )
        
        error_output = "FAILED tests/test_example.py::test_something - AssertionError"
        
        await agent._extract_error_context(session, error_output)
        assert "tests/test_example.py" in (session.file_path or "")


class TestErrorDiagnosis:
    @pytest.fixture
    def agent(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "{}"}})
        return SelfRefineAgent(mock_llm, base_path=Path(__file__).parent.parent)

    @pytest.mark.asyncio
    async def test_diagnose_import_error(self, agent):
        session = DebugSession(
            id="debug_001",
            task_id="task_001",
            started_at=datetime.now(),
            error_type="import_error",
            error_message="ImportError: No module named 'xyz'",
        )
        task = RefinementTask(
            id="task_001",
            title="Test task",
            description="Test",
            file_path="core/test.py",
            priority=RefinementPriority.MEDIUM,
            category="bug",
            original_code="",
            proposed_code="",
            reasoning="",
            risk_level="low",
        )
        
        diagnosis = await agent._debug_import_error(session, task, session.error_message)
        assert diagnosis["root_cause"] is not None
        assert "missing" in diagnosis["root_cause"].lower() or "import" in diagnosis["root_cause"].lower()

    @pytest.mark.asyncio
    async def test_diagnose_key_error(self, agent):
        session = DebugSession(
            id="debug_002",
            task_id="task_002",
            started_at=datetime.now(),
            error_type="key_error",
            error_message="KeyError: 'missing_key'",
        )
        task = RefinementTask(
            id="task_002",
            title="Test task",
            description="Test",
            file_path="core/test.py",
            priority=RefinementPriority.MEDIUM,
            category="bug",
            original_code="",
            proposed_code="",
            reasoning="",
            risk_level="low",
        )
        
        diagnosis = await agent._debug_key_error(session, task, session.error_message)
        assert "missing_key" in diagnosis["root_cause"]

    @pytest.mark.asyncio
    async def test_diagnose_name_error(self, agent):
        session = DebugSession(
            id="debug_003",
            task_id="task_003",
            started_at=datetime.now(),
            error_type="name_error",
            error_message="NameError: name 'undefined_var' is not defined",
        )
        task = RefinementTask(
            id="task_003",
            title="Test task",
            description="Test",
            file_path="core/test.py",
            priority=RefinementPriority.MEDIUM,
            category="bug",
            original_code="",
            proposed_code="",
            reasoning="",
            risk_level="low",
        )
        
        diagnosis = await agent._debug_name_error(session, task, session.error_message)
        assert "undefined_var" in diagnosis["root_cause"]


class TestLearnedFixes:
    @pytest.fixture
    def agent(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "{}"}})
        agent = SelfRefineAgent(mock_llm, base_path=Path(__file__).parent.parent)
        agent._learned_fixes = {
            "import_error": "from correct_module import thing",
        }
        return agent

    def test_learned_fixes_persist(self, agent):
        assert "import_error" in agent._learned_fixes
        assert agent._learned_fixes["import_error"] == "from correct_module import thing"

    def test_debug_stats(self, agent):
        stats = agent.get_debug_stats()
        assert "total_debug_sessions" in stats
        assert "learned_fixes" in stats
        assert stats["learned_fixes"] >= 1


class TestSelfDebugIntegration:
    @pytest.fixture
    def agent(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={
            "message": {
                "content": '{"root_cause": "Test diagnosis", "suggested_fix": "", "explanation": "Test"}'
            }
        })
        return SelfRefineAgent(mock_llm, base_path=Path(__file__).parent.parent)

    @pytest.mark.asyncio
    async def test_self_debug_nonexistent_task(self, agent):
        success, message = await agent.self_debug("nonexistent_task", "Some error")
        assert success is False
        assert "not found" in message

    def test_debug_state_tracking(self, agent):
        assert agent._state == RefinementState.IDLE
        agent._state = RefinementState.DEBUGGING
        assert agent._state == RefinementState.DEBUGGING


class TestErrorPatterns:
    def test_error_patterns_defined(self):
        from core.self_refine import SelfRefineAgent
        
        assert hasattr(SelfRefineAgent, 'ERROR_PATTERNS')
        patterns = SelfRefineAgent.ERROR_PATTERNS
        
        assert "import_error" in patterns
        assert "attribute_error" in patterns
        assert "type_error" in patterns
        assert "key_error" in patterns
        assert "syntax_error" in patterns
        assert "test_failure" in patterns

    def test_error_patterns_have_handlers(self):
        from core.self_refine import SelfRefineAgent
        
        patterns = SelfRefineAgent.ERROR_PATTERNS
        for error_type, config in patterns.items():
            assert "pattern" in config, f"Missing pattern for {error_type}"
            assert "handler" in config, f"Missing handler for {error_type}"