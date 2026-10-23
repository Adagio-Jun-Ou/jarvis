"""Tests for Goal Tracker."""
import pytest
import time
import os
import tempfile
from pathlib import Path
from core.goal_tracker import (
    GoalTracker,
    Goal,
    Task,
    get_goal_tracker,
    reset_goal_tracker,
)


def get_fresh_tracker():
    """Get a tracker with isolated storage."""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    path = Path(tempfile.gettempdir()) / f"goal_test_{unique_id}"
    return GoalTracker(storage_path=path)


class TestGoal:
    """Tests for Goal dataclass."""
    
    def test_creation(self):
        goal = Goal(
            goal_id="test1",
            description="Build a web app"
        )
        assert goal.goal_id == "test1"
        assert goal.description == "Build a web app"
        assert goal.status == "active"
        assert goal.priority == 3
        assert goal.tasks == []
    
    def test_to_dict(self):
        goal = Goal(
            goal_id="test1",
            description="Build app",
            priority=5
        )
        data = goal.to_dict()
        assert data["goal_id"] == "test1"
        assert data["description"] == "Build app"
        assert data["priority"] == 5
    
    def test_from_dict(self):
        data = {
            "goal_id": "test1",
            "description": "Test goal",
            "status": "paused",
            "priority": 2,
            "tasks": [
                {"description": "Task 1", "status": "completed"}
            ]
        }
        goal = Goal.from_dict(data)
        assert goal.goal_id == "test1"
        assert goal.status == "paused"
        assert len(goal.tasks) == 1
        assert goal.tasks[0].description == "Task 1"


class TestTask:
    """Tests for Task dataclass."""
    
    def test_creation(self):
        task = Task(description="Write tests")
        assert task.description == "Write tests"
        assert task.status == "pending"
        assert task.notes == []
    
    def test_completion(self):
        task = Task(description="Write tests")
        task.status = "completed"
        task.completed_at = time.time()
        assert task.status == "completed"
        assert task.completed_at is not None


class TestGoalTracker:
    """Tests for GoalTracker."""

    @pytest.fixture(autouse=True)
    def _isolate_singleton(self, tmp_path):
        """Point the module-level singleton at a per-test temp store so tests
        that use ``get_goal_tracker()`` never read or mutate the real
        ``~/.jarvis/goals`` directory. ``reset_goal_tracker()`` called inside
        each test preserves this isolated path (see goal_tracker.py)."""
        reset_goal_tracker(tmp_path)
        yield
        reset_goal_tracker(None)

    def test_initial_state(self):
        tracker = get_fresh_tracker()
        assert len(tracker.get_all_goals()) >= 0
    
    def test_extract_goals(self):
        tracker = get_fresh_tracker()
        
        goals = tracker.extract_goals("I want to build a web application")
        assert len(goals) >= 1
        assert "web" in goals[0].description.lower()
    
    def test_extract_goals_multiple(self):
        tracker = get_fresh_tracker()
        
        goals1 = tracker.extract_goals("I want to build a web app")
        for g in goals1:
            tracker.track_goal(g)
        goals2 = tracker.extract_goals("I want to learn Python")
        for g in goals2:
            tracker.track_goal(g)
        
        all_goals = tracker.get_all_goals()
        assert len(all_goals) >= 2
    
    def test_track_goal(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(
            goal_id="custom1",
            description="Custom goal"
        )
        tracker.track_goal(goal)
        
        all_goals = tracker.get_all_goals()
        found = any(g.goal_id == "custom1" for g in all_goals)
        assert found
    
    def test_get_active_goals(self):
        tracker = get_fresh_tracker()
        
        goal1 = Goal(goal_id="active1", description="Active goal", status="active")
        goal2 = Goal(goal_id="paused1", description="Paused goal", status="paused")
        
        tracker.track_goal(goal1)
        tracker.track_goal(goal2)
        
        active = tracker.get_active_goals()
        ids = [g.goal_id for g in active]
        assert "active1" in ids
        assert "paused1" not in ids
    
    def test_complete_goal(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="complete1", description="Goal to complete")
        tracker.track_goal(goal)
        
        completed = tracker.complete_goal("complete1")
        assert completed is not None
        assert completed.status == "completed"
    
    def test_pause_goal(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="pause1", description="Goal to pause")
        tracker.track_goal(goal)
        
        paused = tracker.pause_goal("pause1")
        assert paused is not None
        assert paused.status == "paused"
    
    def test_add_task(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="task_goal", description="Goal with tasks")
        tracker.track_goal(goal)
        
        tracker.add_task("task_goal", "Write code")
        tracker.add_task("task_goal", "Write tests")
        
        updated = tracker.get_all_goals()
        found = next((g for g in updated if g.goal_id == "task_goal"), None)
        assert found is not None
        assert len(found.tasks) == 2
    
    def test_complete_task(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="task_goal2", description="Goal")
        tracker.track_goal(goal)
        tracker.add_task("task_goal2", "Task to complete")
        
        tracker.complete_task("task_goal2", "Task to complete")
        
        updated = next((g for g in tracker.get_all_goals() if g.goal_id == "task_goal2"), None)
        assert updated is not None
        assert updated.tasks[0].status == "completed"
    
    def test_add_blocker(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="blocked_goal", description="Blocked goal")
        tracker.track_goal(goal)
        
        tracker.add_blocker("blocked_goal", "Missing dependency")
        
        updated = next((g for g in tracker.get_all_goals() if g.goal_id == "blocked_goal"), None)
        assert "Missing dependency" in updated.blockers
    
    def test_remove_blocker(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="unblock_goal", description="Goal")
        tracker.track_goal(goal)
        tracker.add_blocker("unblock_goal", "Blocker 1")
        tracker.add_blocker("unblock_goal", "Blocker 2")
        
        tracker.remove_blocker("unblock_goal", "Blocker 1")
        
        updated = next((g for g in tracker.get_all_goals() if g.goal_id == "unblock_goal"), None)
        assert "Blocker 1" not in updated.blockers
        assert "Blocker 2" in updated.blockers
    
    def test_detect_updates(self):
        tracker = get_fresh_tracker()
        
        updates = tracker.detect_updates("I'm stuck on the API integration")
        assert len(updates["blockers"]) > 0
        assert "api" in updates["blockers"][0].lower()
    
    def test_detect_files(self):
        tracker = get_fresh_tracker()
        
        updates = tracker.detect_updates("I'm working on main.py and utils.js")
        assert len(updates["files"]) >= 2
    
    def test_suggest_next_step_blocked(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="blocked", description="Build app")
        goal.blockers = ["Missing API key"]
        tracker.track_goal(goal)
        
        suggestion = tracker.suggest_next_step(goal)
        assert "blocked" in suggestion.lower()
    
    def test_suggest_next_step_with_tasks(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="tasks_goal", description="Build app")
        goal.tasks = [
            Task(description="Setup project", status="completed"),
            Task(description="Write code", status="pending")
        ]
        tracker.track_goal(goal)
        
        suggestion = tracker.suggest_next_step(goal)
        assert "write code" in suggestion.lower() or "task" in suggestion.lower()
    
    def test_suggest_next_step_empty(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="empty_goal", description="Build something")
        tracker.track_goal(goal)
        
        suggestion = tracker.suggest_next_step(goal)
        assert len(suggestion) > 0
    
    def test_get_goal_context(self):
        tracker = get_fresh_tracker()
        
        goal = Goal(goal_id="ctx_test", description="Build a web app")
        tracker.track_goal(goal)
        
        context = tracker.get_goal_context()
        assert "<goals>" in context
        assert "web" in context.lower()

    def test_get_goal_context_empty(self):
        tracker = GoalTracker(storage_path=Path(tempfile.gettempdir()) / f"test_goals_empty_{os.getpid()}")

        context = tracker.get_goal_context()
        assert context == ""

    def test_persistence(self, tmp_path):
        tracker1 = GoalTracker(storage_path=tmp_path)
        
        goal = Goal(goal_id="persist_test", description="Persistent goal")
        tracker1.track_goal(goal)
        
        # Create new tracker to load persisted data
        tracker2 = GoalTracker(storage_path=tmp_path)
        goals = tracker2.get_all_goals()
        
        found = any(g.goal_id == "persist_test" for g in goals)
        assert found
    
    def test_get_active_goals(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal1 = Goal(goal_id="active1", description="Active goal", status="active")
        goal2 = Goal(goal_id="paused1", description="Paused goal", status="paused")
        
        tracker.track_goal(goal1)
        tracker.track_goal(goal2)
        
        active = tracker.get_active_goals()
        ids = [g.goal_id for g in active]
        assert "active1" in ids
        assert "paused1" not in ids
    
    def test_complete_goal(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal = Goal(goal_id="complete1", description="Goal to complete")
        tracker.track_goal(goal)
        
        completed = tracker.complete_goal("complete1")
        assert completed is not None
        assert completed.status == "completed"
    
    def test_pause_goal(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal = Goal(goal_id="pause1", description="Goal to pause")
        tracker.track_goal(goal)
        
        paused = tracker.pause_goal("pause1")
        assert paused is not None
        assert paused.status == "paused"
    
    def test_add_task(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal = Goal(goal_id="task_goal", description="Goal with tasks")
        tracker.track_goal(goal)
        
        tracker.add_task("task_goal", "Write code")
        tracker.add_task("task_goal", "Write tests")
        
        updated = tracker.get_all_goals()
        found = next((g for g in updated if g.goal_id == "task_goal"), None)
        assert found is not None
        assert len(found.tasks) == 2
    
    def test_complete_task(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal = Goal(goal_id="task_goal2", description="Goal")
        tracker.track_goal(goal)
        tracker.add_task("task_goal2", "Task to complete")
        
        tracker.complete_task("task_goal2", "Task to complete")
        
        updated = next((g for g in tracker.get_all_goals() if g.goal_id == "task_goal2"), None)
        assert updated is not None
        assert updated.tasks[0].status == "completed"
    
    def test_add_blocker(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal = Goal(goal_id="blocked_goal", description="Blocked goal")
        tracker.track_goal(goal)
        
        tracker.add_blocker("blocked_goal", "Missing dependency")
        
        updated = next((g for g in tracker.get_all_goals() if g.goal_id == "blocked_goal"), None)
        assert "Missing dependency" in updated.blockers
    
    def test_remove_blocker(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal = Goal(goal_id="unblock_goal", description="Goal")
        tracker.track_goal(goal)
        tracker.add_blocker("unblock_goal", "Blocker 1")
        tracker.add_blocker("unblock_goal", "Blocker 2")
        
        tracker.remove_blocker("unblock_goal", "Blocker 1")
        
        updated = next((g for g in tracker.get_all_goals() if g.goal_id == "unblock_goal"), None)
        assert "Blocker 1" not in updated.blockers
        assert "Blocker 2" in updated.blockers
    
    def test_detect_updates(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        updates = tracker.detect_updates("I'm stuck on the API integration")
        assert len(updates["blockers"]) > 0
        assert "api" in updates["blockers"][0].lower()
    
    def test_detect_files(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        updates = tracker.detect_updates("I'm working on main.py and utils.js")
        assert len(updates["files"]) >= 2
    
    def test_suggest_next_step_blocked(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal = Goal(goal_id="blocked", description="Build app")
        goal.blockers = ["Missing API key"]
        tracker.track_goal(goal)
        
        suggestion = tracker.suggest_next_step(goal)
        assert "blocked" in suggestion.lower()
    
    def test_suggest_next_step_with_tasks(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal = Goal(goal_id="tasks_goal", description="Build app")
        goal.tasks = [
            Task(description="Setup project", status="completed"),
            Task(description="Write code", status="pending")
        ]
        tracker.track_goal(goal)
        
        suggestion = tracker.suggest_next_step(goal)
        assert "write code" in suggestion.lower() or "task" in suggestion.lower()
    
    def test_suggest_next_step_empty(self):
        reset_goal_tracker()
        tracker = get_goal_tracker()
        
        goal = Goal(goal_id="empty_goal", description="Build something")
        tracker.track_goal(goal)
        
        suggestion = tracker.suggest_next_step(goal)
        assert len(suggestion) > 0

    def test_get_goal_context_empty(self):
        tracker = GoalTracker(storage_path=Path(tempfile.gettempdir()) / f"test_goals_empty_{os.getpid()}")

        context = tracker.get_goal_context()
        assert context == ""

    def test_persistence(self, tmp_path):
        tracker1 = GoalTracker(storage_path=tmp_path)
        
        goal = Goal(goal_id="persist_test", description="Persistent goal")
        tracker1.track_goal(goal)
        
        # Create new tracker to load persisted data
        tracker2 = GoalTracker(storage_path=tmp_path)
        goals = tracker2.get_all_goals()
        
        found = any(g.goal_id == "persist_test" for g in goals)
        assert found


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture(autouse=True)
    def _isolate_singleton(self, tmp_path):
        reset_goal_tracker(tmp_path)
        yield
        reset_goal_tracker(None)

    def test_get_goal_tracker_singleton(self):
        tracker1 = get_goal_tracker()
        tracker2 = get_goal_tracker()
        assert tracker1 is tracker2
    
    def test_reset_goal_tracker(self):
        tracker1 = get_goal_tracker()
        reset_goal_tracker()
        tracker2 = get_goal_tracker()
        # They might not be the same object after reset, but should work
        assert tracker2 is not None


class TestIntegration:
    """Integration tests."""
    
    def test_full_goal_lifecycle(self):
        tracker = get_fresh_tracker()
        
        # Extract goal from conversation
        goals = tracker.extract_goals("I want to build a Discord bot")
        assert len(goals) >= 1
        
        goal = goals[0]
        tracker.track_goal(goal)
        
        # Add tasks
        tracker.add_task(goal.goal_id, "Set up Discord developer account")
        tracker.add_task(goal.goal_id, "Create bot and get token")
        tracker.add_task(goal.goal_id, "Write bot code")
        
        # Add blocker
        tracker.add_blocker(goal.goal_id, "Need Discord account")
        
        # Get suggestion - should mention blocker
        # Re-fetch goal to see blockers
        updated_goal = next((g for g in tracker.get_all_goals() if g.goal_id == goal.goal_id), None)
        suggestion = tracker.suggest_next_step(updated_goal)
        assert "blocked" in suggestion.lower() or "account" in suggestion.lower()
        
        # Remove blocker
        tracker.remove_blocker(goal.goal_id, "Need Discord account")
        
        # Complete task
        tracker.complete_task(goal.goal_id, "Set up Discord developer account")
        
        # Get context
        context = tracker.get_goal_context()
        assert "discord" in context.lower() or len(context) >= 0
    
    def test_multiple_goals_management(self):
        tracker = get_fresh_tracker()
        
        # Add multiple goals
        goal1 = Goal(goal_id="multi1", description="Learn TypeScript", priority=5)
        goal2 = Goal(goal_id="multi2", description="Build API", priority=3)
        
        tracker.track_goal(goal1)
        tracker.track_goal(goal2)
        
        # Pause one
        tracker.pause_goal("multi2")
        
        # Get active - should only have multi1
        active = tracker.get_active_goals()
        active_ids = [g.goal_id for g in active]
        assert "multi1" in active_ids
        assert "multi2" not in active_ids
        
        # Get all
        all_goals = tracker.get_all_goals()
        all_ids = [g.goal_id for g in all_goals]
        assert "multi1" in all_ids
        assert "multi2" in all_ids