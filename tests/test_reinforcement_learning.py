"""Tests for the Reinforcement Learning module."""

from core.reinforcement_learning import (
    State,
    Action,
    ActionType,
    RewardType,
    Experience,
    ExperienceReplay,
    QLearningAgent,
    PolicyGradientAgent,
    MultiArmedBandit,
    RewardCalculator,
    ReinforcementLearningEngine,
    get_rl_engine,
)


class TestState:
    """Tests for State representation."""

    def test_state_to_vector(self):
        state = State(
            memory_usage=0.5,
            memory_hit_rate=0.8,
            response_time_avg=2.0,
            success_rate=0.9,
            active_tasks=3,
            pending_permissions=2,
            cpu_usage=0.4,
            learning_progress=0.7,
            security_score=0.85,
            evolution_generation=10,
            gene_values={"learning_rate": 0.3},
        )

        vector = state.to_vector()

        assert len(vector) == 10
        assert vector[0] == 0.5
        assert vector[1] == 0.8
        assert vector[2] == 0.2
        assert vector[3] == 0.9
        assert isinstance(vector, list)

    def test_state_hash_consistency(self):
        state1 = State(
            memory_usage=0.5,
            memory_hit_rate=0.8,
            response_time_avg=2.0,
            success_rate=0.9,
            active_tasks=3,
            pending_permissions=2,
            cpu_usage=0.4,
            learning_progress=0.7,
            security_score=0.85,
            evolution_generation=10,
            gene_values={},
        )
        state2 = State(
            memory_usage=0.5,
            memory_hit_rate=0.82,
            response_time_avg=2.5,
            success_rate=0.9,
            active_tasks=5,
            pending_permissions=1,
            cpu_usage=0.45,
            learning_progress=0.7,
            security_score=0.85,
            evolution_generation=15,
            gene_values={},
        )

        assert state1.to_hash() == state2.to_hash()

    def test_state_hash_different(self):
        state1 = State(
            memory_usage=0.25,
            memory_hit_rate=0.8,
            response_time_avg=2.0,
            success_rate=0.75,
            active_tasks=3,
            pending_permissions=2,
            cpu_usage=0.4,
            learning_progress=0.7,
            security_score=0.85,
            evolution_generation=10,
            gene_values={},
        )
        state2 = State(
            memory_usage=0.75,
            memory_hit_rate=0.8,
            response_time_avg=2.0,
            success_rate=0.5,
            active_tasks=3,
            pending_permissions=2,
            cpu_usage=0.4,
            learning_progress=0.5,
            security_score=0.5,
            evolution_generation=10,
            gene_values={},
        )

        assert state1.to_hash() != state2.to_hash()


class TestAction:
    """Tests for Action representation."""

    def test_action_to_key(self):
        action = Action(
            action_type=ActionType.GENE_ADJUSTMENT,
            target="learning_rate",
            value=0.1,
            description="Increase learning rate",
        )

        key = action.to_key()

        assert key == "gene_adjustment:learning_rate"

    def test_action_types(self):
        for action_type in ActionType:
            action = Action(
                action_type=action_type,
                target="test",
                value=0.5,
                description="Test action",
            )
            assert action.action_type == action_type


class TestExperienceReplay:
    """Tests for experience replay buffer."""

    def test_add_experience(self):
        buffer = ExperienceReplay(max_size=100)

        exp = Experience(
            state_hash="hash1",
            state_vector=[0.1, 0.2, 0.3],
            action_key="action1",
            reward=1.0,
            next_state_hash="hash2",
            next_state_vector=[0.2, 0.3, 0.4],
            done=False,
        )

        buffer.add(exp)

        assert buffer.size() == 1

    def test_sample_experiences(self):
        buffer = ExperienceReplay(max_size=100)

        for i in range(150):
            exp = Experience(
                state_hash=f"hash{i}",
                state_vector=[float(i)],
                action_key=f"action{i % 5}",
                reward=float(i),
                next_state_hash=f"hash{i+1}",
                next_state_vector=[float(i+1)],
                done=i % 20 == 0,
            )
            buffer.add(exp)

        assert buffer.size() == 100

        sample = buffer.sample(10)
        assert len(sample) == 10

    def test_sample_too_few(self):
        buffer = ExperienceReplay(max_size=100)

        for i in range(50):
            exp = Experience(
                state_hash=f"hash{i}",
                state_vector=[float(i)],
                action_key="action",
                reward=1.0,
                next_state_hash="hash_next",
                next_state_vector=[0.0],
                done=False,
            )
            buffer.add(exp)

        sample = buffer.sample(10)
        assert len(sample) == 0


class TestQLearningAgent:
    """Tests for Q-Learning agent."""

    def test_get_q_value_default(self):
        agent = QLearningAgent()

        q_val = agent.get_q_value("unknown_state", "unknown_action")

        assert q_val == 0.0

    def test_select_action_exploration(self):
        agent = QLearningAgent(epsilon=1.0)

        state = State(
            memory_usage=0.5,
            memory_hit_rate=0.5,
            response_time_avg=1.0,
            success_rate=0.5,
            active_tasks=0,
            pending_permissions=0,
            cpu_usage=0.5,
            learning_progress=0.5,
            security_score=0.5,
            evolution_generation=0,
            gene_values={},
        )

        actions = [
            Action(ActionType.GENE_ADJUSTMENT, "gene1", 0.1, "a1"),
            Action(ActionType.GENE_ADJUSTMENT, "gene2", 0.2, "a2"),
            Action(ActionType.GENE_ADJUSTMENT, "gene3", 0.3, "a3"),
        ]

        selected = agent.get_best_action(state, actions)

        assert selected in actions

    def test_update_q_value(self):
        agent = QLearningAgent(learning_rate=0.5, epsilon=0.0)

        state = State(
            memory_usage=0.5,
            memory_hit_rate=0.5,
            response_time_avg=1.0,
            success_rate=0.5,
            active_tasks=0,
            pending_permissions=0,
            cpu_usage=0.5,
            learning_progress=0.5,
            security_score=0.5,
            evolution_generation=0,
            gene_values={},
        )
        action = Action(ActionType.GENE_ADJUSTMENT, "test", 0.1, "test")
        next_state = state

        agent.update(state, action, reward=1.0, next_state=next_state, done=False)

        state_hash = state.to_hash()
        action_key = action.to_key()

        q_val = agent.get_q_value(state_hash, action_key)

        assert q_val > 0

    def test_epsilon_decay(self):
        agent = QLearningAgent(epsilon=0.5, learning_rate=0.1)

        agent._total_steps = 100
        agent._decay_epsilon()

        assert agent.epsilon < 0.5

    def test_statistics(self):
        agent = QLearningAgent()
        state = State(
            memory_usage=0.5,
            memory_hit_rate=0.5,
            response_time_avg=1.0,
            success_rate=0.5,
            active_tasks=0,
            pending_permissions=0,
            cpu_usage=0.5,
            learning_progress=0.5,
            security_score=0.5,
            evolution_generation=0,
            gene_values={},
        )
        action = Action(ActionType.GENE_ADJUSTMENT, "test", 0.1, "test")

        for _ in range(10):
            agent.update(state, action, 1.0, state, False)

        stats = agent.get_statistics()

        assert "states_known" in stats
        assert "epsilon" in stats
        assert stats["total_steps"] >= 10


class TestPolicyGradientAgent:
    """Tests for Policy Gradient agent."""

    def test_get_action_probs(self):
        agent = PolicyGradientAgent()

        state = State(
            memory_usage=0.5,
            memory_hit_rate=0.5,
            response_time_avg=1.0,
            success_rate=0.5,
            active_tasks=0,
            pending_permissions=0,
            cpu_usage=0.5,
            learning_progress=0.5,
            security_score=0.5,
            evolution_generation=0,
            gene_values={},
        )

        actions = [
            Action(ActionType.GENE_ADJUSTMENT, "gene1", 0.1, "a1"),
            Action(ActionType.GENE_ADJUSTMENT, "gene2", 0.2, "a2"),
        ]

        probs = agent.get_action_probs(state, actions)

        assert len(probs) == 2
        assert abs(sum(probs.values()) - 1.0) < 0.001

    def test_select_action(self):
        agent = PolicyGradientAgent()

        state = State(
            memory_usage=0.5,
            memory_hit_rate=0.5,
            response_time_avg=1.0,
            success_rate=0.5,
            active_tasks=0,
            pending_permissions=0,
            cpu_usage=0.5,
            learning_progress=0.5,
            security_score=0.5,
            evolution_generation=0,
            gene_values={},
        )

        actions = [
            Action(ActionType.GENE_ADJUSTMENT, "gene1", 0.1, "a1"),
            Action(ActionType.GENE_ADJUSTMENT, "gene2", 0.2, "a2"),
        ]

        selected = agent.select_action(state, actions)

        assert selected in actions

    def test_update_policy(self):
        agent = PolicyGradientAgent()

        agent.store_reward(1.0)
        agent.store_reward(0.5)
        agent.store_gradient("action1", 0.1)

        agent.update_policy()

        stats = agent.get_statistics()

        assert stats["episode_count"] == 1


class TestMultiArmedBandit:
    """Tests for Multi-Armed Bandit."""

    def test_register_arm(self):
        bandit = MultiArmedBandit()

        bandit.register_arm("strategy1", initial_value=0.5)

        assert "strategy1" in bandit.arms
        assert bandit.arms["strategy1"]["value"] == 0.5

    def test_select_arm_unknown(self):
        bandit = MultiArmedBandit()

        selected = bandit.select_arm(["strategy1", "strategy2"])

        assert selected in ["strategy1", "strategy2"]
        assert "strategy1" in bandit.arms
        assert "strategy2" in bandit.arms

    def test_update_reward(self):
        bandit = MultiArmedBandit()

        bandit.register_arm("strategy1")
        bandit.update("strategy1", 1.0)

        assert bandit.arms["strategy1"]["pulls"] == 1
        assert bandit.arms["strategy1"]["total_reward"] == 1.0
        assert bandit.arms["strategy1"]["value"] == 1.0

    def test_ucb_selection(self):
        bandit = MultiArmedBandit(exploration_const=2.0)

        bandit.register_arm("good")
        bandit.register_arm("bad")

        for _ in range(10):
            bandit.update("good", 0.9)
        for _ in range(5):
            bandit.update("bad", 0.1)

        best = bandit.get_best_arms(1)

        assert best[0][0] == "good"

    def test_statistics(self):
        bandit = MultiArmedBandit()

        bandit.register_arm("s1")
        bandit.register_arm("s2")
        bandit.update("s1", 1.0)

        stats = bandit.get_statistics()

        assert stats["total_arms"] == 2
        assert stats["total_pulls"] == 1


class TestRewardCalculator:
    """Tests for reward calculation."""

    def test_calculate_task_success(self):
        calc = RewardCalculator()

        reward = calc.calculate_reward(RewardType.TASK_SUCCESS)

        assert reward == 1.0

    def test_calculate_task_failure(self):
        calc = RewardCalculator()

        reward = calc.calculate_reward(RewardType.TASK_FAILURE)

        assert reward == -0.5

    def test_calculate_with_magnitude(self):
        calc = RewardCalculator()

        reward = calc.calculate_reward(RewardType.TASK_SUCCESS, magnitude=2.0)

        assert reward == 2.0

    def test_reward_clamping(self):
        calc = RewardCalculator()

        reward = calc.calculate_reward(RewardType.USER_POSITIVE, magnitude=10.0)

        assert reward <= 2.0

    def test_context_modifiers(self):
        calc = RewardCalculator()

        reward_normal = calc.calculate_reward(RewardType.TASK_SUCCESS)
        reward_first = calc.calculate_reward(RewardType.TASK_SUCCESS, context={"first_time": True})

        assert reward_first > reward_normal

    def test_statistics(self):
        calc = RewardCalculator()

        calc.calculate_reward(RewardType.TASK_SUCCESS)
        calc.calculate_reward(RewardType.TASK_FAILURE)
        calc.calculate_reward(RewardType.INSIGHT_GENERATED)

        stats = calc.get_statistics()

        assert stats["recent_rewards_count"] == 3


class TestReinforcementLearningEngine:
    """Tests for the RL engine."""

    def test_get_instance(self):
        engine = get_rl_engine()

        assert engine is not None
        assert isinstance(engine, ReinforcementLearningEngine)

    def test_update_state(self):
        engine = ReinforcementLearningEngine()

        state = State(
            memory_usage=0.5,
            memory_hit_rate=0.5,
            response_time_avg=1.0,
            success_rate=0.5,
            active_tasks=0,
            pending_permissions=0,
            cpu_usage=0.5,
            learning_progress=0.5,
            security_score=0.5,
            evolution_generation=0,
            gene_values={},
        )

        engine.update_state(state)

        assert engine.get_state() == state

    def test_episode_management(self):
        engine = ReinforcementLearningEngine()

        engine.start_episode()

        engine.record_reward(RewardType.TASK_SUCCESS)
        engine.record_reward(RewardType.INSIGHT_GENERATED)

        stats = engine.end_episode()

        assert stats["total_reward"] == 1.5
        assert stats["steps"] == 2

    def test_select_strategy(self):
        engine = ReinforcementLearningEngine()

        strategies = ["exploration", "exploitation", "balanced"]

        selected = engine.select_strategy(strategies)

        assert selected in strategies

    def test_update_strategy_reward(self):
        engine = ReinforcementLearningEngine()

        engine.select_strategy(["s1", "s2"])
        engine.update_strategy_reward("s1", 1.0)

        stats = engine.bandit.get_statistics()

        assert stats["total_pulls"] == 1

    def test_suggest_gene_adjustments(self):
        engine = ReinforcementLearningEngine()

        state = State(
            memory_usage=0.5,
            memory_hit_rate=0.5,
            response_time_avg=1.0,
            success_rate=0.5,
            active_tasks=0,
            pending_permissions=0,
            cpu_usage=0.5,
            learning_progress=0.5,
            security_score=0.5,
            evolution_generation=0,
            gene_values={"learning_rate": 0.3, "exploration": 0.5},
        )
        engine.update_state(state)

        for _ in range(20):
            action = Action(ActionType.GENE_ADJUSTMENT, "learning_rate", 0.05, "test")
            engine.q_agent.update(state, action, 1.0, state, False)

        adjustments = engine.suggest_gene_adjustments({"learning_rate": 0.3, "exploration": 0.5})

        assert isinstance(adjustments, dict)

    def test_statistics(self):
        engine = ReinforcementLearningEngine()

        stats = engine.get_statistics()

        assert "q_learning" in stats
        assert "policy_gradient" in stats
        assert "bandit" in stats
        assert "rewards" in stats
        assert "episodes" in stats

    def test_health_check(self):
        engine = ReinforcementLearningEngine()

        is_healthy, details = engine.is_healthy()

        assert is_healthy
        assert "episodes" in details


class TestIntegration:
    """Integration tests for RL with other components."""

    def test_rl_integration_with_engine(self):
        from core.reinforcement_learning import get_rl_engine

        engine = get_rl_engine()

        state = State(
            memory_usage=0.6,
            memory_hit_rate=0.75,
            response_time_avg=1.5,
            success_rate=0.85,
            active_tasks=2,
            pending_permissions=1,
            cpu_usage=0.35,
            learning_progress=0.6,
            security_score=0.9,
            evolution_generation=5,
            gene_values={"learning_rate": 0.3},
        )
        engine.update_state(state)

        engine.start_episode()

        actions = [
            Action(ActionType.GENE_ADJUSTMENT, "learning_rate", 0.05, "increase lr"),
            Action(ActionType.GENE_ADJUSTMENT, "learning_rate", -0.05, "decrease lr"),
            Action(ActionType.GENE_ADJUSTMENT, "exploration", 0.1, "increase exploration"),
        ]

        selected = engine.select_action(actions)
        assert selected is not None

        engine.record_reward(RewardType.TASK_SUCCESS, magnitude=1.0)

        next_state = State(
            memory_usage=0.55,
            memory_hit_rate=0.8,
            response_time_avg=1.2,
            success_rate=0.9,
            active_tasks=1,
            pending_permissions=0,
            cpu_usage=0.3,
            learning_progress=0.65,
            security_score=0.92,
            evolution_generation=5,
            gene_values={"learning_rate": 0.35},
        )

        if selected:
            engine.execute_and_learn(selected, RewardType.PROBLEM_SOLVED, next_state)

        result = engine.end_episode()

        assert result["total_reward"] > 0
        assert engine.q_agent.get_statistics()["states_known"] > 0
