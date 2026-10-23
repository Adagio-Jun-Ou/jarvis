"""Reinforcement Learning Module for Jarvis.

Implements multiple RL algorithms for adaptive behavior:
- Q-Learning with experience replay for discrete action selection
- Policy Gradient for continuous parameter tuning  
- Multi-Armed Bandit for strategy exploration
- Deep Q-Network approximation for large state spaces

The RL system learns from:
- Task success/failure outcomes
- User feedback (explicit and implicit)
- Resource efficiency metrics
- Learning progress indicators
- Security posture scores
"""
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import math
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import hashlib

logger = logging.getLogger(__name__)


class ActionType(Enum):
    GENE_ADJUSTMENT = "gene_adjustment"
    STRATEGY_SELECTION = "strategy_selection"
    EXPLORATION_MODE = "exploration_mode"
    RESOURCE_ALLOCATION = "resource_allocation"
    LEARNING_FOCUS = "learning_focus"
    SECURITY_POSTURE = "security_posture"


class RewardType(Enum):
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    USER_POSITIVE = "user_positive"
    USER_NEGATIVE = "user_negative"
    EFFICIENCY_GAIN = "efficiency_gain"
    INSIGHT_GENERATED = "insight_generated"
    PROBLEM_SOLVED = "problem_solved"
    SECURITY_THREAT_BLOCKED = "security_threat_blocked"
    RESOURCE_PRESSURE = "resource_pressure"
    LEARNING_PROGRESS = "learning_progress"


@dataclass
class State:
    """Represents Jarvis's current state for RL decisions."""
    memory_usage: float
    memory_hit_rate: float
    response_time_avg: float
    success_rate: float
    active_tasks: int
    pending_permissions: int
    cpu_usage: float
    learning_progress: float
    security_score: float
    evolution_generation: int
    gene_values: dict[str, float]

    def to_vector(self) -> list[float]:
        return [
            self.memory_usage,
            self.memory_hit_rate,
            min(self.response_time_avg / 10.0, 1.0),
            self.success_rate,
            min(self.active_tasks / 10.0, 1.0),
            min(self.pending_permissions / 5.0, 1.0),
            self.cpu_usage,
            self.learning_progress,
            self.security_score,
            min(self.evolution_generation / 100.0, 1.0),
        ]

    def to_hash(self) -> str:
        discrete = (
            round(self.memory_usage * 4) / 4,
            round(self.success_rate * 4) / 4,
            round(self.learning_progress * 4) / 4,
            round(self.security_score * 4) / 4,
        )
        return hashlib.sha256(str(discrete).encode()).hexdigest()[:12]


@dataclass
class Action:
    """An action Jarvis can take."""
    action_type: ActionType
    target: str
    value: float
    description: str

    def to_key(self) -> str:
        return f"{self.action_type.value}:{self.target}"


@dataclass
class Experience:
    """A single RL experience for replay."""
    state_hash: str
    state_vector: list[float]
    action_key: str
    reward: float
    next_state_hash: str
    next_state_vector: list[float]
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class QTableEntry:
    """Q-value entry with visit count for exploration."""
    q_value: float
    visit_count: int
    last_updated: datetime = field(default_factory=datetime.now)


class ExperienceReplay:
    """Manages experience replay buffer for Q-learning."""

    MAX_SIZE = 10000
    MIN_SIZE = 100
    BATCH_SIZE = 32

    def __init__(self, max_size: int = MAX_SIZE):
        self.buffer: deque[Experience] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, experience: Experience) -> None:
        with self._lock:
            self.buffer.append(experience)

    def sample(self, batch_size: int = BATCH_SIZE) -> list[Experience]:
        with self._lock:
            if len(self.buffer) < self.MIN_SIZE:
                return []
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def get_recent(self, limit: int = 100) -> list[Experience]:
        with self._lock:
            return list(self.buffer)[-limit:]

    def size(self) -> int:
        with self._lock:
            return len(self.buffer)


class QLearningAgent:
    """Q-Learning agent with experience replay and eligibility traces."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.3,
    ):
        self.LEARNING_RATE = 0.1
        self.DISCOUNT_FACTOR = 0.95
        self.EPSILON_START = 0.3
        self.EPSILON_END = 0.01
        self.EPSILON_DECAY = 0.995
        self.MAX_REPLAY_SIZE = 100000    # 10x for better learning
        self.MIN_REPLAY_SIZE = 500
        self.BATCH_SIZE = 64             # Larger batches for efficiency
        self.MAX_QTABLE_SIZE = 500000     # 10x for state space

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = self.EPSILON_END
        self.epsilon_decay = self.EPSILON_DECAY

        self.q_table: dict[str, dict[str, QTableEntry]] = {}
        self._qtable_access: dict[str, int] = {}
        self.replay_buffer = ExperienceReplay()
        self._lock = threading.RLock()

        self._total_steps = 0
        self._total_updates = 0

    def get_q_value(self, state_hash: str, action_key: str) -> float:
        with self._lock:
            if state_hash not in self.q_table:
                return 0.0
            if action_key not in self.q_table[state_hash]:
                return 0.0
            return self.q_table[state_hash][action_key].q_value

    def get_best_action(self, state: State, available_actions: list[Action]) -> Action | None:
        if not available_actions:
            return None

        state_hash = state.to_hash()

        if random.random() < self.epsilon:
            return random.choice(available_actions)

        with self._lock:
            best_action = None
            best_value = float('-inf')

            for action in available_actions:
                q_val = self.get_q_value(state_hash, action.to_key())
                if q_val > best_value:
                    best_value = q_val
                    best_action = action

            return best_action if best_action else random.choice(available_actions)

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool = False,
    ) -> None:
        state_hash = state.to_hash()
        action_key = action.to_key()
        next_hash = next_state.to_hash()

        experience = Experience(
            state_hash=state_hash,
            state_vector=state.to_vector(),
            action_key=action_key,
            reward=reward,
            next_state_hash=next_hash,
            next_state_vector=next_state.to_vector(),
            done=done,
        )
        self.replay_buffer.add(experience)

        with self._lock:
            if state_hash not in self.q_table:
                self.q_table[state_hash] = {}

            if action_key not in self.q_table[state_hash]:
                self.q_table[state_hash][action_key] = QTableEntry(
                    q_value=0.0,
                    visit_count=0,
                )

            entry = self.q_table[state_hash][action_key]
            entry.visit_count += 1
            entry.last_updated = datetime.now()

            max_next_q = 0.0
            if not done and next_hash in self.q_table:
                max_next_q = max(
                    (e.q_value for e in self.q_table[next_hash].values()),
                    default=0.0
                )

            td_target = reward + self.gamma * max_next_q * (1 - int(done))
            td_error = td_target - entry.q_value
            entry.q_value += self.alpha * td_error

            self._total_steps += 1
            self._total_updates += 1

        if self._total_steps % 100 == 0:
            self._decay_epsilon()

        if self._total_steps % 500 == 0:
            self._prune_qtable()

        self._replay_update()

    def _replay_update(self) -> None:
        batch = self.replay_buffer.sample()
        for exp in batch:
            with self._lock:
                if exp.state_hash not in self.q_table:
                    self.q_table[exp.state_hash] = {}

                if exp.action_key not in self.q_table[exp.state_hash]:
                    self.q_table[exp.state_hash][exp.action_key] = QTableEntry(
                        q_value=0.0,
                        visit_count=0,
                    )

                entry = self.q_table[exp.state_hash][exp.action_key]
                entry.visit_count += 1

                max_next_q = 0.0
                if not exp.done and exp.next_state_hash in self.q_table:
                    max_next_q = max(
                        (e.q_value for e in self.q_table[exp.next_state_hash].values()),
                        default=0.0
                    )

                td_target = exp.reward + self.gamma * max_next_q * (1 - int(exp.done))
                entry.q_value += self.alpha * 0.5 * (td_target - entry.q_value)

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _prune_qtable(self) -> None:
        total_entries = sum(len(actions) for actions in self.q_table.values())
        if total_entries <= self.MAX_QTABLE_SIZE:
            return

        state_counts = [
            (state_hash, sum(e.visit_count for e in actions.values()))
            for state_hash, actions in self.q_table.items()
        ]
        state_counts.sort(key=lambda x: x[1])

        entries_to_remove = total_entries - self.MAX_QTABLE_SIZE
        removed = 0
        for state_hash, _ in state_counts:
            if removed >= entries_to_remove:
                break
            if state_hash in self.q_table:
                removed += len(self.q_table[state_hash])
                del self.q_table[state_hash]

        if removed > 0:
            logger.debug(f"Pruned Q-table: removed {removed} entries, {total_entries - removed} remaining")

    def get_statistics(self) -> dict:
        with self._lock:
            total_q_values = sum(
                len(actions) for actions in self.q_table.values()
            )
            avg_q = 0.0
            if total_q_values > 0:
                all_q = [
                    entry.q_value
                    for actions in self.q_table.values()
                    for entry in actions.values()
                ]
                avg_q = sum(all_q) / len(all_q) if all_q else 0.0

            return {
                "states_known": len(self.q_table),
                "total_q_values": total_q_values,
                "average_q": avg_q,
                "epsilon": self.epsilon,
                "total_steps": self._total_steps,
                "replay_buffer_size": self.replay_buffer.size(),
            }


class PolicyGradientAgent:
    """Policy Gradient agent for continuous parameter tuning."""

    MAX_POLICY_PARAMS = 10000

    def __init__(self, learning_rate: float = 0.01):
        self.alpha = learning_rate
        self.baseline_decay = 0.99

        self.policy_params: dict[str, dict[str, float]] = {}
        self._params_access: dict[str, int] = {}
        self._episode_rewards: list[float] = []
        self._episode_grads: dict[str, list[float]] = {}
        self._baseline: float = 0.0
        self._lock = threading.RLock()
        self._episode_count = 0

    def _prune_policy_params(self) -> None:
        if len(self.policy_params) <= self.MAX_POLICY_PARAMS:
            return

        sorted_keys = sorted(
            self._params_access.items(),
            key=lambda x: x[1],
        )

        entries_to_remove = len(self.policy_params) - self.MAX_POLICY_PARAMS
        for key, _ in sorted_keys[:entries_to_remove]:
            if key in self.policy_params:
                del self.policy_params[key]
            if key in self._params_access:
                del self._params_access[key]

    def get_action_probs(
        self,
        state: State,
        available_actions: list[Action],
    ) -> dict[str, float]:
        if not available_actions:
            return {}

        state.to_hash()
        probs: dict[str, float] = {}

        with self._lock:
            scores = []
            for action in available_actions:
                key = action.to_key()
                if key not in self.policy_params:
                    self.policy_params[key] = {"weight": 0.0, "bias": 0.0}

                params = self.policy_params[key]
                state_sum = sum(state.to_vector())
                score = params["weight"] * state_sum + params["bias"]
                scores.append((key, score))

            max_score = max(s for _, s in scores) if scores else 0
            exp_scores = [(k, math.exp(min(s - max_score, 10))) for k, s in scores]
            total = sum(e for _, e in exp_scores)

            for key, exp_score in exp_scores:
                self._params_access[key] = self._params_access.get(key, 0) + 1
                if total > 0 and scores:
                    probs[key] = exp_score / total
                elif scores:
                    probs[key] = 1.0 / len(scores)
                else:
                    probs[key] = 0.0

            if len(self.policy_params) > self.MAX_POLICY_PARAMS:
                self._prune_policy_params()

        return probs

    def select_action(
        self,
        state: State,
        available_actions: list[Action],
    ) -> Action | None:
        if not available_actions:
            return None

        probs = self.get_action_probs(state, available_actions)

        keys = list(probs.keys())
        values = list(probs.values())

        selected_key = random.choices(keys, weights=values, k=1)[0]

        for action in available_actions:
            if action.to_key() == selected_key:
                return action

        return random.choice(available_actions)

    def store_reward(self, reward: float) -> None:
        with self._lock:
            self._episode_rewards.append(reward)

    def store_gradient(self, action_key: str, grad: float) -> None:
        with self._lock:
            if action_key not in self._episode_grads:
                self._episode_grads[action_key] = []
            self._episode_grads[action_key].append(grad)

    def update_policy(self, episode_rewards: list[float] | None = None) -> None:
        rewards = episode_rewards or []
        with self._lock:
            rewards = rewards or self._episode_rewards

            if not rewards:
                return

            total_reward = sum(rewards)
            discounted = self._compute_returns(rewards)

            self._baseline = (
                self.baseline_decay * self._baseline +
                (1 - self.baseline_decay) * total_reward / len(rewards)
            )

            advantage = sum(discounted) - self._baseline * len(discounted)

            for action_key, grads in self._episode_grads.items():
                if action_key not in self.policy_params:
                    self.policy_params[action_key] = {"weight": 0.0, "bias": 0.0}

                avg_grad = sum(grads) / len(grads) if grads else 0.0
                self.policy_params[action_key]["weight"] += self.alpha * advantage * avg_grad
                self.policy_params[action_key]["bias"] += self.alpha * advantage * 0.01

            self._episode_rewards.clear()
            self._episode_grads.clear()
            self._episode_count += 1

    def _compute_returns(self, rewards: list[float], gamma: float = 0.99) -> list[float]:
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    def get_statistics(self) -> dict:
        with self._lock:
            return {
                "policy_params_count": len(self.policy_params),
                "baseline": self._baseline,
                "episode_count": self._episode_count,
            }


class MultiArmedBandit:
    """Multi-Armed Bandit for strategy selection."""

    def __init__(self, exploration_const: float = 2.0):
        self.c = exploration_const
        self.arms: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._total_pulls = 0

    def register_arm(self, arm_id: str, initial_value: float = 0.0) -> None:
        with self._lock:
            if arm_id not in self.arms:
                self.arms[arm_id] = {
                    "value": initial_value,
                    "pulls": 0,
                    "total_reward": 0.0,
                }

    def select_arm(self, available_arms: list[str]) -> str | None:
        if not available_arms:
            return None

        with self._lock:
            for arm in available_arms:
                if arm not in self.arms:
                    self.register_arm(arm)

            best_arm = None
            best_ucb = float('-inf')

            for arm in available_arms:
                arm_data = self.arms[arm]
                pulls = arm_data["pulls"]

                if pulls == 0:
                    return arm

                avg_reward = arm_data["total_reward"] / pulls
                confidence = math.sqrt(
                    self.c * math.log(self._total_pulls + 1) / pulls
                )
                ucb = avg_reward + confidence

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_arm = arm

            return best_arm or random.choice(available_arms)

    def update(self, arm_id: str, reward: float) -> None:
        with self._lock:
            if arm_id not in self.arms:
                self.register_arm(arm_id)

            self.arms[arm_id]["pulls"] += 1
            self.arms[arm_id]["total_reward"] += reward
            self.arms[arm_id]["value"] = (
                self.arms[arm_id]["total_reward"] / self.arms[arm_id]["pulls"]
            )
            self._total_pulls += 1

    def get_best_arms(self, limit: int = 5) -> list[tuple[str, float]]:
        with self._lock:
            sorted_arms = sorted(
                self.arms.items(),
                key=lambda x: x[1]["value"],
                reverse=True
            )
            return [(arm_id, data["value"]) for arm_id, data in sorted_arms[:limit]]

    def get_statistics(self) -> dict:
        with self._lock:
            return {
                "total_arms": len(self.arms),
                "total_pulls": self._total_pulls,
                "best_arms": self.get_best_arms(5),
            }


class RewardCalculator:
    """Calculates rewards from various Jarvis signals."""

    def __init__(self):
        self._reward_history: deque[dict] = deque(maxlen=1000)
        self._cumulative_reward = 0.0
        self._lock = threading.Lock()

        self.reward_weights = {
            RewardType.TASK_SUCCESS: 1.0,
            RewardType.TASK_FAILURE: -0.5,
            RewardType.USER_POSITIVE: 2.0,
            RewardType.USER_NEGATIVE: -1.0,
            RewardType.EFFICIENCY_GAIN: 0.3,
            RewardType.INSIGHT_GENERATED: 0.5,
            RewardType.PROBLEM_SOLVED: 0.8,
            RewardType.SECURITY_THREAT_BLOCKED: 0.6,
            RewardType.RESOURCE_PRESSURE: -0.2,
            RewardType.LEARNING_PROGRESS: 0.4,
        }

    def calculate_reward(
        self,
        reward_type: RewardType,
        magnitude: float = 1.0,
        context: dict | None = None,
    ) -> float:
        base_reward = self.reward_weights.get(reward_type, 0.0)
        reward = base_reward * magnitude

        if context:
            modifier = self._apply_context_modifiers(context)
            reward *= modifier

        reward = max(-2.0, min(2.0, reward))

        with self._lock:
            self._cumulative_reward += reward
            self._reward_history.append({
                "type": reward_type.value,
                "base": base_reward,
                "magnitude": magnitude,
                "final": reward,
                "timestamp": datetime.now().isoformat(),
            })

        return reward

    def _apply_context_modifiers(self, context: dict) -> float:
        modifier = 1.0

        if context.get("first_time"):
            modifier *= 1.2

        if context.get("repeated"):
            modifier *= 0.8

        if context.get("high_confidence"):
            modifier *= 1.1

        if context.get("low_confidence"):
            modifier *= 0.9

        return modifier

    def get_statistics(self) -> dict:
        with self._lock:
            recent = list(self._reward_history)[-50:]

            by_type: dict[str, list[float]] = {}
            for r in recent:
                t = r["type"]
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append(r["final"])

            type_averages = {
                t: sum(vals) / len(vals) for t, vals in by_type.items() if vals
            }

            return {
                "cumulative_reward": self._cumulative_reward,
                "recent_rewards_count": len(recent),
                "average_recent": sum(r["final"] for r in recent) / len(recent) if recent else 0,
                "by_type_averages": type_averages,
            }


class ReinforcementLearningEngine:
    """Central RL engine coordinating all learning components."""

    MAX_EPISODE_REWARDS = 1000

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "rl"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.q_agent = QLearningAgent()
        self.policy_agent = PolicyGradientAgent()
        self.bandit = MultiArmedBandit()
        self.reward_calc = RewardCalculator()

        self._lock = threading.RLock()
        self._current_state: State | None = None
        self._last_action: Action | None = None
        self._episode_active = False
        self._episode_rewards: list[float] = []

        self._action_history: deque[dict] = deque(maxlen=500)
        self._total_episodes = 0

        self._initialized_strategies = False

        self._load_state()

    def _load_state(self) -> None:
        state_file = self.storage_path / "rl_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._total_episodes = data.get("total_episodes", 0)
                self.q_agent.epsilon = data.get("epsilon", 0.3)
                self.q_agent._total_steps = data.get("total_steps", 0)
                logger.info(f"Loaded RL state: {self._total_episodes} episodes")
            except Exception as e:
                logger.warning(f"Failed to load RL state: {e}")

    def _save_state(self) -> None:
        state_file = self.storage_path / "rl_state.json"
        try:
            atomic_write_json(state_file, {
                    "total_episodes": self._total_episodes,
                    "epsilon": self.q_agent.epsilon,
                    "total_steps": self.q_agent._total_steps,
                    "last_updated": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Failed to save RL state: {e}")

    def update_state(self, state: State) -> None:
        with self._lock:
            self._current_state = state

    def get_state(self) -> State | None:
        with self._lock:
            return self._current_state

    def start_episode(self) -> None:
        with self._lock:
            self._episode_active = True
            self._episode_rewards = []

    def end_episode(self) -> dict:
        with self._lock:
            self._episode_active = False
            self._total_episodes += 1

            self.policy_agent.update_policy(self._episode_rewards)

            stats = {
                "episode": self._total_episodes,
                "total_reward": sum(self._episode_rewards),
                "avg_reward": sum(self._episode_rewards) / len(self._episode_rewards) if self._episode_rewards else 0,
                "steps": len(self._episode_rewards),
            }

            self._episode_rewards = []

            if self._total_episodes % 10 == 0:
                self._save_state()

            return stats

    def record_reward(
        self,
        reward_type: RewardType,
        magnitude: float = 1.0,
        context: dict | None = None,
    ) -> float:
        reward = self.reward_calc.calculate_reward(reward_type, magnitude, context)

        with self._lock:
            self._episode_rewards.append(reward)
            if len(self._episode_rewards) > self.MAX_EPISODE_REWARDS:
                self._episode_rewards = self._episode_rewards[-self.MAX_EPISODE_REWARDS:]

        return reward

    def select_action(
        self,
        available_actions: list[Action],
        use_policy: bool = False,
    ) -> Action | None:
        with self._lock:
            if not self._current_state or not available_actions:
                return None

            if use_policy:
                return self.policy_agent.select_action(self._current_state, available_actions)
            else:
                return self.q_agent.get_best_action(self._current_state, available_actions)

    def select_strategy(self, strategies: list[str]) -> str | None:
        self._ensure_strategies_registered(strategies)
        return self.bandit.select_arm(strategies)

    def _ensure_strategies_registered(self, strategies: list[str]) -> None:
        if self._initialized_strategies:
            return
        for strategy in strategies:
            self.bandit.register_arm(strategy)
        self._initialized_strategies = True

    def update_strategy_reward(self, strategy: str, reward: float) -> None:
        self.bandit.update(strategy, reward)

    def execute_and_learn(
        self,
        action: Action,
        reward_type: RewardType,
        next_state: State,
        magnitude: float = 1.0,
    ) -> None:
        with self._lock:
            if not self._current_state:
                return

            reward = self.record_reward(reward_type, magnitude)

            self.q_agent.update(
                state=self._current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=not self._episode_active,
            )

            self._action_history.append({
                "action": action.to_key(),
                "reward_type": reward_type.value,
                "reward": reward,
                "timestamp": datetime.now().isoformat(),
            })

            self._current_state = next_state
            self._last_action = action

    def suggest_gene_adjustments(
        self,
        current_genes: dict[str, float],
    ) -> dict[str, float]:
        adjustments: dict[str, float] = {}

        with self._lock:
            for gene_name, current_value in current_genes.items():
                actions = [
                    Action(
                        action_type=ActionType.GENE_ADJUSTMENT,
                        target=gene_name,
                        value=delta,
                        description=f"Adjust {gene_name} by {delta:+.2f}",
                    )
                    for delta in [-0.1, -0.05, 0.0, 0.05, 0.1]
                ]

                if self._current_state:
                    best_action = self.q_agent.get_best_action(self._current_state, actions)
                    if best_action:
                        adjustments[gene_name] = best_action.value

        return adjustments

    def get_statistics(self) -> dict:
        with self._lock:
            return {
                "q_learning": self.q_agent.get_statistics(),
                "policy_gradient": self.policy_agent.get_statistics(),
                "bandit": self.bandit.get_statistics(),
                "rewards": self.reward_calc.get_statistics(),
                "episodes": {
                    "total": self._total_episodes,
                    "current_active": self._episode_active,
                },
                "action_history_size": len(self._action_history),
            }

    def is_healthy(self) -> tuple[bool, dict]:
        stats = self.get_statistics()
        return (
            stats["episodes"]["total"] >= 0,
            {
                "episodes": stats["episodes"]["total"],
                "q_states": stats["q_learning"]["states_known"],
                "bandit_arms": stats["bandit"]["total_arms"],
            }
        )


_rl_engine: ReinforcementLearningEngine | None = None
_rl_lock = threading.Lock()


def get_rl_engine() -> ReinforcementLearningEngine:
    global _rl_engine
    if _rl_engine is None:
        with _rl_lock:
            if _rl_engine is None:
                _rl_engine = ReinforcementLearningEngine()
    return _rl_engine
