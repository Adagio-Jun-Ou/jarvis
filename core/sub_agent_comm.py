"""Sub-Agent Communication System for Jarvis.

OpenClaw-style agent-to-agent messaging with session management,
message types, status broadcasting, and inter-subagent protocols.

Key features:
- Session-based communication channels
- Message types: task, status, result, query, broadcast
- Priority messaging with acknowledgment
- Message queuing and delivery guarantees
- Agent discovery and capability exchange
- Secure channel encryption (optional)

Example:
    comm = get_sub_agent_comm()
    
    # Register an agent
    comm.register_agent("researcher", capabilities=["search", "analyze"])
    
    # Send a message
    msg_id = comm.send(
        from_agent="coordinator",
        to_agent="researcher", 
        message_type=MessageType.TASK,
        payload={"task": "search for quantum computing papers"}
    )
    
    # Receive messages
    messages = comm.receive("researcher")
    for msg in messages:
        print(f"Got message: {msg.payload}")
        comm.acknowledge(msg.message_id)
"""
import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MessageType(Enum):
    TASK = "task"
    STATUS = "status"
    RESULT = "result"
    QUERY = "query"
    BROADCAST = "broadcast"
    ACK = "ack"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CAPABILITY = "capability"
    DISCOVERY = "discovery"


class MessagePriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20
    CRITICAL = 50


class MessageStatus(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


class AgentState(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"


@dataclass
class AgentInfo:
    agent_id: str
    name: str = ""
    capabilities: list[str] = field(default_factory=list)
    state: AgentState = AgentState.OFFLINE
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    max_queue_size: int = 100

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "state": self.state.value,
            "last_seen": self.last_seen.isoformat(),
            "queue_size": len(self.message_queue),
        }


@dataclass
class Message:
    message_id: str
    from_agent: str
    to_agent: str | None  # None for broadcast
    message_type: MessageType
    priority: MessagePriority
    payload: dict[str, Any]
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    acknowledged_at: datetime | None = None
    correlation_id: str | None = None  # For request-response correlation
    parent_message_id: str | None = None  # For message threading
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "correlation_id": self.correlation_id,
        }

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class Session:
    session_id: str
    participants: list[str]
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add_message(self, message: Message) -> None:
        self.message_history.append(message)
        self.last_activity = datetime.now()


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._agent_sessions: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._session_timeout_minutes = 60

    def create_session(
        self,
        participants: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        with self._lock:
            session_id = f"session_{secrets.token_hex(8)}"
            session = Session(
                session_id=session_id,
                participants=participants,
                metadata=metadata or {},
            )
            self._sessions[session_id] = session

            for agent_id in participants:
                self._agent_sessions[agent_id].add(session_id)

            logger.info(f"Created session {session_id} with participants: {participants}")
            return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_agent_sessions(self, agent_id: str) -> list[Session]:
        session_ids = self._agent_sessions.get(agent_id, set())
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    def add_participant(self, session_id: str, agent_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            if agent_id not in session.participants:
                session.participants.append(agent_id)
                self._agent_sessions[agent_id].add(session_id)

            return True

    def remove_participant(self, session_id: str, agent_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            if agent_id in session.participants:
                session.participants.remove(agent_id)
                self._agent_sessions[agent_id].discard(session_id)

            if not session.participants:
                del self._sessions[session_id]

            return True

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            for agent_id in session.participants:
                self._agent_sessions[agent_id].discard(session_id)

            del self._sessions[session_id]
            logger.info(f"Closed session {session_id}")
            return True

    def cleanup_expired(self) -> int:
        with self._lock:
            cutoff = datetime.now() - timedelta(minutes=self._session_timeout_minutes)
            expired = [
                sid for sid, session in self._sessions.items()
                if session.last_activity < cutoff
            ]
            for sid in expired:
                self.close_session(sid)
            return len(expired)

    def broadcast_to_session(
        self,
        session_id: str,
        from_agent: str,
        payload: dict[str, Any],
        message_type: MessageType = MessageType.BROADCAST,
    ) -> list[str]:
        session = self._sessions.get(session_id)
        if not session:
            return []

        message_ids = []
        for participant in session.participants:
            if participant != from_agent:
                msg_id = self._create_message_id()
                message = Message(
                    message_id=msg_id,
                    from_agent=from_agent,
                    to_agent=participant,
                    message_type=message_type,
                    priority=MessagePriority.NORMAL,
                    payload={**payload, "session_id": session_id},
                )
                session.add_message(message)
                message_ids.append(msg_id)

        return message_ids

    def _create_message_id(self) -> str:
        return f"msg_{secrets.token_hex(12)}"


class SubAgentCommunication:
    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "agent_comm"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._agents: dict[str, AgentInfo] = {}
        self._messages: dict[str, Message] = {}
        self._pending_acks: dict[str, list[str]] = defaultdict(list)
        self._handlers: dict[str, dict[MessageType, Callable]] = defaultdict(dict)
        self._message_callbacks: list[Callable] = []
        self._MAX_MESSAGE_CALLBACKS = 50

        self._sessions = SessionManager()
        self._lock = threading.RLock()

        self._message_counter = 0
        self._max_stored_messages = 50000    # More message history for learning
        self._default_ttl_seconds = 7200     # 2 hours TTL
        self._max_queue_per_agent = 500      # Larger per-agent queues

        self._discovery_handlers: list[Callable] = []
        self._MAX_DISCOVERY_HANDLERS = 20

    def register_agent(
        self,
        agent_id: str,
        name: str = "",
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentInfo:
        with self._lock:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.state = AgentState.ONLINE
                agent.last_seen = datetime.now()
                if capabilities:
                    agent.capabilities = capabilities
                if metadata:
                    agent.metadata.update(metadata)
            else:
                agent = AgentInfo(
                    agent_id=agent_id,
                    name=name or agent_id,
                    capabilities=capabilities or [],
                    state=AgentState.ONLINE,
                    metadata=metadata or {},
                )
                self._agents[agent_id] = agent

            self._broadcast_presence(agent, "joined")
            logger.info(f"Registered agent: {agent_id} with capabilities: {capabilities or []}")
            return agent

    def unregister_agent(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                return False

            agent = self._agents[agent_id]
            agent.state = AgentState.OFFLINE
            self._broadcast_presence(agent, "left")

            del self._agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
            return True

    def update_agent_state(self, agent_id: str, state: AgentState) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                return False

            agent = self._agents[agent_id]
            agent.state = state
            agent.last_seen = datetime.now()

            if state == AgentState.BUSY:
                self._broadcast_status(agent, "busy")
            elif state == AgentState.IDLE:
                self._broadcast_status(agent, "idle")

            return True

    def heartbeat(self, agent_id: str, state: AgentState = AgentState.IDLE) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                return False

            agent = self._agents[agent_id]
            agent.last_seen = datetime.now()
            if agent.state != state:
                agent.state = state
            return True

    def send(
        self,
        from_agent: str,
        to_agent: str | None,
        message_type: MessageType,
        payload: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_seconds: int | None = None,
        correlation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> str:
        with self._lock:
            message_id = self._create_message_id()

            expires_at = None
            if ttl_seconds is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            elif self._default_ttl_seconds > 0:
                expires_at = datetime.now() + timedelta(seconds=self._default_ttl_seconds)

            message = Message(
                message_id=message_id,
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=message_type,
                priority=priority,
                payload=payload,
                expires_at=expires_at,
                correlation_id=correlation_id,
                parent_message_id=parent_message_id,
            )

            self._store_message(message)

            if to_agent is None:
                self._deliver_broadcast(message)
            else:
                self._deliver_message(message)

            return message_id

    def broadcast(
        self,
        from_agent: str,
        message_type: MessageType,
        payload: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        exclude_agents: list[str] | None = None,
    ) -> list[str]:
        with self._lock:
            exclude = set(exclude_agents or [])
            exclude.add(from_agent)

            message_ids = []
            for agent_id, agent in self._agents.items():
                if agent_id in exclude:
                    continue
                if agent.state == AgentState.OFFLINE:
                    continue

                msg_id = self.send(
                    from_agent=from_agent,
                    to_agent=agent_id,
                    message_type=message_type,
                    payload=payload,
                    priority=priority,
                )
                message_ids.append(msg_id)

            return message_ids

    def receive(
        self,
        agent_id: str,
        message_types: list[MessageType] | None = None,
        limit: int = 10,
    ) -> list[Message]:
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return []

            messages = []
            filtered = []

            for msg in agent.message_queue:
                if message_types and msg.message_type not in message_types:
                    continue
                if msg.is_expired():
                    msg.status = MessageStatus.EXPIRED
                    continue

                filtered.append(msg)

            filtered.sort(key=lambda m: m.priority.value, reverse=True)

            for msg in filtered[:limit]:
                msg.status = MessageStatus.DELIVERED
                messages.append(msg)

            agent.message_queue = deque((m for m in agent.message_queue if m not in messages), maxlen=agent.max_queue_size)

            return messages

    def acknowledge(self, message_id: str, from_agent: str) -> bool:
        with self._lock:
            message = self._messages.get(message_id)
            if not message:
                return False

            if message.to_agent != from_agent:
                return False

            message.status = MessageStatus.ACKNOWLEDGED
            message.acknowledged_at = datetime.now()

            if message.from_agent in self._agents:
                ack_msg = Message(
                    message_id=self._create_message_id(),
                    from_agent=from_agent,
                    to_agent=message.from_agent,
                    message_type=MessageType.ACK,
                    priority=MessagePriority.HIGH,
                    payload={"ack_for": message_id},
                    correlation_id=message.correlation_id,
                    parent_message_id=message_id,
                )
                self._deliver_message(ack_msg)

            return True

    def respond(
        self,
        to_message_id: str,
        from_agent: str,
        payload: dict[str, Any],
        message_type: MessageType = MessageType.RESULT,
    ) -> str | None:
        with self._lock:
            original = self._messages.get(to_message_id)
            if not original:
                return None

            return self.send(
                from_agent=from_agent,
                to_agent=original.from_agent,
                message_type=message_type,
                payload=payload,
                correlation_id=original.correlation_id,
                parent_message_id=to_message_id,
            )

    def query(
        self,
        from_agent: str,
        to_agent: str,
        payload: dict[str, Any],
        timeout_seconds: float = 30.0,
    ) -> Message | None:
        correlation_id = f"query_{secrets.token_hex(8)}"

        self.send(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.QUERY,
            payload=payload,
            priority=MessagePriority.HIGH,
            correlation_id=correlation_id,
        )

        start = time.time()
        while time.time() - start < timeout_seconds:
            responses = self.receive(
                from_agent,
                message_types=[MessageType.RESULT],
            )
            for resp in responses:
                if resp.correlation_id == correlation_id:
                    return resp
            time.sleep(0.1)

        return None

    def discover_agents(
        self,
        capabilities: list[str] | None = None,
        state: AgentState | None = None,
        exclude: list[str] | None = None,
    ) -> list[AgentInfo]:
        with self._lock:
            results = []
            for agent in self._agents.values():
                if exclude and agent.agent_id in exclude:
                    continue
                if state and agent.state != state:
                    continue
                if capabilities:
                    if not all(c in agent.capabilities for c in capabilities):
                        continue
                results.append(agent)
            return results

    def register_handler(
        self,
        agent_id: str,
        message_type: MessageType,
        handler: Callable[[Message], None],
    ) -> None:
        with self._lock:
            self._handlers[agent_id][message_type] = handler

    def unregister_handler(
        self,
        agent_id: str,
        message_type: MessageType,
    ) -> bool:
        """Unregister a message handler for an agent."""
        with self._lock:
            if agent_id in self._handlers and message_type in self._handlers[agent_id]:
                del self._handlers[agent_id][message_type]
                if not self._handlers[agent_id]:
                    del self._handlers[agent_id]
                return True
            return False

    def register_message_callback(self, callback: Callable[[Message], None]) -> None:
        with self._lock:
            if len(self._message_callbacks) < self._MAX_MESSAGE_CALLBACKS:
                self._message_callbacks.append(callback)

    def unregister_message_callback(self, callback: Callable[[Message], None]) -> bool:
        """Unregister a message callback."""
        with self._lock:
            if callback in self._message_callbacks:
                self._message_callbacks.remove(callback)
                return True
            return False

    def register_discovery_handler(self, handler: Callable[[AgentInfo, str], None]) -> None:
        with self._lock:
            if len(self._discovery_handlers) < self._MAX_DISCOVERY_HANDLERS:
                self._discovery_handlers.append(handler)

    def unregister_discovery_handler(self, handler: Callable[[AgentInfo, str], None]) -> bool:
        """Unregister a discovery handler."""
        with self._lock:
            if handler in self._discovery_handlers:
                self._discovery_handlers.remove(handler)
                return True
            return False

    def get_agent_info(self, agent_id: str) -> AgentInfo | None:
        return self._agents.get(agent_id)

    def get_message(self, message_id: str) -> Message | None:
        return self._messages.get(message_id)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            states = defaultdict(int)
            for agent in self._agents.values():
                states[agent.state.value] += 1

            return {
                "total_agents": len(self._agents),
                "agents_by_state": dict(states),
                "total_messages": len(self._messages),
                "pending_acks": sum(len(v) for v in self._pending_acks.values()),
                "active_sessions": len(self._sessions._sessions),
            }

    def create_session(
        self,
        participants: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        return self._sessions.create_session(participants, metadata)

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get_session(session_id)

    def session_send(
        self,
        session_id: str,
        from_agent: str,
        payload: dict[str, Any],
    ) -> list[str]:
        session = self._sessions.get_session(session_id)
        if not session:
            return []

        message_ids = []
        for participant in session.participants:
            if participant != from_agent:
                msg_id = self.send(
                    from_agent=from_agent,
                    to_agent=participant,
                    message_type=MessageType.BROADCAST,
                    payload={**payload, "session_id": session_id},
                )
                message_ids.append(msg_id)

        return message_ids

    def cleanup_stale(self, timeout_seconds: int = 300) -> int:
        with self._lock:
            cutoff = datetime.now() - timedelta(seconds=timeout_seconds)
            stale = [
                aid for aid, agent in self._agents.items()
                if agent.last_seen < cutoff and agent.state != AgentState.OFFLINE
            ]
            for agent_id in stale:
                self._agents[agent_id].state = AgentState.OFFLINE
                self._broadcast_presence(self._agents[agent_id], "timeout")

            expired_msgs = [
                mid for mid, msg in self._messages.items()
                if msg.is_expired()
            ]
            for mid in expired_msgs:
                del self._messages[mid]

            return len(stale)

    def _create_message_id(self) -> str:
        self._message_counter += 1
        return f"msg_{self._message_counter:08d}_{secrets.token_hex(4)}"

    def _store_message(self, message: Message) -> None:
        self._messages[message.message_id] = message

        while len(self._messages) > self._max_stored_messages:
            oldest = min(self._messages.keys())
            del self._messages[oldest]

    def _deliver_message(self, message: Message) -> None:
        if message.to_agent and message.to_agent in self._agents:
            agent = self._agents[message.to_agent]

            if len(agent.message_queue) >= agent.max_queue_size:
                lowest = min(agent.message_queue, key=lambda m: m.priority.value)
                agent.message_queue.remove(lowest)

            agent.message_queue.append(message)
            message.status = MessageStatus.DELIVERED

            handler = self._handlers.get(message.to_agent, {}).get(message.message_type)
            if handler:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Handler error for {message.message_id}: {e}")

            with self._lock:
                callbacks_snapshot = list(self._message_callbacks)
            for callback in callbacks_snapshot:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    def _deliver_broadcast(self, message: Message) -> None:
        for agent_id, agent in self._agents.items():
            if agent_id == message.from_agent:
                continue
            if agent.state == AgentState.OFFLINE:
                continue

            if len(agent.message_queue) < agent.max_queue_size:
                agent.message_queue.append(message)

    def _broadcast_presence(self, agent: AgentInfo, action: str) -> None:
        with self._lock:
            handlers_snapshot = list(self._discovery_handlers)
        for handler in handlers_snapshot:
            try:
                handler(agent, action)
            except Exception as e:
                logger.error(f"Discovery handler error: {e}")

        self.broadcast(
            from_agent=agent.agent_id,
            message_type=MessageType.DISCOVERY,
            payload={"action": action, "agent": agent.to_dict()},
        )

    def _broadcast_status(self, agent: AgentInfo, status: str) -> None:
        self.broadcast(
            from_agent=agent.agent_id,
            message_type=MessageType.STATUS,
            payload={"status": status, "agent_id": agent.agent_id},
            exclude_agents=[agent.agent_id],
        )


_sub_agent_comm: SubAgentCommunication | None = None
_comm_lock = threading.Lock()


def get_sub_agent_comm() -> SubAgentCommunication:
    global _sub_agent_comm
    if _sub_agent_comm is None:
        with _comm_lock:
            if _sub_agent_comm is None:
                _sub_agent_comm = SubAgentCommunication()
    return _sub_agent_comm


def sessions_create(participants: list[str], metadata: dict | None = None) -> Session:
    return get_sub_agent_comm().create_session(participants, metadata)


def sessions_list(agent_id: str) -> list[Session]:
    return get_sub_agent_comm()._sessions.get_agent_sessions(agent_id)


def sessions_get(session_id: str) -> Session | None:
    return get_sub_agent_comm().get_session(session_id)


def sessions_send(session_id: str, from_agent: str, payload: dict) -> list[str]:
    return get_sub_agent_comm().session_send(session_id, from_agent, payload)


def sessions_close(session_id: str) -> bool:
    return get_sub_agent_comm()._sessions.close_session(session_id)


def agents_register(agent_id: str, capabilities: list[str] | None = None) -> AgentInfo:
    return get_sub_agent_comm().register_agent(agent_id, capabilities=capabilities)


def agents_list(capabilities: list[str] | None = None) -> list[AgentInfo]:
    return get_sub_agent_comm().discover_agents(capabilities=capabilities)


def agents_get(agent_id: str) -> AgentInfo | None:
    return get_sub_agent_comm().get_agent_info(agent_id)


def send_message(
    from_agent: str,
    to_agent: str,
    message_type: str,
    payload: dict,
) -> str:
    return get_sub_agent_comm().send(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=MessageType(message_type),
        payload=payload,
    )
