"""Base agent framework for multi-agent orchestration.

Defines the foundational message protocol, agent lifecycle states, and the
abstract :class:`BaseAgent` class from which every specialist agent inherits.
All inter-agent communication is routed through :class:`AgentMessage` objects
placed on per-agent asyncio queues.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message protocol
# ---------------------------------------------------------------------------


@dataclass
class AgentMessage:
    """Immutable message exchanged between agents.

    Attributes
    ----------
    sender : str
        Name of the sending agent.
    recipient : str
        Name of the intended receiving agent.
    content : dict
        Arbitrary payload; schema depends on *message_type*.
    message_type : str
        One of ``"request"``, ``"response"``, ``"event"``, or ``"error"``.
    timestamp : float
        Unix epoch timestamp (seconds) when the message was created.
    correlation_id : str
        Unique identifier linking a request to its response(s).  Defaults to
        a new UUID-4 string if not supplied.
    """

    sender: str
    recipient: str
    content: dict[str, Any]
    message_type: str  # "request", "response", "event", "error"
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        valid_types = {"request", "response", "event", "error"}
        if self.message_type not in valid_types:
            raise ValueError(
                f"Invalid message_type '{self.message_type}'. Must be one of {sorted(valid_types)}."
            )


# ---------------------------------------------------------------------------
# Agent lifecycle states
# ---------------------------------------------------------------------------


class AgentState(Enum):
    """Possible lifecycle states for an agent."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


# ---------------------------------------------------------------------------
# Abstract base agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class for all agents in the teloscopy system.

    Subclasses must implement :meth:`handle_message` to define how
    incoming messages are processed.  The default :meth:`run` loop
    continuously reads from :attr:`message_queue` and dispatches to
    :meth:`handle_message`.

    Parameters
    ----------
    name : str
        Human-readable agent identifier (must be unique within an
        orchestration session).
    capabilities : list[str] | None
        List of capability tags that describe what the agent can do
        (e.g. ``["image_analysis", "segmentation"]``).
    """

    def __init__(
        self,
        name: str,
        capabilities: list[str] | None = None,
    ) -> None:
        self.name: str = name
        self.state: AgentState = AgentState.IDLE
        self._message_queue: asyncio.Queue[AgentMessage] | None = None
        self.capabilities: list[str] = capabilities or []
        self._message_router: _MessageRouter | None = None
        self._running: bool = False
        logger.info("Agent '%s' initialised with capabilities %s", name, self.capabilities)

    @property
    def message_queue(self) -> asyncio.Queue[AgentMessage]:
        """Lazily create the asyncio Queue (requires a running event loop on Python 3.9)."""
        if self._message_queue is None:
            self._message_queue = asyncio.Queue()
        return self._message_queue

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the agent's main processing loop.

        The loop continually awaits messages on :attr:`message_queue` and
        delegates each one to :meth:`handle_message`.  The loop exits when
        :attr:`_running` is set to ``False`` **or** when the agent
        transitions to :attr:`AgentState.COMPLETED`.
        """
        self._running = True
        self.state = AgentState.RUNNING
        logger.info("Agent '%s' entering run loop.", self.name)

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                logger.debug(
                    "Agent '%s' received %s from '%s' (cid=%s)",
                    self.name,
                    msg.message_type,
                    msg.sender,
                    msg.correlation_id,
                )

                try:
                    await self.handle_message(msg)
                except Exception:
                    logger.exception(
                        "Agent '%s' failed handling message cid=%s",
                        self.name,
                        msg.correlation_id,
                    )
                    self.state = AgentState.ERROR
                    # Send error reply back to the sender
                    await self.send_message(
                        recipient=msg.sender,
                        content={
                            "error": f"Agent '{self.name}' encountered an error.",
                            "correlation_id": msg.correlation_id,
                        },
                        msg_type="error",
                        correlation_id=msg.correlation_id,
                    )
        finally:
            self._running = False
            if self.state != AgentState.ERROR:
                self.state = AgentState.COMPLETED
            logger.info("Agent '%s' exited run loop (state=%s).", self.name, self.state.value)

    async def stop(self) -> None:
        """Signal the agent to exit its run loop gracefully."""
        self._running = False
        logger.info("Agent '%s' stop requested.", self.name)

    # ------------------------------------------------------------------
    # Abstract handler
    # ------------------------------------------------------------------

    @abstractmethod
    async def handle_message(self, msg: AgentMessage) -> None:
        """Process an incoming message.

        Subclasses **must** override this method to implement their domain
        logic.

        Parameters
        ----------
        msg : AgentMessage
            The message to process.
        """

    # ------------------------------------------------------------------
    # Outbound messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        recipient: str,
        content: dict[str, Any],
        msg_type: str = "request",
        correlation_id: str | None = None,
    ) -> AgentMessage:
        """Construct and send a message to another agent.

        If a :class:`_MessageRouter` has been attached (typically by the
        orchestrator) the message is routed through it; otherwise the call
        is a no-op that merely returns the constructed message.

        Parameters
        ----------
        recipient : str
            Name of the target agent.
        content : dict
            Message payload.
        msg_type : str
            Message type (default ``"request"``).
        correlation_id : str | None
            Optional correlation identifier to link related messages.

        Returns
        -------
        AgentMessage
            The message that was (or would have been) sent.
        """
        msg = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=msg_type,
            correlation_id=correlation_id or uuid.uuid4().hex,
        )

        if self._message_router is not None:
            await self._message_router.route(msg)
        else:
            logger.warning(
                "Agent '%s' has no message router; message to '%s' dropped.",
                self.name,
                recipient,
            )

        return msg

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of the agent's current status.

        Returns
        -------
        dict
            Keys: ``name``, ``state``, ``queue_size``, ``capabilities``.
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "queue_size": self._message_queue.qsize() if self._message_queue is not None else 0,
            "capabilities": list(self.capabilities),
        }


# ---------------------------------------------------------------------------
# Internal message router (used by the orchestrator)
# ---------------------------------------------------------------------------


class _MessageRouter:
    """Routes :class:`AgentMessage` objects to registered agents.

    This is an internal helper instantiated by the orchestrator.  Agents
    never interact with it directly.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent so that it can receive messages."""
        self._agents[agent.name] = agent
        agent._message_router = self

    def unregister(self, name: str) -> None:
        """Remove an agent from the registry."""
        agent = self._agents.pop(name, None)
        if agent is not None:
            agent._message_router = None

    async def route(self, msg: AgentMessage) -> None:
        """Deliver *msg* to its intended recipient's queue.

        Parameters
        ----------
        msg : AgentMessage
            The message to deliver.

        Raises
        ------
        KeyError
            If the recipient is not registered.
        """
        target = self._agents.get(msg.recipient)
        if target is None:
            logger.error(
                "Cannot route message to unknown agent '%s' (from '%s').",
                msg.recipient,
                msg.sender,
            )
            raise KeyError(f"Agent '{msg.recipient}' is not registered.")
        await target.message_queue.put(msg)
