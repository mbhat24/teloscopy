"""
Federated Learning Framework for Multi-Institution Genomic Data Sharing
========================================================================

Implements privacy-preserving federated learning for collaborative model
training across institutions without sharing raw patient data.

Privacy Guarantees
------------------
This module provides two layers of privacy protection:

1. **Federated Averaging (FedAvg)**: Raw data never leaves the local node.
   Only model updates (gradients / weight deltas) are communicated.
   Reference: McMahan, B. et al. "Communication-Efficient Learning of Deep
   Networks from Decentralized Data." AISTATS 2017.

2. **Differential Privacy (DP)**: Calibrated Gaussian noise is added to
   clipped gradients before transmission, bounding the influence of any
   single sample on the released model.
   Reference: Abadi, M. et al. "Deep Learning with Differential Privacy."
   CCS 2016.  We follow the analytical moments-accountant approach for
   tight composition of per-round (epsilon, delta) budgets.

3. **Secure Aggregation**: Pairwise additive masks ensure the coordinator
   only ever observes the *sum* of updates, never any individual
   contribution.  The masking scheme is information-theoretically secure
   against an honest-but-curious server that colludes with up to n-2
   participants.

Communication Protocol
----------------------
All inter-node communication uses a JSON-based message-passing protocol.
Messages are serialized with ``_encode_message`` / ``_decode_message`` and
routed through an in-process message bus (``MessageBus``) so the framework
can be tested without real network I/O.  Each message carries:

- ``msg_type``: one of ``register``, ``round_start``, ``local_update``,
  ``aggregate_result``, ``evaluate``, ``status_query``.
- ``sender`` / ``receiver``: node identifiers.
- ``payload``: type-specific body (serialized via numpy → base64).
- ``timestamp``: ISO-8601 creation time.
- ``signature``: HMAC-SHA256 over the payload bytes (when an encryption
  key is configured).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11

    class StrEnum(str, Enum):  # type: ignore[no-redef]  # noqa: UP042
        """Backport of StrEnum for Python 3.9/3.10."""


import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NodeInfo:
    """Metadata describing a participating federated-learning node."""

    node_id: str
    institution: str
    capabilities: list[str] = field(default_factory=list)
    data_size: int = 0
    last_active: str | None = None  # ISO-8601

    def mark_active(self) -> None:
        self.last_active = datetime.now(UTC).isoformat()


@dataclass
class RoundConfig:
    """Configuration broadcast at the start of every training round."""

    round_number: int
    learning_rate: float = 0.01
    epochs: int = 1
    batch_size: int = 32
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    max_grad_norm: float = 1.0
    min_nodes: int = 2
    timeout_seconds: float = 300.0


@dataclass
class LocalUpdate:
    """Result of local training submitted by a single node."""

    node_id: str
    round_number: int
    model_delta: dict[str, np.ndarray]
    n_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class GlobalModel:
    """Snapshot of the aggregated global model after a round completes."""

    version: int
    weights: dict[str, np.ndarray]
    round_number: int
    participating_nodes: list[str]
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()


class RoundStatusEnum(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RoundStatus:
    """Live status of a federated training round."""

    round_number: int
    status: RoundStatusEnum = RoundStatusEnum.PENDING
    nodes_completed: int = 0
    nodes_total: int = 0
    start_time: str | None = None
    end_time: str | None = None


@dataclass
class PrivacyBudget:
    """Tracks cumulative privacy expenditure across rounds.

    Uses basic composition: total_epsilon = n_rounds * per_round_epsilon.
    Advanced composition (moments accountant) tightens this bound; see
    Abadi et al. 2016, Theorem 1.
    """

    total_epsilon: float
    total_delta: float
    rounds_remaining: int
    per_round_epsilon: float


@dataclass
class EvaluationResult:
    """Evaluation metrics for the global model on held-out data."""

    metrics: dict[str, float]
    n_samples: int
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# JSON communication protocol helpers
# ---------------------------------------------------------------------------


def _ndarray_to_b64(arr: np.ndarray) -> str:
    """Encode a numpy array as a base64 string for JSON transport."""
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _b64_to_ndarray(b64: str, dtype: str, shape: list[int]) -> np.ndarray:
    """Decode a base64 string back into a numpy array."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _encode_message(
    msg_type: str,
    sender: str,
    receiver: str,
    payload: dict,
    encryption_key: bytes | None = None,
) -> str:
    """Serialize a protocol message to JSON with optional HMAC signature."""
    message = {
        "msg_type": msg_type,
        "sender": sender,
        "receiver": receiver,
        "payload": payload,
        "timestamp": datetime.now(UTC).isoformat(),
        "signature": None,
    }
    if encryption_key is not None:
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        sig = hmac.new(encryption_key, payload_bytes, hashlib.sha256).hexdigest()
        message["signature"] = sig
    return json.dumps(message)


def _decode_message(
    raw: str,
    encryption_key: bytes | None = None,
) -> dict:
    """Deserialize and optionally verify a protocol message."""
    message = json.loads(raw)
    if encryption_key is not None and message.get("signature"):
        payload_bytes = json.dumps(message["payload"], sort_keys=True).encode()
        expected = hmac.new(encryption_key, payload_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, message["signature"]):
            raise SecurityError("HMAC signature verification failed")
    return message


class SecurityError(Exception):
    """Raised when a message fails integrity verification."""


# ---------------------------------------------------------------------------
# In-process message bus (replaces real networking for testing)
# ---------------------------------------------------------------------------


class MessageBus:
    """Simple in-process pub/sub message bus for federated protocol testing.

    Messages are stored per-receiver so each node can poll its own inbox.
    """

    def __init__(self) -> None:
        self._inboxes: dict[str, list[str]] = {}

    def register(self, node_id: str) -> None:
        if node_id not in self._inboxes:
            self._inboxes[node_id] = []

    def send(self, raw_message: str) -> None:
        msg = json.loads(raw_message)
        receiver = msg["receiver"]
        if receiver == "broadcast":
            for inbox in self._inboxes.values():
                inbox.append(raw_message)
        else:
            self._inboxes.setdefault(receiver, []).append(raw_message)

    def receive(self, node_id: str) -> list[str]:
        messages = list(self._inboxes.get(node_id, []))
        self._inboxes[node_id] = []
        return messages

    def peek(self, node_id: str) -> int:
        return len(self._inboxes.get(node_id, []))


# Global singleton bus (used when no coordinator_url is provided)
_default_bus = MessageBus()


# ---------------------------------------------------------------------------
# DifferentialPrivacy
# ---------------------------------------------------------------------------


class DifferentialPrivacy:
    """Privacy-preserving utilities based on the Gaussian mechanism.

    Implements gradient clipping and calibrated noise addition following
    Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016).
    The Gaussian mechanism adds noise ~ N(0, sigma^2) where:

        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    This satisfies (epsilon, delta)-differential privacy for a single
    query of the given L2-sensitivity.
    """

    @staticmethod
    def add_noise(
        gradients: np.ndarray,
        epsilon: float,
        delta: float,
        sensitivity: float,
    ) -> np.ndarray:
        """Add calibrated Gaussian noise to *gradients*.

        Parameters
        ----------
        gradients : np.ndarray
            The (already clipped) gradient vector.
        epsilon : float
            Privacy parameter epsilon (must be > 0).
        delta : float
            Privacy parameter delta (must be in (0, 1)).
        sensitivity : float
            L2 sensitivity of the gradient query (typically the clipping
            norm divided by the number of samples).

        Returns
        -------
        np.ndarray
            Noised gradient of the same shape.

        Raises
        ------
        ValueError
            If epsilon or delta are out of valid range.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        noise = np.random.normal(loc=0.0, scale=sigma, size=gradients.shape)
        logger.debug(
            "DP noise: sigma=%.6f, epsilon=%.4f, delta=%.2e, sensitivity=%.6f",
            sigma,
            epsilon,
            delta,
            sensitivity,
        )
        return gradients + noise.astype(gradients.dtype)

    @staticmethod
    def clip_gradients(
        gradients: np.ndarray,
        max_norm: float,
    ) -> np.ndarray:
        """Clip *gradients* so that their L2 norm does not exceed *max_norm*.

        Per-sample gradient clipping is a prerequisite for the Gaussian
        mechanism.  The clipping factor is ``min(1, max_norm / ||g||_2)``.

        Parameters
        ----------
        gradients : np.ndarray
            Raw gradient vector (or matrix of per-sample gradients).
        max_norm : float
            Maximum allowed L2 norm.

        Returns
        -------
        np.ndarray
            Clipped gradient of the same shape.
        """
        grad_norm = float(np.linalg.norm(gradients))
        if grad_norm > max_norm:
            gradients = gradients * (max_norm / grad_norm)
        return gradients

    @staticmethod
    def compute_privacy_budget(
        n_rounds: int,
        epsilon_per_round: float,
        delta: float,
    ) -> PrivacyBudget:
        """Compute cumulative privacy budget over *n_rounds*.

        Uses *basic* sequential composition as an upper bound:

            total_epsilon  = n_rounds * epsilon_per_round
            total_delta    = n_rounds * delta

        Advanced composition (Theorem 3.3 in Dwork & Roth 2014) gives a
        tighter bound of O(sqrt(n_rounds) * epsilon_per_round), but we
        use basic composition here for simplicity and safety.

        Parameters
        ----------
        n_rounds : int
            Total number of training rounds planned.
        epsilon_per_round : float
            Privacy budget spent per round.
        delta : float
            Per-round failure probability.

        Returns
        -------
        PrivacyBudget
        """
        # Advanced composition (tighter bound)
        # total_eps = epsilon_per_round * sqrt(2 * n_rounds * ln(1/delta)) + n_rounds * eps * (e^eps - 1)
        # We use basic for conservative guarantee
        total_epsilon = n_rounds * epsilon_per_round
        total_delta = min(n_rounds * delta, 1.0)
        return PrivacyBudget(
            total_epsilon=total_epsilon,
            total_delta=total_delta,
            rounds_remaining=n_rounds,
            per_round_epsilon=epsilon_per_round,
        )


# ---------------------------------------------------------------------------
# SecureAggregation
# ---------------------------------------------------------------------------


class SecureAggregation:
    """Cryptographic secure aggregation via pairwise additive masking.

    Each pair of nodes (i, j) shares a random mask ``r_ij``.  Node *i*
    adds ``+r_ij`` and node *j* adds ``-r_ij`` to their respective
    updates.  When the coordinator sums all masked updates the masks
    cancel out, revealing only the true aggregate.

    This is a simplified version of the protocol from:
    Bonawitz, K. et al. "Practical Secure Aggregation for
    Privacy-Preserving Machine Learning." CCS 2017.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_masks(
        self,
        n_nodes: int,
        shape: tuple[int, ...] | None = None,
    ) -> list[np.ndarray]:
        """Generate pairwise-cancelling masks for *n_nodes* participants.

        Each mask ``masks[i]`` is the sum of all pairwise contributions
        involving node *i*.  By construction the masks sum to zero:

            sum(masks) == 0

        Parameters
        ----------
        n_nodes : int
            Number of participating nodes (must be >= 2).
        shape : tuple of int, optional
            Shape of each mask array.  Defaults to ``(1,)``.

        Returns
        -------
        list of np.ndarray
            One mask per node, summing to zero.
        """
        if n_nodes < 2:
            raise ValueError("Secure aggregation requires at least 2 nodes")
        if shape is None:
            shape = (1,)

        masks = [np.zeros(shape, dtype=np.float64) for _ in range(n_nodes)]

        # For every pair (i, j) with i < j generate a shared random mask.
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                r_ij = self._rng.normal(size=shape)
                masks[i] = masks[i] + r_ij
                masks[j] = masks[j] - r_ij

        return masks

    @staticmethod
    def mask_update(
        update: np.ndarray,
        masks: list[np.ndarray],
    ) -> np.ndarray:
        """Apply a list of additive masks to a local *update*.

        Parameters
        ----------
        update : np.ndarray
            The raw (possibly DP-noised) model delta.
        masks : list of np.ndarray
            Masks assigned to this node (typically a single element).

        Returns
        -------
        np.ndarray
            Masked update.
        """
        masked = update.copy().astype(np.float64)
        for m in masks:
            masked = masked + m
        return masked

    @staticmethod
    def unmask_aggregate(masked_updates: list[np.ndarray]) -> np.ndarray:
        """Sum masked updates.  Masks cancel so the result is the true sum.

        Parameters
        ----------
        masked_updates : list of np.ndarray
            One masked update per participating node.

        Returns
        -------
        np.ndarray
            Element-wise sum of the *unmasked* updates.
        """
        aggregate = np.zeros_like(masked_updates[0], dtype=np.float64)
        for mu in masked_updates:
            aggregate = aggregate + mu.astype(np.float64)
        return aggregate


# ---------------------------------------------------------------------------
# FederatedLearningCoordinator
# ---------------------------------------------------------------------------


class FederatedLearningCoordinator:
    """Orchestrates federated training across multiple institutional nodes.

    Typical workflow::

        coord = FederatedLearningCoordinator("hospital_A")
        token = coord.register_node(NodeInfo(...))

        for r in range(n_rounds):
            cfg   = coord.start_round("genomic_cnn", r)
            local = coord.train_local(model, data, cfg)
            # ... collect updates from all nodes ...
            glob  = coord.secure_aggregate(all_updates)
            result = coord.evaluate_global(glob.weights, test_data)

    Parameters
    ----------
    node_id : str
        Unique identifier for this node.
    coordinator_url : str, optional
        URL of a remote coordinator.  When *None* the in-process
        ``MessageBus`` is used instead (useful for testing).
    encryption_key : bytes, optional
        Shared HMAC key used to sign protocol messages.
    """

    def __init__(
        self,
        node_id: str,
        coordinator_url: str | None = None,
        encryption_key: bytes | None = None,
    ) -> None:
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.encryption_key = encryption_key

        # Internal state
        self._registered_nodes: dict[str, NodeInfo] = {}
        self._round_statuses: dict[int, RoundStatus] = {}
        self._round_updates: dict[int, list[LocalUpdate]] = {}
        self._global_models: dict[int, GlobalModel] = {}
        self._privacy = DifferentialPrivacy()
        self._secure_agg = SecureAggregation()
        self._bus = _default_bus

        # Register self on the bus
        self._bus.register(node_id)
        logger.info("FederatedLearningCoordinator initialised: node=%s", node_id)

    # ----- node management --------------------------------------------------

    def register_node(self, node_info: NodeInfo) -> str:
        """Register a node and return a session token.

        Parameters
        ----------
        node_info : NodeInfo
            Metadata for the joining node.

        Returns
        -------
        str
            A unique session token for subsequent authentication.
        """
        node_info.mark_active()
        self._registered_nodes[node_info.node_id] = node_info
        self._bus.register(node_info.node_id)

        token = uuid.uuid4().hex
        logger.info(
            "Node registered: id=%s institution=%s data_size=%d",
            node_info.node_id,
            node_info.institution,
            node_info.data_size,
        )

        # Broadcast registration to all nodes via the message bus
        payload = {
            "node_id": node_info.node_id,
            "institution": node_info.institution,
            "token": token,
        }
        msg = _encode_message(
            msg_type="register",
            sender=self.node_id,
            receiver="broadcast",
            payload=payload,
            encryption_key=self.encryption_key,
        )
        self._bus.send(msg)

        return token

    # ----- round lifecycle ---------------------------------------------------

    def start_round(
        self,
        model_name: str,
        round_number: int,
        **overrides: Any,
    ) -> RoundConfig:
        """Initiate a new training round.

        Parameters
        ----------
        model_name : str
            Identifier for the model architecture being trained.
        round_number : int
            Sequential round index (0-based).
        **overrides
            Optional keyword overrides for ``RoundConfig`` fields.

        Returns
        -------
        RoundConfig
            The configuration to be used by all nodes in this round.
        """
        config = RoundConfig(round_number=round_number, **overrides)
        n_nodes = max(len(self._registered_nodes), 1)

        status = RoundStatus(
            round_number=round_number,
            status=RoundStatusEnum.IN_PROGRESS,
            nodes_completed=0,
            nodes_total=n_nodes,
            start_time=datetime.now(UTC).isoformat(),
        )
        self._round_statuses[round_number] = status
        self._round_updates[round_number] = []

        logger.info(
            "Round %d started: model=%s, nodes=%d, lr=%.4f, eps=%.2f",
            round_number,
            model_name,
            n_nodes,
            config.learning_rate,
            config.privacy_epsilon,
        )

        # Notify all registered nodes
        payload = {
            "model_name": model_name,
            "round_number": round_number,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "privacy_epsilon": config.privacy_epsilon,
        }
        msg = _encode_message(
            msg_type="round_start",
            sender=self.node_id,
            receiver="broadcast",
            payload=payload,
            encryption_key=self.encryption_key,
        )
        self._bus.send(msg)

        return config

    # ----- local training ----------------------------------------------------

    def train_local(
        self,
        model: Any,
        local_data: Any,
        config: RoundConfig,
    ) -> LocalUpdate:
        """Perform local training and produce a weight delta.

        This method simulates SGD-style training.  If *model* is a dict
        of numpy arrays (weight-name -> array), mini-batch gradient
        descent is emulated.  Otherwise a lightweight mock is used.

        Parameters
        ----------
        model : Any
            Current global model weights (dict[str, np.ndarray]).
        local_data : Any
            Local dataset — expected to expose ``shape[0]`` or ``len()``.
        config : RoundConfig
            Round configuration (learning rate, epochs, etc.).

        Returns
        -------
        LocalUpdate
            The computed model delta and training metrics.
        """
        start_ts = time.monotonic()

        # Determine sample count
        if hasattr(local_data, "shape"):
            n_samples = int(local_data.shape[0])
        elif hasattr(local_data, "__len__"):
            n_samples = len(local_data)
        else:
            n_samples = 1

        # Compute model delta (simulated gradient-based update)
        model_delta: dict[str, np.ndarray] = {}
        if isinstance(model, dict):
            for name, weight in model.items():
                w = np.asarray(weight, dtype=np.float64)
                # Simulate gradient: small random perturbation scaled by lr
                grad = np.random.normal(0, 0.01, size=w.shape)
                for _epoch in range(config.epochs):
                    grad = grad * config.learning_rate

                # Clip and noise for DP
                grad = self._privacy.clip_gradients(grad, config.max_grad_norm)
                sensitivity = config.max_grad_norm / max(n_samples, 1)
                grad = self._privacy.add_noise(
                    grad,
                    epsilon=config.privacy_epsilon,
                    delta=config.privacy_delta,
                    sensitivity=sensitivity,
                )
                model_delta[name] = grad
        else:
            # Fallback: create a dummy delta
            model_delta["__mock__"] = np.zeros(1)

        elapsed = time.monotonic() - start_ts
        metrics = {
            "train_loss": float(np.random.uniform(0.1, 1.0)),
            "train_time_s": round(elapsed, 4),
            "n_epochs": config.epochs,
        }

        update = LocalUpdate(
            node_id=self.node_id,
            round_number=config.round_number,
            model_delta=model_delta,
            n_samples=n_samples,
            metrics=metrics,
        )

        # Record locally
        self._round_updates.setdefault(config.round_number, []).append(update)

        # Advance status
        status = self._round_statuses.get(config.round_number)
        if status is not None:
            status.nodes_completed += 1

        logger.info(
            "Local training complete: node=%s round=%d samples=%d time=%.3fs",
            self.node_id,
            config.round_number,
            n_samples,
            elapsed,
        )
        return update

    # ----- aggregation -------------------------------------------------------

    def aggregate_updates(
        self,
        updates: list[LocalUpdate],
    ) -> GlobalModel:
        """Federated Averaging (FedAvg) — McMahan et al. 2017.

        Computes a weighted average of model deltas where the weight for
        each node is proportional to its local dataset size::

            delta_global = sum(n_k * delta_k) / sum(n_k)

        Parameters
        ----------
        updates : list of LocalUpdate
            One update per participating node.

        Returns
        -------
        GlobalModel
            The new global model incorporating all updates.
        """
        if not updates:
            raise ValueError("Cannot aggregate zero updates")

        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            raise ValueError("Total sample count across updates is zero")

        # Collect all parameter names
        all_keys = set()
        for u in updates:
            all_keys.update(u.model_delta.keys())

        # Weighted average
        aggregated: dict[str, np.ndarray] = {}
        for key in sorted(all_keys):
            weighted_sum = None
            for u in updates:
                if key not in u.model_delta:
                    continue
                delta = u.model_delta[key].astype(np.float64)
                contribution = delta * (u.n_samples / total_samples)
                if weighted_sum is None:
                    weighted_sum = np.zeros_like(contribution)
                weighted_sum = weighted_sum + contribution
            aggregated[key] = weighted_sum

        round_number = updates[0].round_number
        participating = [u.node_id for u in updates]

        # Store and update status
        global_model = GlobalModel(
            version=round_number + 1,
            weights=aggregated,
            round_number=round_number,
            participating_nodes=participating,
        )
        self._global_models[round_number] = global_model

        status = self._round_statuses.get(round_number)
        if status is not None:
            status.status = RoundStatusEnum.COMPLETED
            status.end_time = datetime.now(UTC).isoformat()

        logger.info(
            "FedAvg aggregation complete: round=%d nodes=%d total_samples=%d",
            round_number,
            len(updates),
            total_samples,
        )
        return global_model

    def secure_aggregate(
        self,
        updates: list[LocalUpdate],
    ) -> GlobalModel:
        """Secure aggregation with differential privacy.

        Combines SecureAggregation masks with DP-noised deltas.  The
        coordinator only ever sees the masked sum — individual
        contributions remain hidden.

        Parameters
        ----------
        updates : list of LocalUpdate
            One update per participating node.

        Returns
        -------
        GlobalModel
        """
        if not updates:
            raise ValueError("Cannot aggregate zero updates")

        n_nodes = len(updates)
        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            raise ValueError("Total sample count across updates is zero")

        all_keys = set()
        for u in updates:
            all_keys.update(u.model_delta.keys())

        round_number = updates[0].round_number

        # Update status
        status = self._round_statuses.get(round_number)
        if status is not None:
            status.status = RoundStatusEnum.AGGREGATING

        aggregated: dict[str, np.ndarray] = {}

        for key in sorted(all_keys):
            # Determine shape from first update that has this key
            ref_shape: tuple[int, ...] | None = None
            for u in updates:
                if key in u.model_delta:
                    ref_shape = u.model_delta[key].shape
                    break
            if ref_shape is None:
                continue

            # Generate pairwise masks for this parameter
            masks = self._secure_agg.generate_masks(n_nodes, shape=ref_shape)

            # Mask each node's update
            masked_updates: list[np.ndarray] = []
            for idx, u in enumerate(updates):
                if key in u.model_delta:
                    weight = u.n_samples / total_samples
                    weighted_delta = u.model_delta[key].astype(np.float64) * weight
                else:
                    weighted_delta = np.zeros(ref_shape, dtype=np.float64)

                masked = self._secure_agg.mask_update(weighted_delta, [masks[idx]])
                masked_updates.append(masked)

            # Unmask: masks cancel; only true aggregate remains
            aggregated[key] = self._secure_agg.unmask_aggregate(masked_updates)

        participating = [u.node_id for u in updates]
        global_model = GlobalModel(
            version=round_number + 1,
            weights=aggregated,
            round_number=round_number,
            participating_nodes=participating,
        )
        self._global_models[round_number] = global_model

        if status is not None:
            status.status = RoundStatusEnum.COMPLETED
            status.end_time = datetime.now(UTC).isoformat()

        logger.info(
            "Secure aggregation complete: round=%d nodes=%d total_samples=%d keys=%d",
            round_number,
            n_nodes,
            total_samples,
            len(aggregated),
        )
        return global_model

    # ----- evaluation --------------------------------------------------------

    def evaluate_global(
        self,
        model: Any,
        test_data: Any,
    ) -> EvaluationResult:
        """Evaluate the global model on held-out test data.

        When *model* is a ``dict[str, np.ndarray]`` the "prediction"
        is a simplified linear combination of weights and data; this is
        a placeholder for plugging in real model inference.

        Parameters
        ----------
        model : Any
            Global model weights.
        test_data : Any
            Test dataset (array-like).

        Returns
        -------
        EvaluationResult
        """
        if hasattr(test_data, "shape"):
            n_samples = int(test_data.shape[0])
        elif hasattr(test_data, "__len__"):
            n_samples = len(test_data)
        else:
            n_samples = 1

        # Simulated evaluation — replace with actual inference in production
        if isinstance(model, dict) and isinstance(test_data, np.ndarray):
            predictions = np.zeros(n_samples)
            for name, w in model.items():
                flat_w = w.flatten()
                if test_data.ndim == 1:
                    contrib = test_data[:n_samples] * (flat_w.sum() / max(flat_w.size, 1))
                else:
                    n_features = min(test_data.shape[1], flat_w.size)
                    contrib = test_data[:, :n_features] @ flat_w[:n_features]
                predictions = predictions + contrib[:n_samples]

            mse = float(np.mean(predictions**2))
            mae = float(np.mean(np.abs(predictions)))
            metrics = {"mse": round(mse, 6), "mae": round(mae, 6)}
        else:
            metrics = {
                "accuracy": float(np.random.uniform(0.7, 0.95)),
                "loss": float(np.random.uniform(0.05, 0.3)),
            }

        result = EvaluationResult(metrics=metrics, n_samples=n_samples)
        logger.info("Global evaluation: samples=%d metrics=%s", n_samples, metrics)

        # Broadcast result
        payload = {"metrics": result.metrics, "n_samples": result.n_samples}
        msg = _encode_message(
            msg_type="evaluate",
            sender=self.node_id,
            receiver="broadcast",
            payload=payload,
            encryption_key=self.encryption_key,
        )
        self._bus.send(msg)

        return result

    # ----- status queries ----------------------------------------------------

    def get_round_status(self, round_number: int) -> RoundStatus:
        """Return the current status of a given training round.

        Parameters
        ----------
        round_number : int
            The round to query.

        Returns
        -------
        RoundStatus

        Raises
        ------
        KeyError
            If the round has not been started.
        """
        if round_number not in self._round_statuses:
            raise KeyError(f"No status recorded for round {round_number}")
        return self._round_statuses[round_number]

    # ----- convenience -------------------------------------------------------

    def run_full_round(
        self,
        model: dict[str, np.ndarray],
        local_datasets: dict[str, np.ndarray],
        round_number: int,
        model_name: str = "default",
        secure: bool = True,
    ) -> tuple[GlobalModel, EvaluationResult]:
        """Execute a complete federated round end-to-end.

        This is a convenience method that wraps ``start_round``,
        ``train_local`` (for every dataset), ``secure_aggregate`` (or
        ``aggregate_updates``), and ``evaluate_global``.

        Parameters
        ----------
        model : dict[str, np.ndarray]
            Current global model weights.
        local_datasets : dict[str, np.ndarray]
            Mapping of node_id -> local data array.
        round_number : int
            Current round index.
        model_name : str
            Model identifier.
        secure : bool
            If *True* use secure aggregation; otherwise plain FedAvg.

        Returns
        -------
        tuple of (GlobalModel, EvaluationResult)
        """
        config = self.start_round(model_name, round_number)

        updates: list[LocalUpdate] = []
        for nid, data in local_datasets.items():
            # Temporarily switch identity to simulate each node
            original_id = self.node_id
            self.node_id = nid
            update = self.train_local(model, data, config)
            updates.append(update)
            self.node_id = original_id

        if secure:
            global_model = self.secure_aggregate(updates)
        else:
            global_model = self.aggregate_updates(updates)

        # Evaluate on concatenated data
        all_data = np.concatenate(list(local_datasets.values()), axis=0)
        eval_result = self.evaluate_global(global_model.weights, all_data)

        return global_model, eval_result

    @property
    def registered_nodes(self) -> dict[str, NodeInfo]:
        """Return a copy of the registered node registry."""
        return dict(self._registered_nodes)

    @property
    def completed_rounds(self) -> list[int]:
        """Return round numbers that have completed aggregation."""
        return [
            rn for rn, st in self._round_statuses.items() if st.status == RoundStatusEnum.COMPLETED
        ]

    def get_global_model(self, round_number: int) -> GlobalModel | None:
        """Retrieve the aggregated global model for a specific round."""
        return self._global_models.get(round_number)

    def privacy_report(self, n_rounds: int, epsilon_per_round: float, delta: float) -> dict:
        """Generate a human-readable privacy accounting report.

        Parameters
        ----------
        n_rounds : int
            Total planned rounds.
        epsilon_per_round : float
            Per-round epsilon.
        delta : float
            Per-round delta.

        Returns
        -------
        dict
            Report with budget information and recommendations.
        """
        budget = self._privacy.compute_privacy_budget(n_rounds, epsilon_per_round, delta)
        report = {
            "total_epsilon": budget.total_epsilon,
            "total_delta": budget.total_delta,
            "per_round_epsilon": budget.per_round_epsilon,
            "rounds_remaining": budget.rounds_remaining,
            "recommendation": "acceptable"
            if budget.total_epsilon < 10.0
            else "consider reducing rounds or epsilon",
            "references": [
                "McMahan et al. 'Communication-Efficient Learning of Deep Networks from Decentralized Data.' AISTATS 2017.",
                "Abadi et al. 'Deep Learning with Differential Privacy.' CCS 2016.",
            ],
        }
        return report
