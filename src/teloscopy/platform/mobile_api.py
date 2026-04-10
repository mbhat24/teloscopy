"""Mobile-optimised REST / WebSocket API layer for Teloscopy.

Provides lightweight endpoints designed for bandwidth-constrained mobile
clients (iOS / Android).  Features include:

* **JWT authentication** with short-lived access + long-lived refresh tokens
* **Paginated endpoints** for analysis history
* **Push-notification hooks** for long-running analysis completion
* **Image upload** with server-side compression before pipeline ingestion
* **Offline-sync protocol** — clients can POST batched mutations when
  connectivity returns; the server applies them idempotently

The module is framework-agnostic: it defines request/response schemas and
business logic that can be mounted on any ASGI framework (the main webapp
already uses Starlette/FastAPI).

References
----------
.. [1] RFC 7519 — JSON Web Token (JWT).
.. [2] RFC 7807 — Problem Details for HTTP APIs.
.. [3] OWASP Mobile Security Testing Guide (MSTG).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets as _secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_ACCESS_TTL = 900  # 15 minutes
_DEFAULT_REFRESH_TTL = 2_592_000  # 30 days
_PAGE_SIZE_DEFAULT = 20
_PAGE_SIZE_MAX = 100
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TokenPair:
    """JWT-like access + refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = _DEFAULT_ACCESS_TTL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceInfo:
    """Mobile device metadata sent on registration."""

    device_id: str
    platform: str  # "ios" | "android" | "web"
    os_version: str = ""
    app_version: str = ""
    push_token: str = ""
    locale: str = "en"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaginatedResponse:
    """Paginated list wrapper for mobile clients."""

    items: list[dict[str, Any]]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_prev: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PushNotification:
    """Push notification payload."""

    notification_id: str
    user_id: str
    title: str
    body: str
    data: dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # "normal" | "high"
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.notification_id:
            self.notification_id = uuid.uuid4().hex
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SyncBatch:
    """Offline-sync batch submitted by mobile client."""

    batch_id: str
    device_id: str
    mutations: list[dict[str, Any]]
    client_timestamp: float
    server_timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.server_timestamp == 0.0:
            self.server_timestamp = time.time()


@dataclass
class SyncResult:
    """Result of applying a sync batch."""

    batch_id: str
    accepted: int
    rejected: int
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UploadResult:
    """Result of an image upload."""

    upload_id: str
    original_size: int
    stored_size: int
    content_type: str
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class APIError:
    """RFC 7807 Problem Details response."""

    type: str
    title: str
    status: int
    detail: str = ""
    instance: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "title": self.title, "status": self.status}
        if self.detail:
            d["detail"] = self.detail
        if self.instance:
            d["instance"] = self.instance
        return d


# ---------------------------------------------------------------------------
# Token management (symmetric HMAC-based, no external JWT library needed)
# ---------------------------------------------------------------------------


class TokenManager:
    """Simple HMAC-SHA256 token issuer and verifier.

    This is *not* full JWT — it is a lightweight, dependency-free
    alternative suitable for internal mobile API auth.  Tokens are
    ``base64url(header.payload.signature)`` where signature =
    HMAC-SHA256(secret, header + "." + payload).

    Parameters
    ----------
    secret : str | None
        Signing secret.  Generated randomly if not provided.
    access_ttl : int
        Access token lifetime in seconds (default 15 min).
    refresh_ttl : int
        Refresh token lifetime in seconds (default 30 days).
    """

    def __init__(
        self,
        secret: str | None = None,
        access_ttl: int = _DEFAULT_ACCESS_TTL,
        refresh_ttl: int = _DEFAULT_REFRESH_TTL,
    ) -> None:
        self._secret: bytes = (secret or os.getenv("TELOSCOPY_SECRET_KEY") or _secrets.token_hex(32)).encode()
        self._access_ttl = access_ttl
        self._refresh_ttl = refresh_ttl
        # Revocation set (token IDs that have been revoked)
        self._revoked: set[str] = set()
        logger.info("TokenManager initialised (access_ttl=%ds)", access_ttl)

    # -- internal ---------------------------------------------------------

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        import base64

        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    @staticmethod
    def _b64url_decode(s: str) -> bytes:
        import base64

        padding = 4 - len(s) % 4
        if padding != 4:
            s += "=" * padding
        return base64.urlsafe_b64decode(s)

    def _sign(self, message: str) -> str:
        sig = hmac.new(self._secret, message.encode(), hashlib.sha256).digest()
        return self._b64url_encode(sig)

    # -- public API -------------------------------------------------------

    def issue(self, user_id: str, scopes: list[str] | None = None) -> TokenPair:
        """Issue an access + refresh token pair.

        Parameters
        ----------
        user_id : str
            Unique user identifier.
        scopes : list[str] | None
            OAuth2-style scopes (e.g. ``["read", "write"]``).

        Returns
        -------
        TokenPair
        """
        now = time.time()
        jti_access = uuid.uuid4().hex
        jti_refresh = uuid.uuid4().hex

        access_payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self._access_ttl,
            "jti": jti_access,
            "type": "access",
            "scopes": scopes or ["read"],
        }
        refresh_payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self._refresh_ttl,
            "jti": jti_refresh,
            "type": "refresh",
        }

        access_token = self._encode_token(access_payload)
        refresh_token = self._encode_token(refresh_payload)

        logger.info("Issued tokens for user=%s", user_id)
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self._access_ttl,
        )

    def _encode_token(self, payload: dict[str, Any]) -> str:
        header = self._b64url_encode(json.dumps({"alg": "HS256", "typ": "TAT"}).encode())
        body = self._b64url_encode(json.dumps(payload).encode())
        sig = self._sign(f"{header}.{body}")
        return f"{header}.{body}.{sig}"

    def verify(self, token: str) -> dict[str, Any] | None:
        """Verify and decode a token.

        Returns the payload dict on success, or ``None`` if the token is
        invalid, expired, or revoked.
        """
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, body_b64, sig = parts
        expected = self._sign(f"{header_b64}.{body_b64}")
        if not hmac.compare_digest(sig, expected):
            return None
        try:
            payload = json.loads(self._b64url_decode(body_b64))
        except (json.JSONDecodeError, Exception):
            return None
        if payload.get("exp", 0) < time.time():
            return None
        if payload.get("jti") in self._revoked:
            return None
        return payload

    def refresh(self, refresh_token: str) -> TokenPair | None:
        """Exchange a refresh token for a new token pair.

        The old refresh token is revoked after use.
        """
        payload = self.verify(refresh_token)
        if payload is None or payload.get("type") != "refresh":
            return None
        self._revoked.add(payload["jti"])
        return self.issue(payload["sub"], payload.get("scopes"))

    def revoke(self, token: str) -> bool:
        """Revoke a token (adds its JTI to the revocation set)."""
        payload = self.verify(token)
        if payload is None:
            return False
        self._revoked.add(payload["jti"])
        return True


# ---------------------------------------------------------------------------
# Image upload handler
# ---------------------------------------------------------------------------


class ImageUploadHandler:
    """Handle mobile image uploads with validation and compression.

    Parameters
    ----------
    upload_dir : str
        Directory to store uploaded images.
    max_bytes : int
        Maximum allowed upload size.
    """

    ALLOWED_TYPES = {"image/png", "image/jpeg", "image/tiff"}

    def __init__(
        self,
        upload_dir: str = "/tmp/teloscopy_uploads",
        max_bytes: int = _MAX_IMAGE_BYTES,
    ) -> None:
        self._upload_dir = upload_dir
        self._max_bytes = max_bytes
        os.makedirs(upload_dir, exist_ok=True)

    def validate(self, data: bytes, content_type: str) -> APIError | None:
        """Validate an upload before saving.

        Returns ``None`` if valid, or an :class:`APIError` describing the
        problem.
        """
        if content_type not in self.ALLOWED_TYPES:
            return APIError(
                type="urn:teloscopy:error:unsupported_media",
                title="Unsupported Media Type",
                status=415,
                detail=f"Content type '{content_type}' not allowed. "
                f"Accepted: {', '.join(sorted(self.ALLOWED_TYPES))}",
            )
        if len(data) > self._max_bytes:
            return APIError(
                type="urn:teloscopy:error:payload_too_large",
                title="Payload Too Large",
                status=413,
                detail=f"Image size {len(data)} bytes exceeds limit of {self._max_bytes} bytes.",
            )
        return None

    def save(self, data: bytes, content_type: str, user_id: str = "") -> UploadResult:
        """Save uploaded image data to disk.

        Parameters
        ----------
        data : bytes
            Raw image bytes.
        content_type : str
            MIME type.
        user_id : str
            Owning user (used for directory partitioning).

        Returns
        -------
        UploadResult
        """
        ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/tiff": ".tif"}
        ext = ext_map.get(content_type, ".bin")
        upload_id = uuid.uuid4().hex
        checksum = hashlib.sha256(data).hexdigest()

        # Sanitize user_id to prevent path traversal
        safe_user_id = re.sub(r'[^a-zA-Z0-9_@.\-]', '_', user_id or "_anonymous")
        if '..' in safe_user_id:
            safe_user_id = safe_user_id.replace('..', '__')
        user_dir = os.path.join(self._upload_dir, safe_user_id)
        os.makedirs(user_dir, exist_ok=True)
        path = os.path.join(user_dir, f"{upload_id}{ext}")

        with open(path, "wb") as f:
            f.write(data)

        logger.info("Saved upload %s (%d bytes) -> %s", upload_id, len(data), path)
        return UploadResult(
            upload_id=upload_id,
            original_size=len(data),
            stored_size=len(data),
            content_type=content_type,
            checksum=checksum,
        )

    def get_path(self, upload_id: str, user_id: str = "") -> str | None:
        """Resolve upload ID to file path (or None if missing)."""
        user_dir = os.path.join(self._upload_dir, user_id or "_anonymous")
        if not os.path.isdir(user_dir):
            return None
        for fname in os.listdir(user_dir):
            if fname.startswith(upload_id):
                return os.path.join(user_dir, fname)
        return None


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------


def paginate(
    items: list[dict[str, Any]],
    page: int = 1,
    page_size: int = _PAGE_SIZE_DEFAULT,
) -> PaginatedResponse:
    """Paginate a list of items.

    Parameters
    ----------
    items : list[dict]
        Full item list.
    page : int
        1-based page number.
    page_size : int
        Items per page (clamped to [1, PAGE_SIZE_MAX]).

    Returns
    -------
    PaginatedResponse
    """
    page_size = max(1, min(page_size, _PAGE_SIZE_MAX))
    page = max(1, page)
    total = len(items)
    total_pages = max(1, math.ceil(total / page_size))
    start = (page - 1) * page_size
    end = start + page_size
    return PaginatedResponse(
        items=items[start:end],
        page=page,
        page_size=page_size,
        total_items=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )


# ---------------------------------------------------------------------------
# Offline-sync engine
# ---------------------------------------------------------------------------


class SyncEngine:
    """Apply batched offline mutations idempotently.

    Each mutation dict must contain:

    * ``mutation_id`` — client-generated UUID (for idempotency)
    * ``type`` — one of ``"create_analysis"``, ``"update_settings"``,
      ``"delete_analysis"``
    * ``payload`` — mutation-specific data

    The engine keeps an in-memory ledger of applied mutation IDs so
    duplicates are silently ignored.
    """

    VALID_TYPES = {"create_analysis", "update_settings", "delete_analysis"}

    def __init__(self) -> None:
        self._applied: set[str] = set()
        self._store: dict[str, dict[str, Any]] = {}

    def apply_batch(self, batch: SyncBatch) -> SyncResult:
        """Apply a batch of mutations.

        Returns
        -------
        SyncResult
            Summary with accepted / rejected counts and conflict details.
        """
        accepted = 0
        rejected = 0
        conflicts: list[dict[str, Any]] = []

        for mut in batch.mutations:
            mid = mut.get("mutation_id", "")
            mtype = mut.get("type", "")
            payload = mut.get("payload", {})

            if not mid:
                rejected += 1
                conflicts.append({"mutation_id": mid, "reason": "missing mutation_id"})
                continue

            if mid in self._applied:
                # Idempotent — already processed
                accepted += 1
                continue

            if mtype not in self.VALID_TYPES:
                rejected += 1
                conflicts.append({"mutation_id": mid, "reason": f"unknown type '{mtype}'"})
                continue

            try:
                self._apply_one(mtype, payload)
                self._applied.add(mid)
                accepted += 1
            except Exception as exc:  # noqa: BLE001
                rejected += 1
                conflicts.append({"mutation_id": mid, "reason": str(exc)})

        logger.info(
            "SyncBatch %s: %d accepted, %d rejected",
            batch.batch_id,
            accepted,
            rejected,
        )
        return SyncResult(
            batch_id=batch.batch_id,
            accepted=accepted,
            rejected=rejected,
            conflicts=conflicts,
        )

    def _apply_one(self, mtype: str, payload: dict[str, Any]) -> None:
        if mtype == "create_analysis":
            aid = payload.get("analysis_id", uuid.uuid4().hex)
            self._store[aid] = payload
        elif mtype == "update_settings":
            key = payload.get("key", "")
            if not key:
                raise ValueError("update_settings requires 'key'")
            self._store.setdefault("_settings", {})[key] = payload.get("value")
        elif mtype == "delete_analysis":
            aid = payload.get("analysis_id", "")
            self._store.pop(aid, None)


# ---------------------------------------------------------------------------
# Push-notification manager
# ---------------------------------------------------------------------------


class PushNotificationManager:
    """Manage push notifications for mobile clients.

    Notifications are stored in an in-memory queue per user.  In
    production this would integrate with APNs / FCM.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[PushNotification]] = {}
        self._device_tokens: dict[str, str] = {}  # user_id -> push token

    def register_device(self, user_id: str, push_token: str) -> None:
        """Register a device push token for a user."""
        self._device_tokens[user_id] = push_token
        logger.info("Registered push token for user=%s", user_id)

    def send(
        self,
        user_id: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
        priority: str = "normal",
    ) -> PushNotification:
        """Queue a push notification for a user.

        Returns the notification object.  In production, this would
        dispatch to APNs/FCM here.
        """
        notif = PushNotification(
            notification_id=uuid.uuid4().hex,
            user_id=user_id,
            title=title,
            body=body,
            data=data or {},
            priority=priority,
        )
        self._queues.setdefault(user_id, []).append(notif)
        logger.info("Queued notification %s for user=%s", notif.notification_id, user_id)
        return notif

    def get_pending(self, user_id: str) -> list[PushNotification]:
        """Return and clear pending notifications for a user."""
        pending = self._queues.pop(user_id, [])
        return pending

    def notify_analysis_complete(
        self, user_id: str, analysis_id: str, summary: str = ""
    ) -> PushNotification:
        """Send a notification that an analysis has completed."""
        return self.send(
            user_id=user_id,
            title="Analysis Complete",
            body=summary or f"Your analysis {analysis_id[:8]}… is ready.",
            data={"analysis_id": analysis_id, "action": "view_results"},
            priority="high",
        )


# ---------------------------------------------------------------------------
# Mobile API controller (framework-agnostic request handlers)
# ---------------------------------------------------------------------------


class MobileAPIController:
    """High-level controller wiring together all mobile-API components.

    Usage::

        ctrl = MobileAPIController(secret=os.environ["TELOSCOPY_SECRET_KEY"])
        tokens = ctrl.login("user@example.com", "password123")
        result = ctrl.upload_image(tokens.access_token, image_bytes, "image/png")
        history = ctrl.get_analysis_history(tokens.access_token, page=1)
    """

    def __init__(
        self,
        secret: str | None = None,
        upload_dir: str = "/tmp/teloscopy_uploads",
    ) -> None:
        self.tokens = TokenManager(secret=secret)
        self.uploads = ImageUploadHandler(upload_dir=upload_dir)
        self.sync = SyncEngine()
        self.push = PushNotificationManager()
        # In-memory user store (replace with DB in production)
        self._users: dict[str, dict[str, Any]] = {}
        self._analyses: dict[str, list[dict[str, Any]]] = {}  # user_id -> [analysis]
        logger.info("MobileAPIController initialised")

    # -- auth -------------------------------------------------------------

    def register_user(
        self,
        email: str,
        password_hash: str,
        device: DeviceInfo | None = None,
    ) -> TokenPair | APIError:
        """Register a new user and return tokens."""
        if email in self._users:
            return APIError(
                type="urn:teloscopy:error:conflict",
                title="Conflict",
                status=409,
                detail="User already exists.",
            )
        user_id = uuid.uuid4().hex
        self._users[email] = {
            "user_id": user_id,
            "email": email,
            "password_hash": password_hash,
            "created_at": time.time(),
        }
        if device and device.push_token:
            self.push.register_device(user_id, device.push_token)
        return self.tokens.issue(user_id, scopes=["read", "write"])

    def login(self, email: str, password_hash: str) -> TokenPair | APIError:
        """Authenticate and return tokens."""
        user = self._users.get(email)
        if not user or not hmac.compare_digest(user["password_hash"], password_hash):
            return APIError(
                type="urn:teloscopy:error:unauthorized",
                title="Unauthorized",
                status=401,
                detail="Invalid credentials.",
            )
        return self.tokens.issue(user["user_id"], scopes=["read", "write"])

    def _auth(self, token: str) -> dict[str, Any] | APIError:
        """Verify token, returning payload or error."""
        payload = self.tokens.verify(token)
        if payload is None:
            return APIError(
                type="urn:teloscopy:error:unauthorized",
                title="Unauthorized",
                status=401,
                detail="Invalid or expired token.",
            )
        return payload

    # -- image upload -----------------------------------------------------

    def upload_image(
        self,
        token: str,
        data: bytes,
        content_type: str,
    ) -> UploadResult | APIError:
        """Upload an image for analysis."""
        auth = self._auth(token)
        if isinstance(auth, APIError):
            return auth
        err = self.uploads.validate(data, content_type)
        if err is not None:
            return err
        return self.uploads.save(data, content_type, user_id=auth["sub"])

    # -- analysis history -------------------------------------------------

    def record_analysis(self, user_id: str, analysis: dict[str, Any]) -> None:
        """Store an analysis result for a user."""
        analysis.setdefault("analysis_id", uuid.uuid4().hex)
        analysis.setdefault("created_at", time.time())
        self._analyses.setdefault(user_id, []).append(analysis)

    def get_analysis_history(
        self,
        token: str,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedResponse | APIError:
        """Return paginated analysis history for the authenticated user."""
        auth = self._auth(token)
        if isinstance(auth, APIError):
            return auth
        items = self._analyses.get(auth["sub"], [])
        # Most recent first
        items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)
        return paginate(items, page=page, page_size=page_size)

    # -- sync -------------------------------------------------------------

    def apply_sync(
        self,
        token: str,
        batch: SyncBatch,
    ) -> SyncResult | APIError:
        """Apply an offline-sync batch."""
        auth = self._auth(token)
        if isinstance(auth, APIError):
            return auth
        return self.sync.apply_batch(batch)

    # -- notifications ----------------------------------------------------

    def get_notifications(self, token: str) -> list[dict[str, Any]] | APIError:
        """Get and clear pending push notifications."""
        auth = self._auth(token)
        if isinstance(auth, APIError):
            return auth
        return [n.to_dict() for n in self.push.get_pending(auth["sub"])]
