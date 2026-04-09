"""
User feedback loop module for continuous improvement of teloscopy analysis.

Provides mechanisms to collect user feedback, record corrections to model
outputs, analyze feedback patterns, and prepare retraining data so that
the system improves over time.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset({"spot_detection", "disease_risk", "diet", "facial"})
RATING_MIN = 1
RATING_MAX = 5


# --------------------------------------------------------------------------- #
#  Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class FeedbackEntry:
    """A single piece of user feedback tied to an analysis job.

    Attributes:
        entry_id:  Unique identifier for this feedback entry.
        timestamp: ISO-8601 UTC timestamp of when the feedback was recorded.
        job_id:    Identifier of the analysis job being reviewed.
        category:  Subsystem (spot_detection | disease_risk | diet | facial).
        rating:    Numeric quality rating from 1 (poor) to 5 (excellent).
        comment:   Optional free-text comment from the user.
        metadata:  Arbitrary additional context supplied by the caller.
    """

    entry_id: str
    timestamp: str
    job_id: str
    category: str
    rating: int
    comment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FeedbackEntry:
        """Reconstruct from a dictionary."""
        return cls(**data)


@dataclass
class CorrectionEntry:
    """Records a user correction to a specific field of a model output.

    Attributes:
        correction_id:  Unique identifier for this correction.
        job_id:         The analysis job whose output is being corrected.
        field:          Dot-delimited path of the corrected field.
        original_value: The value the model originally produced.
        corrected_value: The value the user believes is correct.
        reason:         Optional explanation for the correction.
        timestamp:      ISO-8601 UTC timestamp.
    """

    correction_id: str
    job_id: str
    field: str
    original_value: Any
    corrected_value: Any
    reason: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CorrectionEntry:
        """Reconstruct from a dictionary."""
        return cls(**data)


@dataclass
class FeedbackSummary:
    """Aggregated feedback metrics over a given time window.

    Attributes:
        total_entries:      Number of feedback entries in the window.
        avg_rating:         Mean rating across all entries.
        category_breakdown: Mapping of category -> {count, avg_rating}.
        trend:              One of 'improving', 'declining', or 'stable'.
        period_days:        Length of the aggregation window in days.
        total_corrections:  Number of correction entries in the window.
    """

    total_entries: int
    avg_rating: float
    category_breakdown: dict[str, dict[str, float]]
    trend: str
    period_days: int
    total_corrections: int = 0

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)


@dataclass
class ImprovementSuggestion:
    """An actionable suggestion for improving a specific subsystem.

    Attributes:
        area:           The subsystem or category targeted.
        description:    Human-readable description of the suggestion.
        priority:       Priority level — 'high', 'medium', or 'low'.
        evidence_count: How many feedback/correction entries support it.
    """

    area: str
    description: str
    priority: str
    evidence_count: int

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)


@dataclass
class TrainingBatch:
    """A batch of training samples derived from user corrections.

    Attributes:
        batch_id:   Unique identifier for this batch.
        model_name: Name of the model the batch targets.
        samples:    List of input samples (feature dicts).
        labels:     Corresponding corrected labels / values.
        metadata:   Extra information (creation time, source stats, etc.).
    """

    batch_id: str
    model_name: str
    samples: list[dict[str, Any]]
    labels: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)


@dataclass
class ImprovementReport:
    """Before/after comparison following a model retrain cycle.

    Attributes:
        report_id:      Unique identifier for this report.
        model_name:     Name of the retrained model.
        metric_changes: Mapping of metric_name -> {before, after, delta}.
        recommendation: 'deploy', 'rollback', or 'neutral'.
        created_at:     ISO-8601 timestamp of report creation.
    """

    report_id: str
    model_name: str
    metric_changes: dict[str, dict[str, float]]
    recommendation: str
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)


# --------------------------------------------------------------------------- #
#  FeedbackCollector                                                           #
# --------------------------------------------------------------------------- #


class FeedbackCollector:
    """Central hub for recording, querying, and exporting user feedback.

    All data is persisted as line-delimited JSON files under *storage_path*:
        ``feedback.jsonl``    — one :class:`FeedbackEntry` per line
        ``corrections.jsonl`` — one :class:`CorrectionEntry` per line

    Args:
        storage_path: Directory for feedback data files (created if absent).
    """

    FEEDBACK_FILE = "feedback.jsonl"
    CORRECTIONS_FILE = "corrections.jsonl"

    def __init__(self, storage_path: str = "/tmp/teloscopy_feedback") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._feedback_path = self.storage_path / self.FEEDBACK_FILE
        self._corrections_path = self.storage_path / self.CORRECTIONS_FILE
        self._feedback_path.touch(exist_ok=True)
        self._corrections_path.touch(exist_ok=True)
        logger.info("FeedbackCollector initialised — storage at %s", self.storage_path)

    # -- Recording --------------------------------------------------------- #

    def record_feedback(
        self,
        job_id: str,
        category: str,
        rating: int,
        comment: str = "",
        metadata: dict | None = None,
    ) -> FeedbackEntry:
        """Record a user feedback entry for a completed analysis job.

        Args:
            job_id:   Identifier of the analysis job.
            category: One of ``spot_detection``, ``disease_risk``, ``diet``, ``facial``.
            rating:   Integer rating from 1 (worst) to 5 (best).
            comment:  Optional free-text comment.
            metadata: Optional dict of additional context.

        Returns:
            The newly created :class:`FeedbackEntry`.

        Raises:
            ValueError: If *category* is invalid or *rating* is out of range.
        """
        category = category.strip().lower()
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of {sorted(VALID_CATEGORIES)}."
            )
        if not (RATING_MIN <= rating <= RATING_MAX):
            raise ValueError(f"Rating must be {RATING_MIN}–{RATING_MAX}, got {rating}.")

        entry = FeedbackEntry(
            entry_id=uuid.uuid4().hex,
            timestamp=datetime.now(UTC).isoformat(),
            job_id=job_id,
            category=category,
            rating=rating,
            comment=comment,
            metadata=metadata or {},
        )
        self._append_jsonl(self._feedback_path, entry.to_dict())
        logger.info(
            "Recorded feedback %s for job %s [%s] rating=%d",
            entry.entry_id,
            job_id,
            category,
            rating,
        )
        return entry

    def record_correction(
        self,
        job_id: str,
        field: str,
        original_value: Any,
        corrected_value: Any,
        reason: str = "",
    ) -> CorrectionEntry:
        """Record a user-supplied correction to a model output field.

        Args:
            job_id:          Identifier of the analysis job.
            field:           Dot-delimited field path (e.g. ``"spots.count"``).
            original_value:  The value the model produced.
            corrected_value: The value the user considers correct.
            reason:          Optional explanation.

        Returns:
            The newly created :class:`CorrectionEntry`.

        Raises:
            ValueError: If *field* is empty or values are identical.
        """
        if not field or not field.strip():
            raise ValueError("Field name must not be empty.")
        if original_value == corrected_value:
            raise ValueError("Corrected value is identical to the original.")

        correction = CorrectionEntry(
            correction_id=uuid.uuid4().hex,
            job_id=job_id,
            field=field.strip(),
            original_value=original_value,
            corrected_value=corrected_value,
            reason=reason,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._append_jsonl(self._corrections_path, correction.to_dict())
        logger.info(
            "Recorded correction %s for job %s field=%s", correction.correction_id, job_id, field
        )
        return correction

    # -- Querying / aggregation -------------------------------------------- #

    def get_feedback_summary(self, days: int = 30) -> FeedbackSummary:
        """Compute aggregated feedback metrics over a rolling window.

        Args:
            days: Number of past days to include (default 30).

        Returns:
            A :class:`FeedbackSummary` with counts, averages, breakdowns, and trend.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        entries = self._load_feedback(since=cutoff)
        corrections = self._load_corrections(since=cutoff)

        total = len(entries)
        if total == 0:
            return FeedbackSummary(
                total_entries=0,
                avg_rating=0.0,
                category_breakdown={},
                trend="stable",
                period_days=days,
                total_corrections=len(corrections),
            )

        ratings = [e.rating for e in entries]
        avg_rating = round(sum(ratings) / total, 2)

        cat_groups: dict[str, list[int]] = defaultdict(list)
        for e in entries:
            cat_groups[e.category].append(e.rating)

        category_breakdown: dict[str, dict[str, float]] = {}
        for cat, cat_ratings in cat_groups.items():
            category_breakdown[cat] = {
                "count": float(len(cat_ratings)),
                "avg_rating": round(sum(cat_ratings) / len(cat_ratings), 2),
            }

        trend = self._compute_trend(entries)
        summary = FeedbackSummary(
            total_entries=total,
            avg_rating=avg_rating,
            category_breakdown=category_breakdown,
            trend=trend,
            period_days=days,
            total_corrections=len(corrections),
        )
        logger.info(
            "Feedback summary (%dd): %d entries, avg %.2f, trend=%s", days, total, avg_rating, trend
        )
        return summary

    def get_improvement_suggestions(self) -> list[ImprovementSuggestion]:
        """Analyse all feedback and corrections to suggest system improvements.

        Heuristics:
        * Categories with average rating < 3.0 → **high** priority.
        * Areas with ≥ 10 corrections → **medium** priority.
        * Categories with declining 60-day trend → **low** priority.

        Returns:
            List of :class:`ImprovementSuggestion`, sorted high → low.
        """
        suggestions: list[ImprovementSuggestion] = []
        all_feedback = self._load_feedback()
        all_corrections = self._load_corrections()

        # Low average rating
        cat_ratings: dict[str, list[int]] = defaultdict(list)
        for fb in all_feedback:
            cat_ratings[fb.category].append(fb.rating)

        for cat, ratings in cat_ratings.items():
            avg = sum(ratings) / len(ratings) if ratings else 0
            if avg < 3.0:
                suggestions.append(
                    ImprovementSuggestion(
                        area=cat,
                        description=(
                            f"Average rating for '{cat}' is {avg:.2f} (below 3.0). "
                            "Consider model retraining or threshold tuning."
                        ),
                        priority="high",
                        evidence_count=len(ratings),
                    )
                )

        # Frequent corrections
        correction_counts: Counter = Counter()
        field_counts: dict[str, Counter] = defaultdict(Counter)
        for c in all_corrections:
            top_field = c.field.split(".")[0]
            correction_counts[top_field] += 1
            field_counts[top_field][c.field] += 1

        for area, count in correction_counts.items():
            if count >= 10:
                most_common_field, mc_count = field_counts[area].most_common(1)[0]
                suggestions.append(
                    ImprovementSuggestion(
                        area=area,
                        description=(
                            f"'{area}' has {count} corrections; most corrected "
                            f"field: '{most_common_field}' ({mc_count}x). "
                            "Consider generating a training batch."
                        ),
                        priority="medium",
                        evidence_count=count,
                    )
                )

        # Declining trend (60 days)
        cutoff_60 = datetime.now(UTC) - timedelta(days=60)
        recent_cats: dict[str, list[FeedbackEntry]] = defaultdict(list)
        for fb in all_feedback:
            if datetime.fromisoformat(fb.timestamp) >= cutoff_60:
                recent_cats[fb.category].append(fb)

        for cat, entries in recent_cats.items():
            if self._compute_trend(entries) == "declining":
                suggestions.append(
                    ImprovementSuggestion(
                        area=cat,
                        description=(
                            f"Satisfaction for '{cat}' is declining over the "
                            "last 60 days. Investigate recent changes."
                        ),
                        priority="low",
                        evidence_count=len(entries),
                    )
                )

        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 99))
        logger.info("Generated %d improvement suggestions.", len(suggestions))
        return suggestions

    # -- Export ------------------------------------------------------------ #

    def export_training_data(self, output_path: str, category: str | None = None) -> int:
        """Export corrections as line-delimited JSON training data.

        Each line has keys ``input``, ``label``, ``field``, ``job_id``, ``reason``.

        Args:
            output_path: Destination file (overwritten if it exists).
            category:    Optional filter matched against the field prefix.

        Returns:
            Number of samples written.
        """
        corrections = self._load_corrections()
        if category:
            category = category.strip().lower()
            corrections = [c for c in corrections if c.field.split(".")[0].lower() == category]

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(out, "w", encoding="utf-8") as fh:
            for c in corrections:
                sample = {
                    "input": c.original_value,
                    "label": c.corrected_value,
                    "field": c.field,
                    "job_id": c.job_id,
                    "reason": c.reason,
                }
                fh.write(json.dumps(sample, default=str) + "\n")
                count += 1
        logger.info(
            "Exported %d training samples to %s (category=%s).",
            count,
            output_path,
            category or "all",
        )
        return count

    # -- Internal helpers -------------------------------------------------- #

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        """Append a single JSON record as a new line to *path*."""
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def _load_feedback(self, since: datetime | None = None) -> list[FeedbackEntry]:
        """Load feedback entries, optionally filtering by timestamp."""
        entries: list[FeedbackEntry] = []
        for record in self._read_jsonl(self._feedback_path):
            try:
                entry = FeedbackEntry.from_dict(record)
            except (TypeError, KeyError) as exc:
                logger.warning("Skipping malformed feedback record: %s", exc)
                continue
            if since and datetime.fromisoformat(entry.timestamp) < since:
                continue
            entries.append(entry)
        return entries

    def _load_corrections(self, since: datetime | None = None) -> list[CorrectionEntry]:
        """Load correction entries, optionally filtering by timestamp."""
        entries: list[CorrectionEntry] = []
        for record in self._read_jsonl(self._corrections_path):
            try:
                entry = CorrectionEntry.from_dict(record)
            except (TypeError, KeyError) as exc:
                logger.warning("Skipping malformed correction record: %s", exc)
                continue
            if since and datetime.fromisoformat(entry.timestamp) < since:
                continue
            entries.append(entry)
        return entries

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        """Read all JSON objects from a line-delimited JSON file."""
        records: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("JSON decode error at %s:%d — %s", path, lineno, exc)
        return records

    @staticmethod
    def _compute_trend(entries: list[FeedbackEntry]) -> str:
        """Determine whether ratings are improving, declining, or stable.

        Splits entries chronologically into two halves and compares averages.
        A difference > 0.3 triggers a label; otherwise ``'stable'``.
        """
        if len(entries) < 4:
            return "stable"
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        mid = len(sorted_entries) // 2
        avg_first = sum(e.rating for e in sorted_entries[:mid]) / mid
        avg_second = sum(e.rating for e in sorted_entries[mid:]) / (len(sorted_entries) - mid)
        delta = avg_second - avg_first
        if delta > 0.3:
            return "improving"
        elif delta < -0.3:
            return "declining"
        return "stable"


# --------------------------------------------------------------------------- #
#  ModelRetrainer                                                              #
# --------------------------------------------------------------------------- #


class ModelRetrainer:
    """Drives model improvement cycles using collected user feedback.

    Works with a :class:`FeedbackCollector` to decide *when* to retrain,
    *what data* to use, and *whether* the new model is better.

    Args:
        feedback_collector:         An initialised FeedbackCollector.
        min_corrections_for_retrain: Min new corrections before retrain (default 20).
        improvement_threshold:      Min metric delta to recommend deployment.
    """

    MODEL_FIELD_MAP: dict[str, list[str]] = {
        "spot_detector": ["spot_detection", "spots"],
        "disease_risk_model": ["disease_risk", "risk"],
        "diet_analyzer": ["diet", "nutrition"],
        "facial_model": ["facial", "face"],
    }

    def __init__(
        self,
        feedback_collector: FeedbackCollector,
        min_corrections_for_retrain: int = 20,
        improvement_threshold: float = 0.02,
    ) -> None:
        self.collector = feedback_collector
        self.min_corrections = min_corrections_for_retrain
        self.improvement_threshold = improvement_threshold
        self._retrain_log: dict[str, str] = {}  # model -> last retrain ISO timestamp
        logger.info(
            "ModelRetrainer initialised (min_corrections=%d, threshold=%.3f).",
            self.min_corrections,
            self.improvement_threshold,
        )

    def should_retrain(self, model_name: str) -> bool:
        """Check if *model_name* has enough new corrections to warrant retraining.

        Returns ``True`` when the count of relevant corrections (since the last
        retrain, if any) meets or exceeds *min_corrections_for_retrain*.
        """
        prefixes = self._prefixes_for_model(model_name)
        last_retrain = self._retrain_log.get(model_name)
        since = datetime.fromisoformat(last_retrain) if last_retrain else None

        corrections = self.collector._load_corrections(since=since)
        relevant = [c for c in corrections if c.field.split(".")[0].lower() in prefixes]

        needed = len(relevant) >= self.min_corrections
        logger.info(
            "should_retrain(%s): %d relevant corrections (need %d) -> %s",
            model_name,
            len(relevant),
            self.min_corrections,
            needed,
        )
        return needed

    def prepare_training_batch(self, model_name: str) -> TrainingBatch:
        """Prepare a :class:`TrainingBatch` from corrections relevant to *model_name*.

        Corrections recorded since the last retrain are packaged into parallel
        ``samples`` and ``labels`` lists.  The retrain timestamp is updated so
        subsequent calls only see newer data.
        """
        prefixes = self._prefixes_for_model(model_name)
        last_retrain = self._retrain_log.get(model_name)
        since = datetime.fromisoformat(last_retrain) if last_retrain else None

        corrections = self.collector._load_corrections(since=since)
        relevant = [c for c in corrections if c.field.split(".")[0].lower() in prefixes]

        samples: list[dict[str, Any]] = []
        labels: list[Any] = []
        for corr in relevant:
            samples.append(
                {
                    "job_id": corr.job_id,
                    "field": corr.field,
                    "original_value": corr.original_value,
                    "reason": corr.reason,
                }
            )
            labels.append(corr.corrected_value)

        batch = TrainingBatch(
            batch_id=uuid.uuid4().hex,
            model_name=model_name,
            samples=samples,
            labels=labels,
            metadata={
                "created_at": datetime.now(UTC).isoformat(),
                "correction_count": len(relevant),
                "since": last_retrain or "epoch",
            },
        )
        self._retrain_log[model_name] = datetime.now(UTC).isoformat()
        logger.info(
            "Prepared training batch %s for '%s' with %d samples.",
            batch.batch_id,
            model_name,
            len(batch),
        )
        return batch

    def evaluate_improvement(
        self,
        model_name: str,
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
    ) -> ImprovementReport:
        """Compare metrics before and after retraining to produce a report.

        For every metric present in *both* dicts the delta is computed.
        Recommendation logic:
        * ``'deploy'``   — at least one metric improved by > *threshold* and
          none degraded by > 2× threshold.
        * ``'rollback'`` — any metric degraded significantly.
        * ``'neutral'``  — otherwise.
        """
        metric_changes: dict[str, dict[str, float]] = {}
        any_improved = False
        any_degraded = False

        for key in sorted(set(before_metrics) & set(after_metrics)):
            before = before_metrics[key]
            after = after_metrics[key]
            delta = round(after - before, 6)
            metric_changes[key] = {"before": before, "after": after, "delta": delta}
            if delta > self.improvement_threshold:
                any_improved = True
            if delta < -self.improvement_threshold * 2:
                any_degraded = True

        if any_degraded:
            recommendation = "rollback"
        elif any_improved:
            recommendation = "deploy"
        else:
            recommendation = "neutral"

        report = ImprovementReport(
            report_id=uuid.uuid4().hex,
            model_name=model_name,
            metric_changes=metric_changes,
            recommendation=recommendation,
        )
        logger.info("Improvement report for '%s': recommendation=%s", model_name, recommendation)
        return report

    # -- Internal helpers -------------------------------------------------- #

    def _prefixes_for_model(self, model_name: str) -> set:
        """Return the set of field prefixes associated with *model_name*.

        Falls back to the model name itself (lower-cased, stripped of
        ``_model`` / ``_``) if no explicit mapping is registered.
        """
        prefixes = self.MODEL_FIELD_MAP.get(model_name)
        if prefixes:
            return {p.lower() for p in prefixes}
        return {model_name.lower().replace("_model", "").replace("_", "")}
