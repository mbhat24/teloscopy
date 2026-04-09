"""Image-analysis agent for qFISH telomere microscopy images.

The :class:`ImageAnalysisAgent` encapsulates the entire image-processing
pipeline — preprocessing, chromosome segmentation, telomere-spot detection,
and intensity quantification — behind an async message-driven interface so
that it can participate in the multi-agent orchestration system.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import AgentMessage, AgentState, BaseAgent

logger = logging.getLogger(__name__)


class ImageAnalysisAgent(BaseAgent):
    """Handles all image processing: preprocessing, segmentation, spot detection, quantification.

    The agent listens for incoming ``"request"`` messages whose ``content``
    dict carries an ``"action"`` key.  Supported actions:

    * ``analyze_image`` — run the full pipeline on a single image.
    * ``preprocess`` — background subtraction + denoising only.
    * ``segment`` — chromosome segmentation only.
    * ``detect_spots`` — telomere spot detection only.
    * ``quantify`` — intensity quantification only.
    * ``validate`` — quality-check a set of results.
    * ``suggest_improvements`` — propose parameter tweaks.

    Parameters
    ----------
    name : str
        Agent name (default ``"image_analysis"``).
    default_config : dict | None
        Pipeline parameters forwarded to the underlying analysis functions.
    """

    def __init__(
        self,
        name: str = "image_analysis",
        default_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            capabilities=[
                "image_analysis",
                "preprocessing",
                "segmentation",
                "spot_detection",
                "quantification",
            ],
        )
        self._default_config: dict[str, Any] = default_config or {}

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def handle_message(self, msg: AgentMessage) -> None:
        """Route an incoming request to the appropriate handler.

        Parameters
        ----------
        msg : AgentMessage
            Incoming message.  ``msg.content["action"]`` determines which
            handler is invoked.
        """
        action = msg.content.get("action", "")
        handlers: dict[str, Any] = {
            "analyze_image": self._handle_analyze,
            "preprocess": self._handle_preprocess,
            "segment": self._handle_segment,
            "detect_spots": self._handle_detect,
            "quantify": self._handle_quantify,
            "validate": self._handle_validate,
            "suggest_improvements": self._handle_suggest,
        }

        handler = handlers.get(action)
        if handler is None:
            await self.send_message(
                recipient=msg.sender,
                content={"error": f"Unknown action '{action}'."},
                msg_type="error",
                correlation_id=msg.correlation_id,
            )
            return

        self.state = AgentState.RUNNING
        try:
            result = handler(msg.content)
            await self.send_message(
                recipient=msg.sender,
                content=result,
                msg_type="response",
                correlation_id=msg.correlation_id,
            )
        except Exception as exc:
            logger.exception("ImageAnalysisAgent action '%s' failed.", action)
            await self.send_message(
                recipient=msg.sender,
                content={"error": str(exc), "action": action},
                msg_type="error",
                correlation_id=msg.correlation_id,
            )
        finally:
            self.state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Public analysis API (synchronous, called from handlers)
    # ------------------------------------------------------------------

    def analyze_image(
        self,
        image_path: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the full analysis pipeline on a single image.

        Delegates to :func:`teloscopy.telomere.pipeline.analyze_image` while
        merging the agent's default config with any per-call overrides.

        Parameters
        ----------
        image_path : str
            Path to a multi-channel TIFF image.
        config : dict | None
            Per-call parameter overrides.

        Returns
        -------
        dict
            Complete analysis results including spots, chromosomes,
            statistics, and association summary.
        """
        from ..telomere.pipeline import analyze_image as _pipeline_analyze

        merged_config = {**self._default_config, **(config or {})}
        result = _pipeline_analyze(image_path, config=merged_config)

        # Strip non-serialisable numpy arrays for messaging
        serialisable = {k: v for k, v in result.items() if k not in ("channels", "labels")}
        serialisable["n_chromosomes"] = (
            int(result.get("labels", np.empty(0)).max()) if result.get("labels") is not None else 0
        )
        return serialisable

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply background subtraction and denoising.

        Parameters
        ----------
        image : np.ndarray
            Single-channel fluorescence image.

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        from ..telomere.pipeline import _preprocess_channel

        return _preprocess_channel(image, self._default_config)

    def segment_chromosomes(self, image: np.ndarray) -> np.ndarray:
        """Segment chromosomes from a DAPI image.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed DAPI channel image.

        Returns
        -------
        np.ndarray
            Integer-labelled segmentation mask.
        """
        from ..telomere.segmentation import segment

        method = self._default_config.get("segmentation_method", "otsu_watershed")
        return segment(image, method=method)

    def detect_spots(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect telomere spots in a Cy3 channel image.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed Cy3 channel image.

        Returns
        -------
        list[dict]
            Detected spots with positions, sigma, and peak intensity.
        """
        from ..telomere.pipeline import _detect_spots

        return _detect_spots(image, self._default_config)

    def quantify(
        self,
        spots: list[dict[str, Any]],
        image: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Quantify spot intensities with background correction.

        Parameters
        ----------
        spots : list[dict]
            Detected spot dictionaries.
        image : np.ndarray
            Original (un-preprocessed) Cy3 channel.

        Returns
        -------
        list[dict]
            Spots with ``corrected_intensity``, ``background_level``, and
            ``snr`` populated.
        """
        from ..telomere.pipeline import _quantify_spots

        return _quantify_spots(spots, image, self._default_config)

    def validate_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Run quality checks on analysis results.

        Checks include: minimum spot count, association rate, SNR
        distribution, and coefficient of variation.

        Parameters
        ----------
        results : dict
            Output from :meth:`analyze_image`.

        Returns
        -------
        dict
            Validation report with boolean ``passed`` flag and a list of
            ``warnings``.
        """
        warnings_list: list[str] = []

        stats = results.get("statistics", {})
        assoc = results.get("association_summary", {})

        # Check minimum telomere count
        n_telomeres = stats.get("n_telomeres", 0)
        if n_telomeres < 10:
            warnings_list.append(f"Low telomere count ({n_telomeres}); results may be unreliable.")

        # Check association rate
        assoc_rate = assoc.get("association_rate", 0.0)
        if assoc_rate < 0.5:
            warnings_list.append(
                f"Low association rate ({assoc_rate:.1%}); check segmentation parameters."
            )

        # Check coefficient of variation
        cv = stats.get("cv", 0.0)
        if cv > 1.0:
            warnings_list.append(f"High intensity CV ({cv:.2f}); sample may be heterogeneous.")

        # Check SNR
        spots = results.get("spots", [])
        if spots:
            snr_values = [s.get("snr", 0.0) for s in spots if s.get("valid", True)]
            if snr_values:
                median_snr = float(np.median(snr_values))
                if median_snr < 5.0:
                    warnings_list.append(
                        f"Low median SNR ({median_snr:.1f}); consider increasing exposure."
                    )

        return {
            "passed": len(warnings_list) == 0,
            "n_warnings": len(warnings_list),
            "warnings": warnings_list,
            "n_telomeres": n_telomeres,
            "association_rate": assoc_rate,
            "cv": cv,
        }

    def suggest_improvements(self, results: dict[str, Any]) -> list[str]:
        """Suggest parameter tweaks based on analysis results.

        Parameters
        ----------
        results : dict
            Output from :meth:`analyze_image`.

        Returns
        -------
        list[str]
            Human-readable suggestions for improving pipeline parameters.
        """
        suggestions: list[str] = []
        assoc = results.get("association_summary", {})
        stats = results.get("statistics", {})

        # Too few spots detected
        total = assoc.get("total_spots", 0)
        if total < 20:
            suggestions.append(
                "Very few spots detected. Try lowering 'spot_threshold' "
                "(e.g. 0.01) or reducing 'spot_min_snr'."
            )

        # Low association rate → segmentation may be off
        if assoc.get("association_rate", 1.0) < 0.5:
            suggestions.append(
                "Low association rate. Consider increasing 'max_tip_distance' "
                "or adjusting 'min_chromosome_area'/'max_chromosome_area'."
            )

        # Many invalid spots → SNR issue
        n_invalid = assoc.get("invalid", 0)
        if total > 0 and n_invalid / total > 0.4:
            suggestions.append(
                "Many spots failing SNR check. Try lowering 'spot_min_snr' "
                "or increasing 'denoise_sigma'."
            )

        # High CV → possibly mixing signal with artefacts
        cv = stats.get("cv", 0.0)
        if cv > 0.8:
            suggestions.append(
                "High coefficient of variation. Consider tightening "
                "'spot_sigma_min'/'spot_sigma_max' to reduce outliers."
            )

        if not suggestions:
            suggestions.append("Results look good — no parameter changes recommended.")

        return suggestions

    # ------------------------------------------------------------------
    # Internal message handlers
    # ------------------------------------------------------------------

    def _handle_analyze(self, content: dict[str, Any]) -> dict[str, Any]:
        image_path = content["image_path"]
        config = content.get("config")
        result = self.analyze_image(image_path, config)
        validation = self.validate_results(result)
        suggestions = self.suggest_improvements(result)
        result["validation"] = validation
        result["suggestions"] = suggestions
        return result

    def _handle_preprocess(self, content: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(content["image"], dtype=np.float64)
        processed = self.preprocess(image)
        return {"processed_shape": list(processed.shape)}

    def _handle_segment(self, content: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(content["image"], dtype=np.float64)
        labels = self.segment_chromosomes(image)
        return {"n_chromosomes": int(labels.max()), "labels_shape": list(labels.shape)}

    def _handle_detect(self, content: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(content["image"], dtype=np.float64)
        spots = self.detect_spots(image)
        return {"n_spots": len(spots), "spots": spots}

    def _handle_quantify(self, content: dict[str, Any]) -> dict[str, Any]:
        spots = content["spots"]
        image = np.asarray(content["image"], dtype=np.float64)
        quantified = self.quantify(spots, image)
        return {"spots": quantified}

    def _handle_validate(self, content: dict[str, Any]) -> dict[str, Any]:
        return self.validate_results(content.get("results", content))

    def _handle_suggest(self, content: dict[str, Any]) -> dict[str, Any]:
        suggestions = self.suggest_improvements(content.get("results", content))
        return {"suggestions": suggestions}
