"""Central orchestrator agent for the teloscopy multi-agent system.

The :class:`OrchestratorAgent` is the single entry-point for external
callers.  It manages an agent registry, routes messages between specialist
agents, executes pre-defined workflows (image analysis, disease-risk
prediction, diet planning), and implements retry logic with error recovery.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from .base import AgentMessage, BaseAgent, _MessageRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default retry / timeout settings
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 3
_RETRY_DELAY: float = 1.0  # seconds
_STEP_TIMEOUT: float = 300.0  # seconds per workflow step


# ---------------------------------------------------------------------------
# Orchestrator agent
# ---------------------------------------------------------------------------


class OrchestratorAgent(BaseAgent):
    """Coordinates all specialist agents, manages workflows, handles user requests.

    The orchestrator maintains a registry of specialist agents, routes
    messages between them, and exposes high-level workflow methods that
    chain multiple agent interactions into end-to-end analyses.

    Parameters
    ----------
    name : str
        Agent name (default ``"orchestrator"``).
    """

    def __init__(self, name: str = "orchestrator") -> None:
        super().__init__(name=name, capabilities=["orchestration", "routing", "workflow"])
        self._router = _MessageRouter()
        self._router.register(self)
        self._agents: dict[str, BaseAgent] = {}
        self._workflows: dict[str, dict[str, Any]] = {}
        self._pending_responses: dict[str, asyncio.Future[AgentMessage]] = {}
        self._session_results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Agent registry
    # ------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialist agent so it can participate in workflows.

        Parameters
        ----------
        agent : BaseAgent
            The agent to register.  Its :attr:`name` must be unique within
            this orchestrator's scope.

        Raises
        ------
        ValueError
            If an agent with the same name is already registered.
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered.")
        self._agents[agent.name] = agent
        self._router.register(agent)
        logger.info("Registered agent '%s' with capabilities %s", agent.name, agent.capabilities)

    def get_registered_agents(self) -> dict[str, dict[str, Any]]:
        """Return status dicts for every registered agent.

        Returns
        -------
        dict[str, dict]
            Mapping of agent name → :meth:`BaseAgent.get_status` dict.
        """
        return {name: agent.get_status() for name, agent in self._agents.items()}

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def handle_message(self, msg: AgentMessage) -> None:
        """Process messages directed at the orchestrator.

        Responses with a ``correlation_id`` present in
        :attr:`_pending_responses` are resolved so the corresponding
        workflow step can proceed.

        Parameters
        ----------
        msg : AgentMessage
            Incoming message.
        """
        cid = msg.correlation_id
        if cid in self._pending_responses:
            future = self._pending_responses.pop(cid)
            if not future.done():
                future.set_result(msg)
            return

        # Generic request handling
        action = msg.content.get("action")
        if action == "status":
            await self.send_message(
                recipient=msg.sender,
                content={"agents": self.get_registered_agents()},
                msg_type="response",
                correlation_id=cid,
            )
        else:
            logger.warning(
                "Orchestrator received unhandled message from '%s': %s",
                msg.sender,
                msg.content,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request_and_wait(
        self,
        recipient: str,
        content: dict[str, Any],
        timeout: float = _STEP_TIMEOUT,
    ) -> AgentMessage:
        """Send a request to *recipient* and await its response.

        Parameters
        ----------
        recipient : str
            Target agent name.
        content : dict
            Request payload.
        timeout : float
            Maximum seconds to wait for a response.

        Returns
        -------
        AgentMessage
            The response message.

        Raises
        ------
        TimeoutError
            If no response is received within *timeout* seconds.
        KeyError
            If *recipient* is not registered.
        """
        if recipient not in self._agents:
            raise KeyError(f"Agent '{recipient}' is not registered.")

        loop = asyncio.get_running_loop()
        cid = uuid.uuid4().hex
        future: asyncio.Future[AgentMessage] = loop.create_future()
        self._pending_responses[cid] = future

        await self.send_message(
            recipient=recipient,
            content=content,
            msg_type="request",
            correlation_id=cid,
        )

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_responses.pop(cid, None)
            raise TimeoutError(f"Agent '{recipient}' did not respond within {timeout}s.")

    async def _request_with_retry(
        self,
        recipient: str,
        content: dict[str, Any],
        max_retries: int = _MAX_RETRIES,
        retry_delay: float = _RETRY_DELAY,
        timeout: float = _STEP_TIMEOUT,
    ) -> AgentMessage:
        """Send a request with automatic retry on failure.

        Parameters
        ----------
        recipient : str
            Target agent name.
        content : dict
            Request payload.
        max_retries : int
            Number of retry attempts before raising.
        retry_delay : float
            Delay between retries (seconds).
        timeout : float
            Per-attempt timeout (seconds).

        Returns
        -------
        AgentMessage
            The successful response message.

        Raises
        ------
        RuntimeError
            If all retry attempts are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await self._request_and_wait(recipient, content, timeout=timeout)
                if response.message_type == "error":
                    raise RuntimeError(f"Agent '{recipient}' returned error: {response.content}")
                return response
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d to '%s' failed: %s",
                    attempt,
                    max_retries,
                    recipient,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)

        raise RuntimeError(
            f"All {max_retries} attempts to '{recipient}' failed. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    async def process_image_workflow(
        self,
        image_path: str,
        user_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the image-analysis workflow.

        Steps
        -----
        1. Send the image to the ``image_analysis`` agent.
        2. Optionally send results to the ``improvement`` agent for quality
           evaluation and parameter-tuning suggestions.

        Parameters
        ----------
        image_path : str
            Path to the qFISH microscopy image.
        user_profile : dict | None
            Optional user / sample metadata.

        Returns
        -------
        dict
            Workflow results containing analysis output, quality metrics,
            and improvement suggestions.
        """
        workflow_id = uuid.uuid4().hex
        self._workflows[workflow_id] = {
            "type": "image_analysis",
            "status": "running",
            "started_at": time.time(),
            "steps_completed": [],
        }

        results: dict[str, Any] = {"workflow_id": workflow_id}

        try:
            # Step 1 – image analysis
            analysis_response = await self._request_with_retry(
                recipient="image_analysis",
                content={
                    "action": "analyze_image",
                    "image_path": image_path,
                    "config": (user_profile or {}).get("analysis_config", {}),
                },
            )
            results["image_analysis"] = analysis_response.content
            self._workflows[workflow_id]["steps_completed"].append("image_analysis")

            # Step 2 – quality evaluation (optional)
            if "improvement" in self._agents:
                try:
                    quality_response = await self._request_with_retry(
                        recipient="improvement",
                        content={
                            "action": "evaluate_quality",
                            "results": [analysis_response.content],
                        },
                    )
                    results["quality"] = quality_response.content
                    self._workflows[workflow_id]["steps_completed"].append("quality_evaluation")
                except Exception:
                    logger.warning("Quality evaluation step failed; skipping.")

            self._workflows[workflow_id]["status"] = "completed"

        except Exception as exc:
            self._workflows[workflow_id]["status"] = "failed"
            self._workflows[workflow_id]["error"] = str(exc)
            results["error"] = str(exc)
            logger.exception("Image workflow %s failed.", workflow_id)

        self._workflows[workflow_id]["finished_at"] = time.time()
        self._session_results[workflow_id] = results
        return results

    async def process_full_analysis(
        self,
        image_path: str,
        user_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the complete end-to-end analysis workflow.

        Steps
        -----
        1. **Image analysis** — preprocessing, segmentation, spot detection,
           quantification.
        2. **Genomics risk assessment** — integrate telomere data with
           genetic variants and compute disease-risk scores.
        3. **Nutrition planning** — generate a personalised diet plan based
           on genetic risks and geographic region.
        4. **Report generation** — compile everything into a structured
           report.
        5. **Continuous improvement** — evaluate overall pipeline quality.

        Parameters
        ----------
        image_path : str
            Path to the qFISH microscopy image.
        user_profile : dict | None
            User metadata including ``region``, ``age``, ``sex``,
            ``variants``, ``dietary_restrictions``, etc.

        Returns
        -------
        dict
            Comprehensive results from every workflow step.
        """
        workflow_id = uuid.uuid4().hex
        profile = user_profile or {}
        self._workflows[workflow_id] = {
            "type": "full_analysis",
            "status": "running",
            "started_at": time.time(),
            "steps_completed": [],
        }

        results: dict[str, Any] = {"workflow_id": workflow_id}

        try:
            # Step 1 – Image analysis
            img_response = await self._request_with_retry(
                recipient="image_analysis",
                content={
                    "action": "analyze_image",
                    "image_path": image_path,
                    "config": profile.get("analysis_config", {}),
                },
            )
            results["image_analysis"] = img_response.content
            self._workflows[workflow_id]["steps_completed"].append("image_analysis")

            # Step 2 – Genomics risk assessment
            if "genomics" in self._agents:
                genomics_response = await self._request_with_retry(
                    recipient="genomics",
                    content={
                        "action": "assess_risk",
                        "telomere_data": img_response.content,
                        "variants": profile.get("variants", {}),
                        "profile": profile,
                    },
                )
                results["genomics"] = genomics_response.content
                self._workflows[workflow_id]["steps_completed"].append("genomics")
            else:
                logger.warning("Genomics agent not registered; skipping risk assessment.")

            # Step 3 – Nutrition planning
            if "nutrition" in self._agents:
                nutrition_response = await self._request_with_retry(
                    recipient="nutrition",
                    content={
                        "action": "generate_diet_plan",
                        "genetic_risks": results.get("genomics", {}).get("risks", []),
                        "region": profile.get("region", "global"),
                        "profile": profile,
                    },
                )
                results["nutrition"] = nutrition_response.content
                self._workflows[workflow_id]["steps_completed"].append("nutrition")
            else:
                logger.warning("Nutrition agent not registered; skipping diet planning.")

            # Step 4 – Report generation
            if "report" in self._agents:
                report_response = await self._request_with_retry(
                    recipient="report",
                    content={
                        "action": "generate_full_report",
                        "analysis": results.get("image_analysis", {}),
                        "risks": results.get("genomics", {}),
                        "diet": results.get("nutrition", {}),
                        "profile": profile,
                    },
                )
                results["report"] = report_response.content
                self._workflows[workflow_id]["steps_completed"].append("report")
            else:
                logger.warning("Report agent not registered; skipping report generation.")

            # Step 5 – Continuous improvement
            if "improvement" in self._agents:
                try:
                    improvement_response = await self._request_with_retry(
                        recipient="improvement",
                        content={
                            "action": "evaluate_quality",
                            "results": [results.get("image_analysis", {})],
                        },
                    )
                    results["improvement"] = improvement_response.content
                    self._workflows[workflow_id]["steps_completed"].append("improvement")
                except Exception:
                    logger.warning("Improvement step failed; skipping.")

            self._workflows[workflow_id]["status"] = "completed"

        except Exception as exc:
            self._workflows[workflow_id]["status"] = "failed"
            self._workflows[workflow_id]["error"] = str(exc)
            results["error"] = str(exc)
            logger.exception("Full-analysis workflow %s failed.", workflow_id)

        self._workflows[workflow_id]["finished_at"] = time.time()
        self._session_results[workflow_id] = results
        return results

    # ------------------------------------------------------------------
    # Workflow status
    # ------------------------------------------------------------------

    def get_workflow_status(self, workflow_id: str) -> dict[str, Any]:
        """Query the status of a running or completed workflow.

        Parameters
        ----------
        workflow_id : str
            The identifier returned as ``workflow_id`` by a workflow method.

        Returns
        -------
        dict
            Workflow metadata: ``type``, ``status``, timestamps, completed
            steps, and any error information.

        Raises
        ------
        KeyError
            If *workflow_id* does not exist.
        """
        if workflow_id not in self._workflows:
            raise KeyError(f"Unknown workflow '{workflow_id}'.")
        return dict(self._workflows[workflow_id])

    # ------------------------------------------------------------------
    # Continuous improvement loop
    # ------------------------------------------------------------------

    async def continuous_improvement_loop(
        self,
        interval: float = 60.0,
        max_iterations: int | None = None,
    ) -> None:
        """Periodically ask the improvement agent to evaluate recent results.

        This is intended to run as a background task.  It collects session
        results accumulated since the last check, sends them to the
        ``improvement`` agent, and logs the returned recommendations.

        Parameters
        ----------
        interval : float
            Seconds between evaluation cycles.
        max_iterations : int | None
            If set, stop after this many cycles (useful for testing).
        """
        if "improvement" not in self._agents:
            logger.warning("No improvement agent registered; loop will not run.")
            return

        iteration = 0
        evaluated_ids: set[str] = set()

        while self._running:
            await asyncio.sleep(interval)
            iteration += 1

            # Gather new results
            new_results = [v for k, v in self._session_results.items() if k not in evaluated_ids]
            if not new_results:
                continue

            try:
                response = await self._request_with_retry(
                    recipient="improvement",
                    content={"action": "evaluate_quality", "results": new_results},
                )
                logger.info(
                    "Improvement cycle %d: %s",
                    iteration,
                    response.content.get("summary", "no summary"),
                )
                evaluated_ids.update(
                    r.get("workflow_id", "") for r in new_results if isinstance(r, dict)
                )
            except Exception:
                logger.exception("Improvement cycle %d failed.", iteration)

            if max_iterations is not None and iteration >= max_iterations:
                break
