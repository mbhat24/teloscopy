"""Tests for the multi-agent orchestration system."""

from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from teloscopy.agents.base import AgentMessage, AgentState, BaseAgent, _MessageRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously via asyncio.run()."""
    return asyncio.run(coro)


def _make_dummy(name: str = "dummy"):
    """Create a _DummyAgent (no event loop needed thanks to lazy Queue init)."""
    return _DummyAgent(name)


class _DummyAgent(BaseAgent):
    """Minimal concrete subclass for testing BaseAgent."""

    def __init__(self, name: str = "dummy") -> None:
        super().__init__(name=name, capabilities=["testing"])
        self.received_messages: list[AgentMessage] = []

    async def handle_message(self, msg: AgentMessage) -> None:
        """Store received messages and echo a response."""
        self.received_messages.append(msg)
        await self.send_message(
            recipient=msg.sender,
            content={"echo": msg.content},
            msg_type="response",
            correlation_id=msg.correlation_id,
        )


# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------


class TestAgentMessage:
    """Tests for the AgentMessage dataclass."""

    def test_creation_with_all_fields(self):
        """Create a message with all fields explicitly set."""
        msg = AgentMessage(
            sender="agent_a",
            recipient="agent_b",
            content={"action": "test"},
            message_type="request",
            timestamp=1234567890.0,
            correlation_id="abc123",
        )
        assert msg.sender == "agent_a"
        assert msg.recipient == "agent_b"
        assert msg.content == {"action": "test"}
        assert msg.message_type == "request"
        assert msg.timestamp == 1234567890.0
        assert msg.correlation_id == "abc123"

    def test_creation_defaults(self):
        """Fields timestamp and correlation_id should auto-populate."""
        msg = AgentMessage(
            sender="a",
            recipient="b",
            content={},
            message_type="event",
        )
        assert msg.timestamp > 0
        assert len(msg.correlation_id) > 0

    def test_valid_message_types(self):
        """All four valid types should be accepted."""
        for mtype in ("request", "response", "event", "error"):
            msg = AgentMessage(sender="a", recipient="b", content={}, message_type=mtype)
            assert msg.message_type == mtype

    def test_invalid_message_type_raises(self):
        """An invalid message_type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid message_type"):
            AgentMessage(sender="a", recipient="b", content={}, message_type="invalid")


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class TestAgentState:
    """Tests for the AgentState enum."""

    def test_enum_values(self):
        """All expected states should exist."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.WAITING.value == "waiting"
        assert AgentState.ERROR.value == "error"
        assert AgentState.COMPLETED.value == "completed"

    def test_enum_count(self):
        """There should be exactly five agent states."""
        assert len(AgentState) == 5


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class TestBaseAgent:
    """Tests for BaseAgent via the _DummyAgent subclass."""

    def test_instantiation(self):
        """A concrete subclass should be instantiable."""

        async def _test():
            agent = _DummyAgent("test_agent")
            assert agent.name == "test_agent"
            assert agent.state == AgentState.IDLE
            assert "testing" in agent.capabilities

        _run(_test())

    def test_initial_state_idle(self):
        """New agents should start in the IDLE state."""

        async def _test():
            agent = _DummyAgent()
            assert agent.state == AgentState.IDLE

        _run(_test())

    def test_get_status(self):
        """get_status() should return name, state, queue_size, capabilities."""

        async def _test():
            agent = _DummyAgent("status_agent")
            status = agent.get_status()
            assert status["name"] == "status_agent"
            assert status["state"] == "idle"
            assert status["queue_size"] == 0
            assert "testing" in status["capabilities"]

        _run(_test())

    def test_send_message_without_router(self):
        """Sending a message with no router should still return the message."""

        async def _test():
            agent = _DummyAgent()
            msg = await agent.send_message(
                recipient="other",
                content={"test": True},
                msg_type="request",
            )
            assert msg.sender == "dummy"
            assert msg.recipient == "other"

        _run(_test())

    def test_run_and_stop(self):
        """Agent should transition IDLE → RUNNING → COMPLETED on run/stop."""

        async def _test():
            agent = _DummyAgent()
            assert agent.state == AgentState.IDLE
            task = asyncio.create_task(agent.run())
            await asyncio.sleep(0.1)
            assert agent.state == AgentState.RUNNING
            await agent.stop()
            await asyncio.wait_for(task, timeout=3.0)
            assert agent.state == AgentState.COMPLETED

        _run(_test())

    def test_handle_message_on_queue(self):
        """Messages placed on the queue should be dispatched to handle_message."""

        async def _test():
            agent = _DummyAgent("handler_test")
            task = asyncio.create_task(agent.run())
            await asyncio.sleep(0.1)

            msg = AgentMessage(
                sender="test_sender",
                recipient="handler_test",
                content={"hello": "world"},
                message_type="request",
            )
            await agent.message_queue.put(msg)
            await asyncio.sleep(0.3)

            assert len(agent.received_messages) == 1
            assert agent.received_messages[0].content == {"hello": "world"}

            await agent.stop()
            await asyncio.wait_for(task, timeout=3.0)

        _run(_test())


# ---------------------------------------------------------------------------
# _MessageRouter
# ---------------------------------------------------------------------------


class TestMessageRouter:
    """Tests for the internal _MessageRouter."""

    def test_register_and_route(self):
        """Routing a message to a registered agent should place it on its queue."""

        async def _test():
            router = _MessageRouter()
            agent = _DummyAgent("target")
            router.register(agent)

            msg = AgentMessage(
                sender="sender",
                recipient="target",
                content={"data": 42},
                message_type="request",
            )
            await router.route(msg)
            assert agent.message_queue.qsize() == 1

        _run(_test())

    def test_route_unknown_raises(self):
        """Routing to an unregistered agent should raise KeyError."""

        async def _test():
            router = _MessageRouter()
            msg = AgentMessage(sender="a", recipient="unknown", content={}, message_type="request")
            with pytest.raises(KeyError, match="not registered"):
                await router.route(msg)

        _run(_test())

    def test_unregister(self):
        """After unregister, routing to the agent should fail."""

        async def _test():
            router = _MessageRouter()
            agent = _DummyAgent("ephemeral")
            router.register(agent)
            router.unregister("ephemeral")

            msg = AgentMessage(
                sender="a", recipient="ephemeral", content={}, message_type="request"
            )
            with pytest.raises(KeyError):
                await router.route(msg)

        _run(_test())


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------


class TestOrchestratorAgent:
    """Tests for the OrchestratorAgent."""

    def test_instantiation(self):
        """Orchestrator should initialise with default name and capabilities."""

        async def _test():
            from teloscopy.agents.orchestrator import OrchestratorAgent

            orch = OrchestratorAgent()
            assert orch.name == "orchestrator"
            assert "orchestration" in orch.capabilities

        _run(_test())

    def test_register_agent(self):
        """Registering an agent should make it queryable."""

        async def _test():
            from teloscopy.agents.orchestrator import OrchestratorAgent

            orch = OrchestratorAgent()
            dummy = _DummyAgent("test_worker")
            orch.register_agent(dummy)
            agents = orch.get_registered_agents()
            assert "test_worker" in agents
            assert agents["test_worker"]["state"] == "idle"

        _run(_test())

    def test_register_duplicate_raises(self):
        """Registering the same name twice should raise ValueError."""

        async def _test():
            from teloscopy.agents.orchestrator import OrchestratorAgent

            orch = OrchestratorAgent()
            orch.register_agent(_DummyAgent("worker"))
            with pytest.raises(ValueError, match="already registered"):
                orch.register_agent(_DummyAgent("worker"))

        _run(_test())

    def test_get_registered_agents_empty(self):
        """Before registration, get_registered_agents should be empty."""

        async def _test():
            from teloscopy.agents.orchestrator import OrchestratorAgent

            orch = OrchestratorAgent()
            assert orch.get_registered_agents() == {}

        _run(_test())

    def test_handle_status_request(self):
        """Sending a status action should produce a response with agent info."""

        async def _test():
            from teloscopy.agents.orchestrator import OrchestratorAgent

            orch = OrchestratorAgent()
            dummy = _DummyAgent("probe")
            orch.register_agent(dummy)

            task = asyncio.create_task(orch.run())
            await asyncio.sleep(0.1)

            msg = AgentMessage(
                sender="probe",
                recipient="orchestrator",
                content={"action": "status"},
                message_type="request",
            )
            await orch.message_queue.put(msg)
            await asyncio.sleep(0.5)

            # The response should be on the probe's queue
            assert dummy.message_queue.qsize() >= 1

            await orch.stop()
            await asyncio.wait_for(task, timeout=3.0)

        _run(_test())


# ---------------------------------------------------------------------------
# ImageAnalysisAgent
# ---------------------------------------------------------------------------


class TestImageAnalysisAgent:
    """Tests for the ImageAnalysisAgent."""

    def test_instantiation(self):
        """Should initialise with default name and image-related capabilities."""

        async def _test():
            from teloscopy.agents.image_agent import ImageAnalysisAgent

            agent = ImageAnalysisAgent()
            assert agent.name == "image_analysis"
            assert "image_analysis" in agent.capabilities
            assert "segmentation" in agent.capabilities

        _run(_test())

    def test_handle_unknown_action(self):
        """An unknown action should result in an error message sent back."""

        async def _test():
            from teloscopy.agents.image_agent import ImageAnalysisAgent

            agent = ImageAnalysisAgent()
            router = _MessageRouter()
            receiver = _DummyAgent("caller")
            router.register(agent)
            router.register(receiver)

            msg = AgentMessage(
                sender="caller",
                recipient="image_analysis",
                content={"action": "nonexistent"},
                message_type="request",
            )
            await agent.handle_message(msg)

            assert receiver.message_queue.qsize() >= 1
            resp = await receiver.message_queue.get()
            assert resp.message_type == "error"

        _run(_test())

    def test_validate_results(self):
        """validate_results should return a dict with 'passed' flag."""

        async def _test():
            from teloscopy.agents.image_agent import ImageAnalysisAgent

            agent = ImageAnalysisAgent()
            result = agent.validate_results(
                {
                    "statistics": {"n_telomeres": 30, "cv": 0.3},
                    "association_summary": {"association_rate": 0.8},
                    "spots": [{"snr": 15.0, "valid": True}],
                }
            )
            assert result["passed"] is True
            assert result["n_warnings"] == 0

        _run(_test())

    def test_validate_results_low_spots(self):
        """Low telomere count should trigger a warning."""

        async def _test():
            from teloscopy.agents.image_agent import ImageAnalysisAgent

            agent = ImageAnalysisAgent()
            result = agent.validate_results(
                {
                    "statistics": {"n_telomeres": 3, "cv": 0.3},
                    "association_summary": {"association_rate": 0.8},
                    "spots": [],
                }
            )
            assert result["passed"] is False
            assert result["n_warnings"] > 0

        _run(_test())

    def test_suggest_improvements(self):
        """suggest_improvements should return a non-empty list of strings."""

        async def _test():
            from teloscopy.agents.image_agent import ImageAnalysisAgent

            agent = ImageAnalysisAgent()
            suggestions = agent.suggest_improvements(
                {
                    "statistics": {"cv": 1.2},
                    "association_summary": {
                        "total_spots": 5,
                        "association_rate": 0.3,
                        "invalid": 3,
                    },
                }
            )
            assert isinstance(suggestions, list)
            assert len(suggestions) > 0

        _run(_test())


# ---------------------------------------------------------------------------
# GenomicsAgent
# ---------------------------------------------------------------------------


class TestGenomicsAgent:
    """Tests for the GenomicsAgent."""

    def test_instantiation(self):
        """Should initialise with genomics capabilities."""

        async def _test():
            from teloscopy.agents.genomics_agent import GenomicsAgent

            agent = GenomicsAgent()
            assert agent.name == "genomics"
            assert "risk_assessment" in agent.capabilities

        _run(_test())

    def test_assess_risk_short_telomeres(self):
        """Short telomeres should produce elevated risk scores."""

        async def _test():
            from teloscopy.agents.genomics_agent import GenomicsAgent

            agent = GenomicsAgent()
            result = agent.assess_risk(
                telomere_data={"mean_length_bp": 3000.0},
                profile={"age": 50},
            )
            assert "risks" in result
            assert "overall_risk_score" in result
            assert result["overall_risk_score"] > 0.3

        _run(_test())

    def test_assess_risk_normal_telomeres(self):
        """Normal-length telomeres should produce low risk scores."""

        async def _test():
            from teloscopy.agents.genomics_agent import GenomicsAgent

            agent = GenomicsAgent()
            result = agent.assess_risk(
                telomere_data={"mean_length_bp": 8000.0},
                profile={"age": 40},
            )
            assert result["overall_risk_score"] < 0.5

        _run(_test())

    def test_assess_risk_with_snp_modifiers(self):
        """SNP risk alleles should increase the score beyond telomere-only."""

        async def _test():
            from teloscopy.agents.genomics_agent import GenomicsAgent

            agent = GenomicsAgent()
            base = agent.assess_risk(
                telomere_data={"mean_length_bp": 5000.0},
                profile={"age": 50},
            )
            modified = agent.assess_risk(
                telomere_data={"mean_length_bp": 5000.0},
                variants={"APOE": "e4", "TERT": "risk"},
                profile={"age": 50},
            )
            assert modified["overall_risk_score"] >= base["overall_risk_score"]

        _run(_test())

    def test_project_health_timeline(self):
        """project_health_timeline should return a timeline of length years+1."""

        async def _test():
            from teloscopy.agents.genomics_agent import GenomicsAgent

            agent = GenomicsAgent()
            risk_profile = agent.assess_risk(
                telomere_data={"mean_length_bp": 6000.0},
            )
            projection = agent.project_health_timeline(risk_profile, years=10)
            assert "timeline" in projection
            assert len(projection["timeline"]) == 11
            assert "summary" in projection

        _run(_test())

    def test_get_prevention_recommendations(self):
        """Recommendations should be ordered by risk score descending."""

        async def _test():
            from teloscopy.agents.genomics_agent import GenomicsAgent

            agent = GenomicsAgent()
            risks = [
                {"disease": "cardiovascular", "risk_score": 0.7},
                {"disease": "cancer", "risk_score": 0.3},
            ]
            recs = agent.get_prevention_recommendations(risks)
            assert len(recs) == 2
            assert recs[0]["priority"] == "high"
            assert recs[1]["priority"] == "moderate"

        _run(_test())

    def test_integrate_telomere_with_snp(self):
        """Integration should return telomere_summary and variant_summary."""

        async def _test():
            from teloscopy.agents.genomics_agent import GenomicsAgent

            agent = GenomicsAgent()
            result = agent.integrate_telomere_with_snp(
                telomere_results={"statistics": {"mean_intensity": 5000.0, "cv": 0.4}},
                snp_data={"APOE": "e4", "UNKNOWN_GENE": "normal"},
            )
            assert "telomere_summary" in result
            assert "variant_summary" in result
            assert result["variant_summary"]["relevant_variants"] == 1
            assert result["variant_summary"]["risk_allele_count"] == 1

        _run(_test())

    def test_handle_message_assess_risk(self):
        """The agent should process an 'assess_risk' message correctly."""

        async def _test():
            from teloscopy.agents.genomics_agent import GenomicsAgent

            agent = GenomicsAgent()
            router = _MessageRouter()
            receiver = _DummyAgent("caller")
            router.register(agent)
            router.register(receiver)

            msg = AgentMessage(
                sender="caller",
                recipient="genomics",
                content={
                    "action": "assess_risk",
                    "telomere_data": {"mean_length_bp": 4000.0},
                    "profile": {"age": 55},
                },
                message_type="request",
            )
            await agent.handle_message(msg)

            assert receiver.message_queue.qsize() >= 1
            resp = await receiver.message_queue.get()
            assert resp.message_type == "response"
            assert "risks" in resp.content

        _run(_test())


# ---------------------------------------------------------------------------
# NutritionAgent
# ---------------------------------------------------------------------------


class TestNutritionAgent:
    """Tests for the NutritionAgent."""

    def test_instantiation(self):
        """Should initialise with nutrition capabilities."""

        async def _test():
            from teloscopy.agents.nutrition_agent import NutritionAgent

            agent = NutritionAgent()
            assert agent.name == "nutrition"
            assert "diet_planning" in agent.capabilities

        _run(_test())

    def test_generate_diet_plan(self):
        """A diet plan should contain priority_nutrients, foods, and meal_plan."""

        async def _test():
            from teloscopy.agents.nutrition_agent import NutritionAgent

            agent = NutritionAgent()
            plan = agent.generate_diet_plan(
                genetic_risks=[
                    {"disease": "cardiovascular", "risk_score": 0.6},
                    {"disease": "cancer", "risk_score": 0.3},
                ],
                region="mediterranean",
            )
            assert "priority_nutrients" in plan
            assert "recommended_foods" in plan
            assert "meal_plan" in plan
            assert len(plan["priority_nutrients"]) > 0
            assert len(plan["recommended_foods"]) > 0

        _run(_test())

    def test_get_protective_foods_global(self):
        """Global region should always return foods."""

        async def _test():
            from teloscopy.agents.nutrition_agent import NutritionAgent

            agent = NutritionAgent()
            foods = agent.get_telomere_protective_foods("global")
            assert len(foods) >= 8

        _run(_test())

    def test_get_protective_foods_regional(self):
        """Regional foods should include both global and region-specific items."""

        async def _test():
            from teloscopy.agents.nutrition_agent import NutritionAgent

            agent = NutritionAgent()
            global_foods = agent.get_telomere_protective_foods("global")
            med_foods = agent.get_telomere_protective_foods("mediterranean")
            assert len(med_foods) > len(global_foods)

        _run(_test())

    def test_adapt_to_vegetarian(self):
        """Vegetarian restriction should remove fish items."""

        async def _test():
            from teloscopy.agents.nutrition_agent import NutritionAgent

            agent = NutritionAgent()
            plan = agent.generate_diet_plan(
                genetic_risks=[{"disease": "cardiovascular", "risk_score": 0.5}],
                region="global",
            )
            adapted = agent.adapt_to_preferences(plan, restrictions=["vegetarian"])
            assert "dietary_restrictions_applied" in adapted
            food_names = [f.get("name", "") for f in adapted.get("recommended_foods", [])]
            assert "Salmon" not in food_names

        _run(_test())

    def test_calculate_nutritional_gaps(self):
        """Gaps should be identified for nutrients with low/unknown intake."""

        async def _test():
            from teloscopy.agents.nutrition_agent import NutritionAgent

            agent = NutritionAgent()
            result = agent.calculate_nutritional_gaps(
                current_diet={"omega_3": "low", "fiber": "adequate"},
                genetic_needs=["omega_3", "fiber", "vitamin_d"],
            )
            assert "omega_3" in result["gaps"]
            assert "vitamin_d" in result["gaps"]
            assert "fiber" in result["adequate"]
            assert result["gap_count"] == 2

        _run(_test())


# ---------------------------------------------------------------------------
# ContinuousImprovementAgent
# ---------------------------------------------------------------------------


class TestContinuousImprovementAgent:
    """Tests for the ContinuousImprovementAgent."""

    def test_instantiation(self):
        """Should initialise with improvement capabilities."""

        async def _test():
            from teloscopy.agents.improvement_agent import ContinuousImprovementAgent

            agent = ContinuousImprovementAgent()
            assert agent.name == "improvement"
            assert "quality_evaluation" in agent.capabilities

        _run(_test())

    def test_evaluate_pipeline_quality_empty(self):
        """Empty results should return grade F."""

        async def _test():
            from teloscopy.agents.improvement_agent import ContinuousImprovementAgent

            agent = ContinuousImprovementAgent()
            result = agent.evaluate_pipeline_quality([])
            assert result["grade"] == "F"
            assert result["overall_quality"] == 0.0

        _run(_test())

    def test_evaluate_pipeline_quality_good(self):
        """Good results should produce a reasonable grade."""

        async def _test():
            from teloscopy.agents.improvement_agent import ContinuousImprovementAgent

            agent = ContinuousImprovementAgent()
            results = [
                {
                    "statistics": {"n_telomeres": 40, "cv": 0.3, "mean_intensity": 5000.0},
                    "association_summary": {"association_rate": 0.85, "total_spots": 50},
                    "spots": [{"snr": 15.0, "valid": True} for _ in range(40)],
                }
            ]
            quality = agent.evaluate_pipeline_quality(results)
            assert quality["grade"] in ("A", "B", "C")
            assert quality["overall_quality"] > 0.5

        _run(_test())

    def test_track_metrics(self):
        """track_metrics should accumulate entries."""

        async def _test():
            from teloscopy.agents.improvement_agent import ContinuousImprovementAgent

            agent = ContinuousImprovementAgent()
            agent.track_metrics(
                {
                    "statistics": {"n_telomeres": 30, "cv": 0.4, "mean_intensity": 4000.0},
                    "association_summary": {"association_rate": 0.7},
                    "image_path": "test.tif",
                }
            )
            assert len(agent._metrics_history) == 1

        _run(_test())

    def test_generate_improvement_report_empty(self):
        """Report with no tracked data should indicate insufficient data."""

        async def _test():
            from teloscopy.agents.improvement_agent import ContinuousImprovementAgent

            agent = ContinuousImprovementAgent()
            report = agent.generate_improvement_report()
            assert report["n_images_tracked"] == 0

        _run(_test())

    def test_generate_improvement_report_with_data(self):
        """Report with tracked data should include trends."""

        async def _test():
            from teloscopy.agents.improvement_agent import ContinuousImprovementAgent

            agent = ContinuousImprovementAgent()
            for i in range(10):
                agent.track_metrics(
                    {
                        "statistics": {
                            "n_telomeres": 30 + i,
                            "cv": 0.3 + i * 0.01,
                            "mean_intensity": 4000.0 + i * 100,
                        },
                        "association_summary": {"association_rate": 0.7 + i * 0.01},
                        "image_path": f"test_{i}.tif",
                    }
                )
            report = agent.generate_improvement_report()
            assert report["n_images_tracked"] == 10
            assert "trends" in report
            assert "recommendations" in report

        _run(_test())

    def test_suggest_parameter_tuning(self):
        """Suggestions should be returned for poor-quality results."""

        async def _test():
            from teloscopy.agents.improvement_agent import ContinuousImprovementAgent

            agent = ContinuousImprovementAgent()
            result = agent.suggest_parameter_tuning(
                [
                    {
                        "statistics": {"n_telomeres": 5, "cv": 1.0, "mean_intensity": 1000.0},
                        "association_summary": {"association_rate": 0.2, "total_spots": 8},
                        "spots": [{"snr": 2.0, "valid": True}],
                    }
                ]
            )
            assert len(result["suggestions"]) > 0

        _run(_test())

    def test_auto_tune_parameters(self):
        """auto_tune should return a best_config dict."""

        async def _test():
            from teloscopy.agents.improvement_agent import ContinuousImprovementAgent

            agent = ContinuousImprovementAgent()
            result = agent.auto_tune_parameters(
                metric="association_rate",
                target=0.8,
            )
            assert "best_config" in result
            assert "search_space_size" in result
            assert result["search_space_size"] > 0

        _run(_test())


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------


class TestReportAgent:
    """Tests for the ReportAgent."""

    def test_instantiation(self):
        """Should initialise with reporting capabilities."""

        async def _test():
            from teloscopy.agents.report_agent import ReportAgent

            agent = ReportAgent()
            assert agent.name == "report"
            assert "report_generation" in agent.capabilities

        _run(_test())

    def test_generate_full_report(self):
        """A full report should have all expected sections."""

        async def _test():
            from teloscopy.agents.report_agent import ReportAgent

            agent = ReportAgent()
            report = agent.generate_full_report(
                analysis={
                    "statistics": {
                        "n_telomeres": 30,
                        "mean_intensity": 4500.0,
                        "median_intensity": 4200.0,
                        "std_intensity": 800.0,
                        "cv": 0.35,
                    },
                    "association_summary": {"association_rate": 0.8, "total_spots": 40},
                    "validation": {"passed": True, "warnings": []},
                    "suggestions": ["Results look good."],
                },
                risks={
                    "overall_risk_score": 0.35,
                    "risk_category": "moderate",
                    "telomere_percentile": 42.0,
                    "risks": [
                        {"disease": "cardiovascular", "risk_score": 0.4, "category": "moderate"}
                    ],
                },
                diet={
                    "region": "mediterranean",
                    "priority_nutrients": ["omega_3", "folate"],
                    "recommended_foods": [{"name": "Salmon"}, {"name": "Spinach"}],
                    "meal_plan": {"daily_plan": {}},
                    "notes": ["Consult a dietitian."],
                },
                profile={"age": 50, "sex": "male"},
            )
            assert "summary" in report
            assert "telomere_analysis" in report
            assert "risk_assessment" in report
            assert "nutrition_plan" in report
            assert "recommendations" in report
            assert "metadata" in report
            assert report["report_version"] == "1.0"

        _run(_test())

    def test_format_as_html(self):
        """format_as_html should return a valid HTML string."""

        async def _test():
            from teloscopy.agents.report_agent import ReportAgent

            agent = ReportAgent()
            report = agent.generate_full_report(
                analysis={"statistics": {}, "association_summary": {}},
                risks={"risks": []},
                diet={},
            )
            html_str = agent.format_as_html(report)
            assert "<!DOCTYPE html>" in html_str
            assert "Teloscopy Analysis Report" in html_str

        _run(_test())

    def test_format_as_json(self):
        """format_as_json should produce a JSON-safe dict."""

        async def _test():
            from teloscopy.agents.report_agent import ReportAgent

            agent = ReportAgent()
            report = {
                "value": np.float64(3.14),
                "count": np.int64(42),
                "arr": np.array([1, 2, 3]),
                "nested": {"flag": np.bool_(True)},
            }
            safe = agent.format_as_json(report)
            assert isinstance(safe["value"], float)
            assert isinstance(safe["count"], int)
            assert isinstance(safe["arr"], list)
            assert safe["nested"]["flag"] is True

        _run(_test())


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestAgentStateTransitions:
    """Tests for agent lifecycle state transitions."""

    def test_idle_to_running_to_completed(self):
        """An agent should transition IDLE → RUNNING → COMPLETED."""

        async def _test():
            agent = _DummyAgent("lifecycle")
            assert agent.state == AgentState.IDLE
            task = asyncio.create_task(agent.run())
            await asyncio.sleep(0.15)
            assert agent.state == AgentState.RUNNING
            await agent.stop()
            await asyncio.wait_for(task, timeout=3.0)
            assert agent.state == AgentState.COMPLETED

        _run(_test())

    def test_error_state_on_exception(self):
        """An unhandled exception in handle_message should set ERROR state."""

        async def _test():
            class _FailingAgent(BaseAgent):
                async def handle_message(self, msg: AgentMessage) -> None:
                    raise RuntimeError("Intentional failure")

            agent = _FailingAgent("failing", capabilities=[])
            router = _MessageRouter()
            receiver = _DummyAgent("error_catcher")
            router.register(agent)
            router.register(receiver)

            task = asyncio.create_task(agent.run())
            await asyncio.sleep(0.1)

            msg = AgentMessage(
                sender="error_catcher",
                recipient="failing",
                content={},
                message_type="request",
            )
            await agent.message_queue.put(msg)
            await asyncio.sleep(0.5)

            assert agent.state == AgentState.ERROR

            await agent.stop()
            await asyncio.wait_for(task, timeout=3.0)

        _run(_test())

    def test_multiple_agents_orchestration(self):
        """Multiple agents registered with orchestrator can co-exist."""

        async def _test():
            from teloscopy.agents.genomics_agent import GenomicsAgent
            from teloscopy.agents.image_agent import ImageAnalysisAgent
            from teloscopy.agents.improvement_agent import ContinuousImprovementAgent
            from teloscopy.agents.nutrition_agent import NutritionAgent
            from teloscopy.agents.orchestrator import OrchestratorAgent
            from teloscopy.agents.report_agent import ReportAgent

            orch = OrchestratorAgent()
            agents = [
                ImageAnalysisAgent(),
                GenomicsAgent(),
                NutritionAgent(),
                ContinuousImprovementAgent(),
                ReportAgent(),
            ]
            for agent in agents:
                orch.register_agent(agent)

            registered = orch.get_registered_agents()
            assert len(registered) == 5
            for agent in agents:
                assert agent.name in registered

        _run(_test())
