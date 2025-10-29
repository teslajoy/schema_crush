"""tests for ClaudeAgent with tool use."""

import pytest
import os
from unittest.mock import patch, MagicMock
from schema_crush.orchestrator.agents import ClaudeAgent, MappingProposal


@pytest.fixture
def mock_anthropic_api():
    """mock anthropic api to avoid real api calls in tests."""
    with patch('schema_crush.orchestrator.agents.claude_agent.ChatAnthropic') as mock:
        yield mock


@pytest.fixture
def mock_tools():
    """mock matcher tools."""
    with patch('schema_crush.orchestrator.agents.tools.biobert_match') as biobert, \
         patch('schema_crush.orchestrator.agents.tools.magneto_match') as magneto, \
         patch('schema_crush.orchestrator.agents.tools.rule_match') as rule:
        yield biobert, magneto, rule


class TestClaudeAgentInitialization:
    """test ClaudeAgent initialization."""

    def test_init_with_defaults(self, mock_anthropic_api):
        """test initialization with default parameters."""
        agent = ClaudeAgent()

        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.task == "field"
        assert agent.name == "claude_agent_field"
        mock_anthropic_api.assert_called_once()

    def test_init_with_custom_params(self, mock_anthropic_api):
        """test initialization with custom parameters."""
        agent = ClaudeAgent(
            model="claude-opus-4-20250514",
            api_key="test-key",
            task="entity"
        )

        assert agent.model == "claude-opus-4-20250514"
        assert agent.task == "entity"
        assert agent.name == "claude_agent_entity"

    def test_init_field_task_prompt(self, mock_anthropic_api):
        """test field task gets correct prompt."""
        agent = ClaudeAgent(task="field")
        assert "field" in agent.system_prompt.lower()
        assert "magneto" in agent.system_prompt.lower()

    def test_init_entity_task_prompt(self, mock_anthropic_api):
        """test entity task gets correct prompt."""
        agent = ClaudeAgent(task="entity")
        assert "entity" in agent.system_prompt.lower()
        assert "resource" in agent.system_prompt.lower()

    def test_init_content_task_prompt(self, mock_anthropic_api):
        """test content task gets correct prompt."""
        agent = ClaudeAgent(task="content")
        assert "values" in agent.system_prompt.lower() or "content" in agent.system_prompt.lower()
        assert "coding" in agent.system_prompt.lower() or "coded" in agent.system_prompt.lower()


class TestClaudeAgentProposeMappings:
    """test ClaudeAgent.propose_mappings()."""

    def test_propose_mappings_returns_proposals(self, mock_anthropic_api):
        """test that propose_mappings returns MappingProposal objects."""
        # mock llm response with structured output
        mock_response = MagicMock()
        mock_response.content = """
CHOSEN TARGET: Patient.id
CONFIDENCE: 0.85
REASONING: This is a test mapping based on structural patterns.
"""
        mock_response.tool_calls = None

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_anthropic_api.return_value.bind_tools.return_value = mock_llm_instance

        agent = ClaudeAgent(task="field")
        proposals = agent.propose_mappings(
            source_field="case_id",
            candidate_targets=["Patient.id", "Patient.identifier"],
            context={}
        )

        assert len(proposals) == 1
        assert isinstance(proposals[0], MappingProposal)
        assert proposals[0].source_field == "case_id"
        assert proposals[0].target_field == "Patient.id"
        assert proposals[0].confidence == 0.85

    def test_propose_mappings_with_tool_calls(self, mock_anthropic_api, mock_tools):
        """test that tool calls are executed and results fed back to llm."""
        biobert, magneto, rule = mock_tools

        # mock first response with tool calls
        first_response = MagicMock()
        first_response.tool_calls = [{
            'name': 'magneto_match',
            'args': {'source': 'case_id', 'candidates': ['Patient.id']},
            'id': 'tool_1',
            'type': 'tool_call'
        }]

        # mock second response after tool execution
        second_response = MagicMock()
        second_response.content = """
CHOSEN TARGET: Patient.id
CONFIDENCE: 0.90
REASONING: Magneto shows high structural similarity.
"""
        second_response.tool_calls = None

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = [first_response, second_response]
        mock_anthropic_api.return_value.bind_tools.return_value = mock_llm_instance

        # mock tool result
        magneto.invoke.return_value = [{"target": "Patient.id", "score": 0.85}]

        agent = ClaudeAgent(task="field")
        proposals = agent.propose_mappings(
            source_field="case_id",
            candidate_targets=["Patient.id"],
            context={}
        )

        assert len(proposals) == 1
        assert proposals[0].target_field == "Patient.id"
        assert proposals[0].confidence == 0.90
        magneto.invoke.assert_called_once()

    def test_propose_mappings_multi_turn_tool_calling(self, mock_anthropic_api, mock_tools):
        """test multi-turn agentic loop with multiple tool calls."""
        biobert, magneto, rule = mock_tools

        # iteration 1: call magneto
        resp1 = MagicMock()
        resp1.tool_calls = [{'name': 'magneto_match', 'args': {}, 'id': '1', 'type': 'tool_call'}]

        # iteration 2: call biobert
        resp2 = MagicMock()
        resp2.tool_calls = [{'name': 'biobert_match', 'args': {}, 'id': '2', 'type': 'tool_call'}]

        # iteration 3: final answer
        resp3 = MagicMock()
        resp3.content = "CHOSEN TARGET: Patient.id\nCONFIDENCE: 0.88\nREASONING: test"
        resp3.tool_calls = None

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = [resp1, resp2, resp3]
        mock_anthropic_api.return_value.bind_tools.return_value = mock_llm_instance

        magneto.invoke.return_value = [{"target": "Patient.id", "score": 0.60}]
        biobert.invoke.return_value = [{"target": "Patient.id", "score": 0.82}]

        agent = ClaudeAgent(task="field")
        proposals = agent.propose_mappings(
            source_field="case_id",
            candidate_targets=["Patient.id"],
            context={}
        )

        assert len(proposals) == 1
        assert mock_llm_instance.invoke.call_count == 3
        magneto.invoke.assert_called_once()
        biobert.invoke.assert_called_once()


class TestClaudeAgentResponseParsing:
    """test response parsing logic."""

    def test_parse_structured_response(self, mock_anthropic_api):
        """test parsing of structured CHOSEN TARGET/CONFIDENCE/REASONING format."""
        mock_response = MagicMock()
        mock_response.content = """
Based on the evidence, I recommend:

CHOSEN TARGET: Specimen.processing.method
CONFIDENCE: 0.75
REASONING: The field represents tissue preservation techniques which are
processing steps that occur after specimen collection. Magneto shows 0.60
similarity and BioBERT shows 0.82 similarity.
"""
        mock_response.tool_calls = None

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_anthropic_api.return_value.bind_tools.return_value = mock_llm_instance

        agent = ClaudeAgent(task="field")
        proposals = agent.propose_mappings(
            source_field="tissue_preservation_method",
            candidate_targets=["Specimen.processing.method"],
            context={}
        )

        assert proposals[0].target_field == "Specimen.processing.method"
        assert proposals[0].confidence == 0.75
        assert "preservation techniques" in proposals[0].reasoning

    def test_parse_markdown_formatting(self, mock_anthropic_api):
        """test parsing strips markdown formatting from target."""
        mock_response = MagicMock()
        mock_response.content = """
CHOSEN TARGET: **Patient.id**
CONFIDENCE: 0.90
REASONING: Test
"""
        mock_response.tool_calls = None

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_anthropic_api.return_value.bind_tools.return_value = mock_llm_instance

        agent = ClaudeAgent(task="field")
        proposals = agent.propose_mappings(
            source_field="case_id",
            candidate_targets=["Patient.id"],
            context={}
        )

        # should strip ** asterisks
        assert proposals[0].target_field == "Patient.id"

    def test_parse_fallback_to_content_search(self, mock_anthropic_api):
        """test fallback parsing when structured format not found."""
        mock_response = MagicMock()
        mock_response.content = "I recommend Patient.id based on the analysis."
        mock_response.tool_calls = None

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_anthropic_api.return_value.bind_tools.return_value = mock_llm_instance

        agent = ClaudeAgent(task="field")
        proposals = agent.propose_mappings(
            source_field="case_id",
            candidate_targets=["Patient.id", "Patient.identifier"],
            context={}
        )

        # should find "Patient.id" in content
        assert proposals[0].target_field == "Patient.id"

    def test_parse_final_fallback_to_first_candidate(self, mock_anthropic_api):
        """test final fallback uses first candidate when nothing matches."""
        mock_response = MagicMock()
        mock_response.content = "This is ambiguous."
        mock_response.tool_calls = None

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_anthropic_api.return_value.bind_tools.return_value = mock_llm_instance

        agent = ClaudeAgent(task="field")
        proposals = agent.propose_mappings(
            source_field="unknown_field",
            candidate_targets=["Patient.id", "Patient.identifier"],
            context={}
        )

        # should default to first candidate
        assert proposals[0].target_field == "Patient.id"


class TestClaudeAgentExplainDecision:
    """test explain_decision()."""

    def test_explain_decision(self, mock_anthropic_api):
        """test that explain_decision returns formatted explanation."""
        agent = ClaudeAgent(task="field")

        proposal = MappingProposal(
            source_field="case_id",
            target_field="Patient.id",
            confidence=0.85,
            reasoning="Test reasoning",
            supporting_evidence={"agent": "claude_agent_field"}
        )

        explanation = agent.explain_decision(proposal)

        assert "claude_agent_field" in explanation
        assert "field matching" in explanation
        assert "case_id" in explanation
        assert "Patient.id" in explanation
        assert "0.850" in explanation


class TestClaudeAgentGroundTruthContext:
    """test ground truth handling in context."""

    def test_ground_truth_included_in_prompt(self, mock_anthropic_api):
        """test that ground truth is included in llm prompt for validation."""
        mock_response = MagicMock()
        mock_response.content = "CHOSEN TARGET: Patient.id\nCONFIDENCE: 0.9\nREASONING: test"
        mock_response.tool_calls = None

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_anthropic_api.return_value.bind_tools.return_value = mock_llm_instance

        agent = ClaudeAgent(task="field")
        agent.propose_mappings(
            source_field="case_id",
            candidate_targets=["Patient.id"],
            context={"ground_truth": "Patient.identifier"}
        )

        # check that ground truth was included in the prompt
        call_args = mock_llm_instance.invoke.call_args[0][0]
        user_message = str(call_args[1].content)
        assert "ground truth" in user_message.lower()
        assert "Patient.identifier" in user_message


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping integration test"
)
class TestClaudeAgentIntegration:
    """integration tests with real claude api (requires ANTHROPIC_API_KEY)."""

    def test_simple_field_mapping_integration(self):
        """test simple field mapping with real claude api."""
        agent = ClaudeAgent(task="field")

        proposals = agent.propose_mappings(
            source_field="patient_id",
            candidate_targets=[
                "Patient.id",
                "Patient.identifier",
                "Person.id",
                "Practitioner.id"
            ],
            context={}
        )

        # should return a proposal
        assert len(proposals) == 1
        assert isinstance(proposals[0], MappingProposal)

        # should pick a patient-related field
        assert "Patient" in proposals[0].target_field

        # should have reasonable confidence
        assert 0.0 <= proposals[0].confidence <= 1.0

        # should provide reasoning
        assert len(proposals[0].reasoning) > 50