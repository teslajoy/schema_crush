"""autonomous agent interface for multi-agent reasoning."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MappingProposal:
    """agent's proposed mapping with reasoning."""
    source_field: str
    target_field: str
    confidence: float
    reasoning: str
    supporting_evidence: Optional[dict] = None


class AutonomousAgent(ABC):
    """autonomous agent that proposes mappings with reasoning.

    unlike matchers (which score similarities), autonomous agents:
    - analyze mapping context
    - generate proposals with explanations
    - provide supporting evidence
    - can collaborate with other agents
    """

    @abstractmethod
    def propose_mappings(
        self,
        source_field: str,
        candidate_targets: List[str],
        context: dict
    ) -> List[MappingProposal]:
        """propose mappings with reasoning and confidence scores.

        args:
            source_field: source field to map
            candidate_targets: list of potential fhir targets
            context: additional context (ex. schema, previous mappings)

        returns:
            list of mapping proposals with reasoning
        """
        pass

    @abstractmethod
    def explain_decision(self, proposal: MappingProposal) -> str:
        """provide detailed explanation for a mapping proposal.

        args:
            proposal: the mapping proposal to explain

        returns:
            human-readable explanation
        """
        pass