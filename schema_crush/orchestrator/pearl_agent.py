"""pearl agentic orchestrator using langgraph.

implements pearl loop as stateful agent workflow with:
- perceive: extract schema information
- reason: generate matches using embedders
- act: filter and rank matches
- hitl: pause for human feedback on medium confidence matches
- learn: update from user feedback

supports multi-tier execution (entity -> field -> content) in one run.
"""

from typing import List, Dict, Tuple, Optional, Any, TypedDict, Annotated, Literal
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


class MatchTier(Enum):
    """matching tiers for schema matching."""
    ENTITY = "entity"
    FIELD = "field"
    CONTENT = "content"


class ConfidenceLevel(Enum):
    """confidence levels for matches."""
    HIGH = "high"  # >= 95%
    MEDIUM = "medium"  # 85-95% (requires HITL)
    LOW = "low"  # < 85%


@dataclass
class Match:
    """represents a schema match."""
    source_entity: str
    source_field: str
    target_entity: str
    target_field: str
    score: float
    confidence: str  # use string for json serialization
    tier: str
    embedder_scores: Dict[str, float]
    requires_hitl: bool
    user_feedback: Optional[bool] = None
    reasoning: Optional[str] = None
    match_id: Optional[str] = None  # unique id for hitl tracking


class PEARLState(TypedDict):
    """state for pearl agent workflow."""
    # input
    source_data: Dict[str, Any]  # entity -> dataframe dict
    target_schema: Dict[str, List[str]]  # entity -> field list

    # perception
    source_entities: List[str]
    target_entities: List[str]
    source_fields: Dict[str, List[str]]
    target_fields: Dict[str, List[str]]

    # reasoning
    current_tier: str
    matches_by_tier: Dict[str, List[Dict]]  # tier -> matches
    embedder_scores: Dict[str, Any]
    reasoning_log: Annotated[List[str], operator.add]  # accumulate reasoning steps

    # action
    filtered_matches: Annotated[List[Dict], operator.add]  # accumulate across tiers
    hitl_queue: Annotated[List[Dict], operator.add]  # accumulate hitl items

    # learning
    user_feedback: Dict[str, bool]  # match_id -> correct/incorrect
    embedder_weights: Dict[str, float]
    metrics: Dict[str, Any]

    # control flow
    next_action: str
    iteration: int
    tiers_to_run: List[str]
    awaiting_hitl: bool


TIERS = ["entity", "field", "content"]


class PerceiveAgent:
    """perceive phase: extract schema information."""

    def __init__(self):
        self.name = "perceive_agent"

    def __call__(self, state: PEARLState) -> Dict[str, Any]:
        """extract schema metadata from source and target."""
        print(f"\n[{self.name}] extracting schema information...")

        source_data = state['source_data']
        target_schema = state['target_schema']

        # extract entities
        source_entities = list(source_data.keys())
        target_entities = list(target_schema.keys())

        # extract fields
        source_fields = {}
        for entity_name, df_dict in source_data.items():
            if isinstance(df_dict, dict) and 'dataframe' in df_dict:
                df = df_dict['dataframe']
            else:
                df = df_dict
            source_fields[entity_name] = list(df.columns)

        reasoning = f"perceived {len(source_entities)} source entities, {len(target_entities)} target entities"
        print(f"[{self.name}] {reasoning}")

        return {
            'source_entities': source_entities,
            'target_entities': target_entities,
            'source_fields': source_fields,
            'target_fields': target_schema,
            'reasoning_log': [reasoning],
            'matches_by_tier': {}
        }


class ReasonAgent:
    """reason phase: generate matches using embedders."""

    def __init__(
        self,
        embedders: Dict[str, Any],
        embedder_weights: Dict[str, float],
        confidence_thresholds: Dict[str, float]
    ):
        self.name = "reason_agent"
        self.embedders = embedders

        # ensure all embedders have weights (default to equal if missing)
        complete_weights = {}
        for embedder_name in embedders.keys():
            complete_weights[embedder_name] = embedder_weights.get(embedder_name, 1.0)

        # normalize weights to sum=1
        total = sum(complete_weights.values())
        self.embedder_weights = {k: v / total for k, v in complete_weights.items()}
        self.confidence_thresholds = confidence_thresholds

    def __call__(self, state: PEARLState) -> Dict[str, Any]:
        """execute matching for current tier."""
        tier_str = state['current_tier']
        tier = MatchTier(tier_str)

        print(f"\n[{self.name}] reasoning at tier: {tier.value}")

        if tier == MatchTier.ENTITY:
            matches, reasoning = self._reason_entity(state)
        elif tier == MatchTier.FIELD:
            matches, reasoning = self._reason_field(state)
        elif tier == MatchTier.CONTENT:
            matches, reasoning = self._reason_content(state)
        else:
            matches, reasoning = [], f"unknown tier: {tier}"

        print(f"[{self.name}] generated {len(matches)} candidate matches")

        # store matches by tier
        matches_by_tier = state.get('matches_by_tier', {})
        matches_by_tier[tier_str] = matches

        return {
            'matches_by_tier': matches_by_tier,
            'reasoning_log': [reasoning]
        }

    def _reason_entity(self, state: PEARLState) -> Tuple[List[Dict], str]:
        """tier 1: entity matching."""
        source_entities = state['source_entities']
        target_entities = state['target_entities']

        matches = []
        embedder_scores_all = {}

        # get embeddings from all embedders
        for embedder_name, embedder in self.embedders.items():
            source_embs = embedder.embed(source_entities)
            target_embs = embedder.embed(target_entities)

            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(source_embs, target_embs)
            embedder_scores_all[embedder_name] = sim_matrix

        # combine scores
        combined = self._combine_scores(embedder_scores_all)

        # create matches
        for i, src_entity in enumerate(source_entities):
            for j, tgt_entity in enumerate(target_entities):
                score = float(combined[i, j])
                confidence = self._determine_confidence(score)

                individual_scores = {
                    name: float(embedder_scores_all[name][i, j])
                    for name in self.embedders.keys()
                }

                match_id = f"{src_entity}.*_{tgt_entity}.*_entity"

                match = Match(
                    source_entity=src_entity,
                    source_field='*',
                    target_entity=tgt_entity,
                    target_field='*',
                    score=score,
                    confidence=confidence.value,
                    tier=MatchTier.ENTITY.value,
                    embedder_scores=individual_scores,
                    requires_hitl=(confidence == ConfidenceLevel.MEDIUM),
                    reasoning=f"entity match: {src_entity} -> {tgt_entity}",
                    match_id=match_id
                )
                matches.append(asdict(match))

        reasoning = f"entity matching: evaluated {len(source_entities)}x{len(target_entities)} pairs"
        return matches, reasoning

    def _reason_field(self, state: PEARLState) -> Tuple[List[Dict], str]:
        """tier 2: field matching."""
        matches = []
        total_pairs = 0

        for src_entity, src_fields in state['source_fields'].items():
            for tgt_entity, tgt_fields in state['target_fields'].items():
                # get embeddings
                embedder_scores_all = {}
                for embedder_name, embedder in self.embedders.items():
                    source_embs = embedder.embed(src_fields)
                    target_embs = embedder.embed(tgt_fields)

                    from sklearn.metrics.pairwise import cosine_similarity
                    sim_matrix = cosine_similarity(source_embs, target_embs)
                    embedder_scores_all[embedder_name] = sim_matrix

                # combine scores
                combined = self._combine_scores(embedder_scores_all)

                # create matches
                for i, src_field in enumerate(src_fields):
                    for j, tgt_field in enumerate(tgt_fields):
                        score = float(combined[i, j])
                        confidence = self._determine_confidence(score)

                        individual_scores = {
                            name: float(embedder_scores_all[name][i, j])
                            for name in self.embedders.keys()
                        }

                        match_id = f"{src_entity}.{src_field}_{tgt_entity}.{tgt_field}_field"

                        match = Match(
                            source_entity=src_entity,
                            source_field=src_field,
                            target_entity=tgt_entity,
                            target_field=tgt_field,
                            score=score,
                            confidence=confidence.value,
                            tier=MatchTier.FIELD.value,
                            embedder_scores=individual_scores,
                            requires_hitl=(confidence == ConfidenceLevel.MEDIUM),
                            reasoning=f"field match: {src_field} -> {tgt_field} (score: {score:.4f})",
                            match_id=match_id
                        )
                        matches.append(asdict(match))
                        total_pairs += 1

        reasoning = f"field matching: evaluated {total_pairs} field pairs"
        return matches, reasoning

    def _reason_content(self, state: PEARLState) -> Tuple[List[Dict], str]:
        """tier 3: content matching."""
        matches = []
        total_pairs = 0

        for src_entity, src_fields in state['source_fields'].items():
            # get source dataframe
            src_data = state['source_data'][src_entity]
            if isinstance(src_data, dict) and 'dataframe' in src_data:
                src_df = src_data['dataframe']
            else:
                src_df = src_data

            for tgt_entity, tgt_fields in state['target_fields'].items():
                # get embeddings
                embedder_scores_all = {}
                for embedder_name, embedder in self.embedders.items():
                    # source: embed with content
                    source_embs = []
                    for src_field in src_fields:
                        column_values = src_df[src_field].tolist()
                        # try with dataframe param (magneto), fallback to without (biobert)
                        try:
                            emb = embedder.embed_content(src_field, column_values, dataframe=src_df)
                        except TypeError:
                            emb = embedder.embed_content(src_field, column_values)
                        source_embs.append(emb[0])
                    source_embs = np.array(source_embs)

                    # target: just field names
                    target_embs = embedder.embed(tgt_fields)

                    from sklearn.metrics.pairwise import cosine_similarity
                    sim_matrix = cosine_similarity(source_embs, target_embs)
                    embedder_scores_all[embedder_name] = sim_matrix

                # combine scores
                combined = self._combine_scores(embedder_scores_all)

                # create matches
                for i, src_field in enumerate(src_fields):
                    for j, tgt_field in enumerate(tgt_fields):
                        score = float(combined[i, j])
                        confidence = self._determine_confidence(score)

                        individual_scores = {
                            name: float(embedder_scores_all[name][i, j])
                            for name in self.embedders.keys()
                        }

                        match_id = f"{src_entity}.{src_field}_{tgt_entity}.{tgt_field}_content"

                        match = Match(
                            source_entity=src_entity,
                            source_field=src_field,
                            target_entity=tgt_entity,
                            target_field=tgt_field,
                            score=score,
                            confidence=confidence.value,
                            tier=MatchTier.CONTENT.value,
                            embedder_scores=individual_scores,
                            requires_hitl=(confidence == ConfidenceLevel.MEDIUM),
                            reasoning=f"content match: {src_field} -> {tgt_field} (with values)",
                            match_id=match_id
                        )
                        matches.append(asdict(match))
                        total_pairs += 1

        reasoning = f"content matching: evaluated {total_pairs} field pairs with values"
        return matches, reasoning

    def _combine_scores(self, embedder_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """combine embedder scores using normalized weighted average."""
        combined = None
        for embedder_name, scores in embedder_scores.items():
            weight = self.embedder_weights[embedder_name]
            if combined is None:
                combined = weight * scores
            else:
                combined += weight * scores
        return combined

    def _determine_confidence(self, score: float) -> ConfidenceLevel:
        """determine confidence level based on thresholds."""
        if score >= self.confidence_thresholds['high']:
            return ConfidenceLevel.HIGH
        elif score >= self.confidence_thresholds['medium']:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class ActAgent:
    """act phase: filter and rank matches."""

    def __init__(self, top_k: int = 5, filter_low: bool = True):
        self.name = "act_agent"
        self.top_k = top_k
        self.filter_low = filter_low

    def __call__(self, state: PEARLState) -> Dict[str, Any]:
        """filter and rank matches for current tier."""
        print(f"\n[{self.name}] filtering and ranking matches...")

        tier_str = state['current_tier']
        matches_by_tier = state.get('matches_by_tier', {})
        matches = matches_by_tier.get(tier_str, [])

        # filter low confidence
        if self.filter_low:
            filtered = [m for m in matches if m['confidence'] != 'low']
        else:
            filtered = matches

        # group by source field
        grouped = {}
        for match in filtered:
            key = (match['source_entity'], match['source_field'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(match)

        # keep top k per source field
        final_matches = []
        hitl_items = []

        for key, group in grouped.items():
            sorted_group = sorted(group, key=lambda m: m['score'], reverse=True)
            top_matches = sorted_group[:self.top_k]
            final_matches.extend(top_matches)

            # collect hitl items
            for match in top_matches:
                if match['requires_hitl']:
                    hitl_items.append(match)

        reasoning = f"tier {tier_str}: filtered to {len(final_matches)} matches, {len(hitl_items)} require hitl"
        print(f"[{self.name}] {reasoning}")

        return {
            'filtered_matches': final_matches,
            'hitl_queue': hitl_items,
            'reasoning_log': [reasoning]
        }


class HitlAgent:
    """hitl phase: pause for human feedback on medium confidence matches."""

    def __init__(self):
        self.name = "hitl_agent"

    def __call__(self, state: PEARLState) -> Command:
        """check if hitl is needed and pause if so."""
        hitl_queue = state.get('hitl_queue', [])

        if hitl_queue and not state.get('awaiting_hitl', False):
            print(f"\n[{self.name}] pausing for human feedback on {len(hitl_queue)} matches")
            # signal interruption - workflow will pause here
            # frontend should provide user_feedback and resume
            return Command(update={'awaiting_hitl': True}, resume="learn")
        else:
            print(f"\n[{self.name}] no hitl required or feedback provided, continuing")
            return Command(update={'awaiting_hitl': False}, resume="learn")


class LearnAgent:
    """learn phase: update from feedback and metrics."""

    def __init__(self):
        self.name = "learn_agent"
        self.memory = []  # store past matches and feedback

    def __call__(self, state: PEARLState) -> Dict[str, Any]:
        """analyze results and update weights."""
        print(f"\n[{self.name}] learning from results...")

        filtered_matches = state.get('filtered_matches', [])
        user_feedback = state.get('user_feedback', {})

        # calculate metrics for current tier
        tier_str = state['current_tier']
        tier_matches = [m for m in filtered_matches if m['tier'] == tier_str]

        metrics = {
            'tier': tier_str,
            'total_matches': len(tier_matches),
            'high_confidence': len([m for m in tier_matches if m['confidence'] == 'high']),
            'medium_confidence': len([m for m in tier_matches if m['confidence'] == 'medium']),
            'low_confidence': len([m for m in tier_matches if m['confidence'] == 'low']),
            'requires_hitl': len([m for m in tier_matches if m['requires_hitl']]),
            'feedback_count': len(user_feedback)
        }

        if user_feedback:
            metrics['correct_feedback'] = sum(1 for v in user_feedback.values() if v)
            metrics['incorrect_feedback'] = sum(1 for v in user_feedback.values() if not v)

        # store in memory
        self.memory.append({
            'tier': tier_str,
            'iteration': state.get('iteration', 0),
            'matches': tier_matches,
            'metrics': metrics,
            'feedback': user_feedback
        })

        reasoning = f"tier {tier_str}: learned from {len(tier_matches)} matches, memory size: {len(self.memory)}"
        print(f"[{self.name}] {reasoning}")

        return {
            'metrics': metrics,
            'reasoning_log': [reasoning]
        }


def maybe_next_tier(state: PEARLState) -> Literal["bump_tier", "finish"]:
    """decide if we should run next tier or finish."""
    tiers_to_run = state.get('tiers_to_run', TIERS)
    idx = state.get('iteration', 0)
    has_more = (idx + 1) < len(tiers_to_run)
    return "bump_tier" if has_more else "finish"


def bump_tier(state: PEARLState) -> Dict[str, Any]:
    """advance to next tier."""
    tiers_to_run = state.get('tiers_to_run', TIERS)
    idx = state.get('iteration', 0) + 1
    next_tier = tiers_to_run[idx]
    print(f"\n[workflow] advancing to tier: {next_tier} (iteration {idx})")
    return {
        'current_tier': next_tier,
        'iteration': idx,
        'awaiting_hitl': False  # reset hitl flag for next tier
    }


class PEARLAgent:
    """
    pearl agentic orchestrator using langgraph.

    implements stateful workflow:
    perceive -> reason -> act -> hitl? -> learn -> (next tier or end)
    """

    def __init__(
        self,
        embedders: Dict[str, Any],
        embedder_weights: Optional[Dict[str, float]] = None,
        confidence_thresholds: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        enable_hitl: bool = True
    ):
        """
        initialize pearl agent.

        args:
            embedders: dict of embedder instances
            embedder_weights: optional weights for each embedder (will be normalized)
            confidence_thresholds: thresholds for confidence levels
            top_k: top matches to keep per field
            enable_hitl: whether to enable hitl interrupts
        """
        self.embedders = embedders
        self.embedder_weights = embedder_weights or {name: 1.0 for name in embedders}
        self.confidence_thresholds = confidence_thresholds or {
            'high': 0.95,
            'medium': 0.85,
            'low': 0.0
        }
        self.top_k = top_k
        self.enable_hitl = enable_hitl

        # cache for dataframes (not serialized in checkpoints)
        self._data_cache = {}

        # initialize agents
        self.perceive_agent = PerceiveAgent()
        self.reason_agent = ReasonAgent(
            self.embedders,
            self.embedder_weights,
            self.confidence_thresholds
        )
        self.act_agent = ActAgent(top_k=self.top_k)
        self.hitl_agent = HitlAgent() if enable_hitl else None
        self.learn_agent = LearnAgent()

        # build langgraph workflow
        self.memory = MemorySaver()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """build langgraph workflow with conditional tier progression."""
        workflow = StateGraph(PEARLState)

        # add nodes
        workflow.add_node("perceive", self.perceive_agent)
        workflow.add_node("reason", self.reason_agent)
        workflow.add_node("act", self.act_agent)
        if self.enable_hitl:
            workflow.add_node("hitl", self.hitl_agent)
        workflow.add_node("learn", self.learn_agent)
        workflow.add_node("bump_tier", bump_tier)

        # set entry point
        workflow.set_entry_point("perceive")

        # add edges
        workflow.add_edge("perceive", "reason")
        workflow.add_edge("reason", "act")

        if self.enable_hitl:
            workflow.add_edge("act", "hitl")
            # hitl returns Command, not dict
        else:
            workflow.add_edge("act", "learn")

        # conditional: next tier or finish
        workflow.add_conditional_edges(
            "learn",
            maybe_next_tier,
            {
                "bump_tier": "bump_tier",
                "finish": END
            }
        )

        # bump tier goes back to reason
        workflow.add_edge("bump_tier", "reason")

        # compile workflow
        # note: checkpointing disabled for now due to dataframe serialization issues
        # TODO: implement custom checkpointer that handles dataframes
        if self.enable_hitl:
            return workflow.compile(
                checkpointer=None,  # disabled for now
                interrupt_before=["hitl"]
            )
        else:
            return workflow.compile()

    def run(
        self,
        source_data: Dict[str, pd.DataFrame],
        target_schema: Dict[str, List[str]],
        tiers: Optional[List[str]] = None,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        run pearl workflow across multiple tiers.

        args:
            source_data: source dataframes
            target_schema: target schema definition
            tiers: which tiers to run (default: all)
            thread_id: conversation thread id for memory

        returns:
            results dict with matches, metrics, hitl items
        """
        tiers = tiers or TIERS

        print(f"\n{'='*60}")
        print(f"pearl agent - running tiers: {tiers}")
        print(f"{'='*60}")

        # initialize state
        initial_state = {
            'source_data': source_data,
            'target_schema': target_schema,
            'current_tier': tiers[0],
            'tiers_to_run': tiers,
            'matches_by_tier': {},
            'reasoning_log': [],
            'filtered_matches': [],
            'hitl_queue': [],
            'user_feedback': {},
            'embedder_weights': self.embedder_weights,
            'iteration': 0,
            'awaiting_hitl': False
        }

        # run workflow
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.workflow.invoke(initial_state, config)

        # deduplicate matches across tiers (keep highest tier match for each field pair)
        filtered_matches = final_state.get('filtered_matches', [])
        deduped_matches = self._deduplicate_matches(filtered_matches)

        # format results
        results = {
            'matches_by_tier': final_state.get('matches_by_tier', {}),
            'matches': deduped_matches,
            'metrics': final_state.get('metrics', {}),
            'hitl_queue': final_state.get('hitl_queue', []),
            'reasoning_log': final_state.get('reasoning_log', []),
            'iteration': final_state.get('iteration', 0)
        }

        print(f"\n{'='*60}")
        print(f"results: {len(deduped_matches)} matches (deduplicated), "
              f"{len(results['hitl_queue'])} require hitl")
        print(f"{'='*60}\n")

        return results

    def _deduplicate_matches(self, matches: List[Dict]) -> List[Dict]:
        """
        deduplicate matches across tiers.

        keeps the highest-scoring match for each (source_field, target_field) pair,
        with tier priority: content > field > entity.
        """
        # group by field pair
        grouped = {}
        tier_priority = {'content': 3, 'field': 2, 'entity': 1}

        for match in matches:
            key = (match['source_entity'], match['source_field'],
                   match['target_entity'], match['target_field'])

            if key not in grouped:
                grouped[key] = match
            else:
                # keep match with higher tier priority, or higher score if same tier
                existing = grouped[key]
                if (tier_priority.get(match['tier'], 0) > tier_priority.get(existing['tier'], 0) or
                    (match['tier'] == existing['tier'] and match['score'] > existing['score'])):
                    grouped[key] = match

        return list(grouped.values())