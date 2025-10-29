"""evaluate rule matcher against gdc ground truth."""

import json
import importlib.resources
from pathlib import Path
import numpy as np
from schema_crush.tools.matchers import RuleMatcher
from schema_crush.knowledge.mapping_rules import RuleDatabase, load_htan_rules, load_gdc_rules


def load_gdc_mappings():
    """load gdc->fhir ground truth mappings."""
    base_path = Path(importlib.resources.files('schema_crush'))
    mapping_file = base_path / 'data' / 'resources' / 'gdc_mapping' / 'case.json'

    with open(mapping_file) as f:
        data = json.load(f)

    # extract source->destination mappings
    mappings = {}
    for m in data['mappings']:
        src = m['source']['name']
        dst = m['destination']['name']
        mappings[src] = dst

    return mappings


def evaluate_rule_matcher():
    """evaluate rule matcher against gdc ground truth."""
    print("loading mapping rules...")
    db = RuleDatabase()
    db.add_rules(load_htan_rules())
    db.add_rules(load_gdc_rules())
    print(f"loaded {db.count()} rules\n")

    print("loading gdc ground truth...")
    gdc_mappings = load_gdc_mappings()
    print(f"loaded {len(gdc_mappings)} gdc->fhir mappings\n")

    print("creating rule matcher...")
    matcher = RuleMatcher(db)

    # evaluate on first 20 fields (same as evaluate_gdc_embedders.py)
    source_fields = list(gdc_mappings.keys())[:20]
    target_fields = sorted(set(gdc_mappings.values()))

    print(f"\nevaluating rule matcher on {len(source_fields)} fields...")
    print("="*60)

    # get similarity matrix
    sim_matrix = matcher.batch_similarity(source_fields, target_fields)

    correct_at_1 = 0
    correct_at_5 = 0
    top_k = 5

    print("\npredictions:")
    for i, src_field in enumerate(source_fields):
        ground_truth = gdc_mappings[src_field]

        # get top k predictions
        scores = sim_matrix[i]
        top_k_indices = np.argsort(-scores, kind='stable')[:top_k]
        top_k_targets = [target_fields[idx] for idx in top_k_indices]
        top_k_scores = [scores[idx] for idx in top_k_indices]

        # check if ground truth is in top k
        if ground_truth in top_k_targets:
            rank = top_k_targets.index(ground_truth) + 1
            correct_at_5 += 1
            if rank == 1:
                correct_at_1 += 1
                status = "correct"
            else:
                status = f"correct@{rank}"
        else:
            status = "miss"

        # print prediction
        pred = top_k_targets[0]
        pred_score = top_k_scores[0]
        symbol = "ok" if status == "correct" else "x"
        print(f"  [{symbol}] {src_field} -> {pred} (score: {pred_score:.4f}, truth: {ground_truth})")

    # calculate metrics
    num_evaluated = len(source_fields)
    precision_at_1 = correct_at_1 / num_evaluated if num_evaluated > 0 else 0
    recall_at_5 = correct_at_5 / num_evaluated if num_evaluated > 0 else 0
    f1 = (2 * precision_at_1 * recall_at_5 / (precision_at_1 + recall_at_5)
          if (precision_at_1 + recall_at_5) > 0 else 0)

    print(f"\n{'='*60}")
    print(f"rule matcher metrics:")
    print(f"  precision@1: {precision_at_1:.2%} ({correct_at_1}/{num_evaluated})")
    print(f"  recall@5: {recall_at_5:.2%} ({correct_at_5}/{num_evaluated})")
    print(f"  f1 score: {f1:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    evaluate_rule_matcher()