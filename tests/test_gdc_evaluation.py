"""evaluate embedders on gdc→fhir mappings with ground truth."""

import pytest
import json
import importlib.resources
from pathlib import Path
import pandas as pd
from schema_crush.tools.embeddings.biobert_embedder import BioBERTEmbedder
from schema_crush.tools.embeddings.magneto_embedder import MagnetoEmbedder


@pytest.fixture
def gdc_mappings():
    """load gdc→fhir ground truth mappings."""
    base_path = Path(importlib.resources.files('schema_crush'))
    mapping_file = base_path / 'data' / 'resources' / 'gdc_mapping' / 'case.json'

    with open(mapping_file) as f:
        data = json.load(f)

    # extract source→destination mappings
    mappings = {}
    for m in data['mappings']:
        src = m['source']['name']
        dst = m['destination']['name']
        mappings[src] = dst

    return mappings


@pytest.fixture
def biobert_embedder():
    """create biobert embedder instance."""
    return BioBERTEmbedder()


@pytest.fixture
def magneto_embedder():
    """create magneto embedder instance."""
    return MagnetoEmbedder()


def test_gdc_mappings_structure(gdc_mappings):
    """test that gdc mappings loaded correctly."""
    print(f"\nloaded {len(gdc_mappings)} gdc→fhir mappings")

    # show first 10
    print("\nfirst 10 mappings:")
    for i, (src, dst) in enumerate(list(gdc_mappings.items())[:10]):
        print(f"  {src} → {dst}")

    assert len(gdc_mappings) > 0


def evaluate_embedder(embedder, embedder_name, gdc_mappings, top_k=5):
    """
    evaluate embedder against gdc ground truth.

    args:
        embedder: embedder instance
        embedder_name: name for display
        gdc_mappings: ground truth source→destination mappings
        top_k: number of top predictions to consider

    returns:
        dict with precision, recall, mrr metrics
    """
    source_fields = list(gdc_mappings.keys())[:20]  # test on first 20
    target_fields = list(set(gdc_mappings.values()))  # unique fhir fields

    print(f"\n{'='*60}")
    print(f"{embedder_name} evaluation")
    print(f"{'='*60}")
    print(f"source fields: {len(source_fields)}")
    print(f"target fields: {len(target_fields)}")

    # get similarity matrix
    sim_matrix = embedder.batch_similarity(source_fields, target_fields)

    correct_at_1 = 0
    correct_at_k = 0
    reciprocal_ranks = []

    print("\npredictions:")
    for i, src_field in enumerate(source_fields):
        ground_truth = gdc_mappings[src_field]

        # get top k predictions
        scores = sim_matrix[i]
        top_k_indices = scores.argsort()[-top_k:][::-1]
        top_k_targets = [target_fields[idx] for idx in top_k_indices]
        top_k_scores = [scores[idx] for idx in top_k_indices]

        # check if ground truth is in top k
        if ground_truth in top_k_targets:
            rank = top_k_targets.index(ground_truth) + 1
            reciprocal_ranks.append(1.0 / rank)
            correct_at_k += 1
            if rank == 1:
                correct_at_1 += 1
                status = "✓"
            else:
                status = f"✓@{rank}"
        else:
            reciprocal_ranks.append(0.0)
            status = "✗"

        # print prediction
        pred = top_k_targets[0]
        pred_score = top_k_scores[0]
        print(f"  {status} {src_field} → {pred} (score: {pred_score:.4f}, truth: {ground_truth})")

    # calculate metrics
    precision_at_1 = correct_at_1 / len(source_fields)
    recall_at_k = correct_at_k / len(source_fields)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print(f"\n{'='*60}")
    print(f"metrics:")
    print(f"  precision@1: {precision_at_1:.2%} ({correct_at_1}/{len(source_fields)})")
    print(f"  recall@{top_k}: {recall_at_k:.2%} ({correct_at_k}/{len(source_fields)})")
    print(f"  mrr (mean reciprocal rank): {mrr:.4f}")
    print(f"{'='*60}\n")

    return {
        'precision_at_1': precision_at_1,
        f'recall_at_{top_k}': recall_at_k,
        'mrr': mrr,
        'correct_at_1': correct_at_1,
        'correct_at_k': correct_at_k,
        'total': len(source_fields)
    }


def test_biobert_on_gdc(biobert_embedder, gdc_mappings):
    """evaluate biobert on gdc mappings."""
    results = evaluate_embedder(biobert_embedder, "biobert", gdc_mappings, top_k=5)

    # biobert should do reasonably well due to biomedical training
    assert results['precision_at_1'] >= 0.0  # no minimum requirement, just measure


def test_magneto_on_gdc(magneto_embedder, gdc_mappings):
    """evaluate magneto on gdc mappings."""
    results = evaluate_embedder(magneto_embedder, "magneto (mpnet)", gdc_mappings, top_k=5)

    # magneto was trained on gdc, should perform well
    assert results['precision_at_1'] >= 0.0  # no minimum requirement, just measure


def test_comparison(biobert_embedder, magneto_embedder, gdc_mappings):
    """compare biobert vs magneto on gdc mappings."""
    biobert_results = evaluate_embedder(biobert_embedder, "biobert", gdc_mappings, top_k=5)
    magneto_results = evaluate_embedder(magneto_embedder, "magneto", gdc_mappings, top_k=5)

    print("\n" + "="*60)
    print("comparison summary")
    print("="*60)
    print(f"{'metric':<30} {'biobert':<15} {'magneto':<15}")
    print("-"*60)
    print(f"{'precision@1':<30} {biobert_results['precision_at_1']:<15.2%} {magneto_results['precision_at_1']:<15.2%}")
    print(f"{'recall@5':<30} {biobert_results['recall_at_5']:<15.2%} {magneto_results['recall_at_5']:<15.2%}")
    print(f"{'mrr':<30} {biobert_results['mrr']:<15.4f} {magneto_results['mrr']:<15.4f}")
    print("="*60)

    # store results for analysis
    assert biobert_results['total'] == magneto_results['total']