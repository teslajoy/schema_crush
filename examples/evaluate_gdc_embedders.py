"""run gdc evaluation and generate comparison report."""

import json
import importlib.resources
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from schema_crush.tools.embeddings.biobert_embedder import BioBERTEmbedder
from schema_crush.tools.embeddings.magneto_embedder import MagnetoEmbedder

# set seed for reproducibility
np.random.seed(1234)


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


def evaluate_embedder(embedder, embedder_name, gdc_mappings, top_k=5, max_fields=20):
    """
    evaluate embedder against gdc ground truth.

    args:
        embedder: embedder instance
        embedder_name: name for display
        gdc_mappings: ground truth source->destination mappings
        top_k: number of top predictions to consider
        max_fields: maximum number of source fields to evaluate (none for all)

    returns:
        dict with precision, recall, mrr, f1 metrics and detailed results
    """
    source_fields = list(gdc_mappings.keys())
    if max_fields:
        source_fields = source_fields[:max_fields]
    target_fields = sorted(set(gdc_mappings.values()))  # sorted for determinism

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
    detailed_results = []

    print("\npredictions:")
    for i, src_field in enumerate(source_fields):
        # handle missing mappings gracefully
        if src_field not in gdc_mappings:
            print(f"  [skip] {src_field} -> not in ground truth")
            continue

        ground_truth = gdc_mappings[src_field]

        # get top k predictions (stable sort for reproducibility)
        scores = sim_matrix[i]
        top_k_indices = np.argsort(-scores, kind='stable')[:top_k]
        top_k_targets = [target_fields[idx] for idx in top_k_indices]
        top_k_scores = [scores[idx] for idx in top_k_indices]

        # check if ground truth is in top k
        if ground_truth in top_k_targets:
            rank = top_k_targets.index(ground_truth) + 1
            reciprocal_ranks.append(1.0 / rank)
            correct_at_k += 1
            if rank == 1:
                correct_at_1 += 1
                status = "correct"
            else:
                status = f"correct@{rank}"
        else:
            reciprocal_ranks.append(0.0)
            status = "miss"

        # print prediction
        pred = top_k_targets[0]
        pred_score = top_k_scores[0]
        symbol = "ok" if status == "correct" else "x"
        print(f"  [{symbol}] {src_field} -> {pred} (score: {pred_score:.4f}, truth: {ground_truth})")

        # store detailed result
        detailed_results.append({
            'source_field': src_field,
            'predicted': pred,
            'predicted_score': float(pred_score),
            'ground_truth': ground_truth,
            'status': status,
            'top_k_predictions': [
                {'field': tgt, 'score': float(score)}
                for tgt, score in zip(top_k_targets, top_k_scores)
            ]
        })

    # calculate metrics
    num_evaluated = len(detailed_results)
    precision_at_1 = correct_at_1 / num_evaluated if num_evaluated > 0 else 0
    recall_at_k = correct_at_k / num_evaluated if num_evaluated > 0 else 0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

    # f1 score (balanced metric)
    f1 = (2 * precision_at_1 * recall_at_k / (precision_at_1 + recall_at_k)
          if (precision_at_1 + recall_at_k) > 0 else 0)

    print(f"\n{'='*60}")
    print(f"metrics:")
    print(f"  precision@1: {precision_at_1:.2%} ({correct_at_1}/{num_evaluated})")
    print(f"  recall@{top_k}: {recall_at_k:.2%} ({correct_at_k}/{num_evaluated})")
    print(f"  f1 score: {f1:.4f}")
    print(f"  mrr (mean reciprocal rank): {mrr:.4f}")
    print(f"{'='*60}\n")

    return {
        'embedder': embedder_name,
        'metrics': {
            'precision_at_1': precision_at_1,
            f'recall_at_{top_k}': recall_at_k,
            'f1_score': f1,
            'mrr': mrr,
            'correct_at_1': correct_at_1,
            'correct_at_k': correct_at_k,
            'total': num_evaluated
        },
        'detailed_results': detailed_results
    }


def save_comparison_report(biobert_results, magneto_results, output_dir='examples/reports'):
    """save comprehensive comparison report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_dir) / f"gdc_evaluation_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # save detailed results
    detailed_path = report_dir / "detailed.json"
    with open(detailed_path, 'w') as f:
        json.dump({
            'biobert': biobert_results,
            'magneto': magneto_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # save comparison summary
    summary_path = report_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("gdc->fhir embedder evaluation\n")
        f.write("="*60 + "\n\n")

        f.write("evaluation setup:\n")
        f.write(f"  ground truth: gdc case mappings\n")
        f.write(f"  source fields: {biobert_results['metrics']['total']}\n")
        f.write(f"  top_k: 5\n\n")

        f.write("embedders tested:\n")
        f.write("  biobert: dmis-lab/biobert-v1.1 (biomedical text embeddings)\n")
        f.write("  magneto: mpnet (schema matching trained on gdc benchmark)\n\n")

        f.write("comparison summary:\n")
        f.write("="*60 + "\n")
        f.write(f"{'metric':<30} {'biobert':<15} {'magneto':<15}\n")
        f.write("-"*60 + "\n")

        bb_m = biobert_results['metrics']
        mg_m = magneto_results['metrics']

        f.write(f"{'precision@1':<30} {bb_m['precision_at_1']:<15.2%} {mg_m['precision_at_1']:<15.2%}\n")
        f.write(f"{'recall@5':<30} {bb_m['recall_at_5']:<15.2%} {mg_m['recall_at_5']:<15.2%}\n")
        f.write(f"{'f1 score':<30} {bb_m['f1_score']:<15.4f} {mg_m['f1_score']:<15.4f}\n")
        f.write(f"{'mrr':<30} {bb_m['mrr']:<15.4f} {mg_m['mrr']:<15.4f}\n")
        f.write("="*60 + "\n\n")

        f.write("key finding:\n")
        improvement = mg_m['precision_at_1'] / bb_m['precision_at_1'] if bb_m['precision_at_1'] > 0 else float('inf')
        f.write(f"  magneto outperforms biobert {improvement:.1f}x on gdc schema matching\n")
        f.write("  magneto was trained on gdc benchmark data\n")
        f.write("  biobert is general biomedical text model\n\n")

        f.write("generated files:\n")
        f.write(f"  report directory: {report_dir}\n")
        f.write(f"  detailed results: detailed.json\n")
        f.write(f"  summary: summary.txt\n")
        f.write(f"  produced mappings: produced_mappings.txt\n")
        f.write(f"  csv exports: biobert_results.csv, magneto_results.csv\n")

    # save csv exports for easy analysis
    biobert_df = pd.DataFrame(biobert_results['detailed_results'])
    biobert_df.to_csv(report_dir / "biobert_results.csv", index=False)

    magneto_df = pd.DataFrame(magneto_results['detailed_results'])
    magneto_df.to_csv(report_dir / "magneto_results.csv", index=False)

    # save simple produced mappings view
    produced_path = report_dir / "produced_mappings.txt"
    with open(produced_path, 'w') as f:
        f.write("biobert mappings:\n\n")
        for result in biobert_results['detailed_results']:
            src = result['source_field']
            pred = result['predicted']
            score = result['predicted_score']
            status = "ok" if result['status'] == "correct" else "x"
            f.write(f"   [{status}] {src} -> {pred} (score: {score:.3f})\n")

        f.write("\n" + "="*60 + "\n\n")
        f.write("magneto mappings:\n\n")
        for result in magneto_results['detailed_results']:
            src = result['source_field']
            pred = result['predicted']
            score = result['predicted_score']
            status = "ok" if result['status'] == "correct" else "x"
            f.write(f"   [{status}] {src} -> {pred} (score: {score:.3f})\n")

        f.write("\n" + "="*60 + "\n\n")
        f.write("evaluation complete:\n")
        f.write(f"   biobert: {bb_m['correct_at_1']}/{bb_m['total']} correct ({bb_m['precision_at_1']:.1%})\n")
        f.write(f"   magneto: {mg_m['correct_at_1']}/{mg_m['total']} correct ({mg_m['precision_at_1']:.1%})\n")

    return {
        'report_dir': str(report_dir),
        'detailed': str(detailed_path),
        'summary': str(summary_path),
        'produced_mappings': str(produced_path),
        'biobert_csv': str(report_dir / "biobert_results.csv"),
        'magneto_csv': str(report_dir / "magneto_results.csv")
    }


def main(top_k=5, max_fields=20):
    """
    run evaluation and generate report.

    args:
        top_k: number of top predictions to consider
        max_fields: maximum number of fields to evaluate (none for all)
    """
    print("loading gdc mappings...")
    gdc_mappings = load_gdc_mappings()
    print(f"loaded {len(gdc_mappings)} gdc->fhir mappings\n")

    print("creating embedders...")
    biobert = BioBERTEmbedder()
    magneto = MagnetoEmbedder()

    print(f"\nrunning evaluations (top_k={top_k}, max_fields={max_fields})...")
    biobert_results = evaluate_embedder(biobert, "biobert", gdc_mappings,
                                       top_k=top_k, max_fields=max_fields)
    magneto_results = evaluate_embedder(magneto, "magneto", gdc_mappings,
                                       top_k=top_k, max_fields=max_fields)

    print("\ngenerating comparison report...")
    files = save_comparison_report(biobert_results, magneto_results)

    print("\n" + "="*60)
    print("evaluation complete")
    print("="*60)
    print(f"report directory: {files['report_dir']}")
    print(f"  - detailed.json")
    print(f"  - summary.txt")
    print(f"  - produced_mappings.txt")
    print(f"  - biobert_results.csv")
    print(f"  - magneto_results.csv")
    print("\nquick comparison:")
    print(f"  biobert precision@1: {biobert_results['metrics']['precision_at_1']:.2%}")
    print(f"  magneto precision@1: {magneto_results['metrics']['precision_at_1']:.2%}")
    print(f"  biobert f1: {biobert_results['metrics']['f1_score']:.4f}")
    print(f"  magneto f1: {magneto_results['metrics']['f1_score']:.4f}")


if __name__ == '__main__':
    import sys

    # allow command line args: python evaluate_gdc_embedders.py [top_k] [max_fields]
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_fields = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    main(top_k=top_k, max_fields=max_fields)