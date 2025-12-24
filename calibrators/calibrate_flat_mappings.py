"""calibrate matchers using flat mapping database with tier-aware calibration.

uses flat mappings (gdc + htan) as ground truth instead of hierarchical rules.
generates separate calibration curves per tier (entity, field, content).
now includes content_values table calibration (source_value -> FHIR path).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from schema_crush.mappings.flat_loader import load_flat_mappings
from schema_crush.mappings.flat import Tier
from schema_crush.tools.matchers.rule_matcher import RuleMatcher
from schema_crush.tools.matchers.biobert_matcher import BioBERTMatcher
from schema_crush.tools.matchers.magneto_matcher import MagnetoMatcher
from schema_crush.orchestrator.calibration import ConfidenceCalibrator


def get_tier_candidates(db, tier):
    """get destination candidates for a specific tier.

    filters destinations to only those associated with sources of the given tier.
    this improves accuracy by reducing candidate space.
    """
    candidates = set()
    for src in db.sources.values():
        if src.tier == tier:
            dests = db._dest_by_source.get(src.id, [])
            for d in dests:
                candidates.add(d.destination)
    return list(candidates)


def calibrate_matcher_on_tier(
    matcher,
    matcher_name,
    db,
    tier,
    candidates,
    calibrator,
    max_samples=None,
):
    """calibrate a single matcher on entity/field tiers (NOT content).

    args:
        matcher: matcher instance
        matcher_name: name for calibration keys
        db: flat mapping database
        tier: Tier enum (ENTITY or FIELD only)
        candidates: list of candidate destinations
        calibrator: ConfidenceCalibrator to add predictions to
        max_samples: optional limit on samples

    returns:
        list of result dicts with predictions
    """
    # get sources for this tier
    sources = [s for s in db.sources.values() if s.tier == tier]

    if max_samples and len(sources) > max_samples:
        sources = random.sample(sources, max_samples)

    results = []

    for i, src in enumerate(sources):
        # get ground truth destinations
        dests = db._dest_by_source.get(src.id, [])
        if not dests:
            continue
        ground_truths = {d.destination.lower() for d in dests}
        gt_display = dests[0].destination

        # get matcher prediction
        try:
            match_results = matcher.match(src.source, candidates)
            if match_results:
                top_pred, score = match_results[0]
                correct = top_pred.lower() in ground_truths
            else:
                top_pred, score, correct = "", 0.0, False
        except Exception as e:
            print(f"  warning: {matcher_name} failed on {src.source}: {e}")
            continue

        results.append({
            "source": src.source,
            "context": src.source_context,
            "tier": tier.value,
            "ground_truth": gt_display,
            "prediction": top_pred,
            "score": score,
            "correct": correct,
        })

        # add to calibrator - both global and tier-specific
        calibrator.add_prediction(matcher_name, score, correct)
        calibrator.add_prediction(matcher_name, score, correct, tier=tier.value)

        if (i + 1) % 100 == 0:
            print(f"    processed {i + 1}/{len(sources)}...")

    return results


def calibrate_matcher_on_content(
    matcher,
    matcher_name,
    db,
    calibrator,
    max_samples=None,
):
    """calibrate matcher on content tier using content_values table.

    uses content_values + content_fhir_targets tables (your YAML terminology files).
    tests: source_value (e.g., "Adenocarcinoma") -> fhir_path (e.g., "Condition.code")

    args:
        matcher: matcher instance
        matcher_name: name for calibration keys
        db: flat mapping database with content_values
        calibrator: ConfidenceCalibrator to add predictions to
        max_samples: optional limit on samples

    returns:
        list of result dicts with predictions
    """
    # build lookup: source_value -> all fhir_paths (handles duplicates)
    source_to_paths = {}
    source_to_category = {}
    source_to_code = {}
    for cv in db.content_values:
        key = cv.source_value.lower()
        if key not in source_to_paths:
            source_to_paths[key] = set()
            source_to_category[key] = cv.source_category
            source_to_code[key] = cv.code
        # get all fhir_paths for this content_value
        for t in db.content_fhir_targets:
            if t.content_value_id == cv.id:
                source_to_paths[key].add(t.fhir_path.lower())

    # get unique source values to evaluate
    unique_sources = list(source_to_paths.keys())
    if max_samples and len(unique_sources) > max_samples:
        unique_sources = random.sample(unique_sources, max_samples)

    # candidates = all unique FHIR paths from content_fhir_targets
    candidates = list(set(t.fhir_path for t in db.content_fhir_targets))

    results = []

    for i, source_lower in enumerate(unique_sources):
        ground_truths = source_to_paths[source_lower]
        if not ground_truths:
            continue

        gt_display = list(ground_truths)[0]  # for display only

        # get original case source_value for matching
        original_source = next(
            (cv.source_value for cv in db.content_values if cv.source_value.lower() == source_lower),
            source_lower
        )

        # get matcher prediction
        try:
            match_results = matcher.match(original_source, candidates)
            if match_results:
                top_pred, score = match_results[0]
                correct = top_pred.lower() in ground_truths
            else:
                top_pred, score, correct = "", 0.0, False
        except Exception as e:
            print(f"  warning: {matcher_name} failed on {original_source}: {e}")
            continue

        results.append({
            "source": original_source,
            "context": source_to_category[source_lower],
            "tier": "content",
            "ground_truth": gt_display,
            "prediction": top_pred,
            "score": score,
            "correct": correct,
            "code": source_to_code[source_lower],
        })

        # add to calibrator - global and content tier
        calibrator.add_prediction(matcher_name, score, correct)
        calibrator.add_prediction(matcher_name, score, correct, tier="content")

        if (i + 1) % 200 == 0:
            print(f"    processed {i + 1}/{len(unique_sources)}...")

    return results


def plot_tier_calibration(calibrator, matcher_name, output_path):
    """plot calibration curves for all tiers of a matcher."""

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    tiers = [None, "entity", "field", "content"]
    tier_labels = ["global", "entity", "field", "content"]
    colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71"]

    for idx, (tier, label, color) in enumerate(zip(tiers, tier_labels, colors)):
        ax = axes[idx]

        key = f"{matcher_name}:{tier}" if tier else matcher_name

        if key not in calibrator.calibration_curves:
            ax.set_title(f"{label}\nno data", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        bins = calibrator.calibration_curves[key]
        confs = []
        accs = []
        counts = []

        for bin_data in bins.values():
            if bin_data.sample_count > 0:
                confs.append(bin_data.predicted_confidence)
                accs.append(bin_data.actual_accuracy)
                counts.append(bin_data.sample_count)

        # perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)

        if confs:
            ax.plot(confs, accs, 'o-', color=color, linewidth=2.5, markersize=8)

            # marker size by sample count
            max_count = max(counts) if counts else 1
            sizes = [100 + (c / max_count) * 300 for c in counts]
            ax.scatter(confs, accs, s=sizes, color=color, alpha=0.3)

        stats = calibrator.get_statistics(key)
        acc = stats.get('overall_accuracy', 0)
        ece = stats.get('calibration_error', 0)
        n = stats.get('total_predictions', 0)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted confidence", fontsize=11)
        ax.set_ylabel("actual accuracy", fontsize=11)
        ax.set_title(f"{label} (n={n})\nacc: {acc:.1%} | ece: {ece:.3f}",
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

    plt.suptitle(f"{matcher_name} calibration by tier", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  saved: {output_path}")


def main(use_expert_embeddings: bool = False):
    """calibrate all matchers using flat mappings with tier awareness.

    args:
        use_expert_embeddings: if True, use expert embeddings for BioBERT/Magneto
    """
    print("="*70)
    print("FLAT MAPPING CALIBRATION (tier-aware)")
    if use_expert_embeddings:
        print("  >> using expert embeddings for BioBERT/Magneto")
    print("="*70)

    # load flat database
    print("\nloading flat mapping database...")
    db = load_flat_mappings()
    stats = db.stats()
    print(f"  sources: {stats['counts']['sources']}")
    print(f"  destinations: {stats['counts']['destinations']}")
    print(f"  content_values: {len(db.content_values)}")
    print(f"  content_fhir_targets: {len(db.content_fhir_targets)}")
    print(f"  schemas: {stats['schemas']}")

    # get candidates per tier
    print("\nbuilding candidate sets per tier...")
    entity_candidates = get_tier_candidates(db, Tier.ENTITY)
    field_candidates = get_tier_candidates(db, Tier.FIELD)
    content_candidates = list(set(t.fhir_path for t in db.content_fhir_targets))

    print(f"  entity candidates: {len(entity_candidates)}")
    print(f"  field candidates: {len(field_candidates)}")
    print(f"  content candidates (FHIR paths): {len(content_candidates)}")

    # initialize matchers
    print("\ninitializing matchers...")
    rule_matcher = RuleMatcher(db)
    print("  rule matcher initialized")

    biobert_matcher = BioBERTMatcher(use_expert_embeddings=use_expert_embeddings)
    print(f"  biobert matcher initialized (expert={use_expert_embeddings})")

    magneto_matcher = MagnetoMatcher(use_expert_embeddings=use_expert_embeddings)
    print(f"  magneto matcher initialized (expert={use_expert_embeddings})")

    # calibrate each matcher
    matchers = [
        ("rule", rule_matcher),
        ("biobert", biobert_matcher),
        ("magneto", magneto_matcher),
    ]

    calibrators = {}
    all_results = []

    for matcher_name, matcher in matchers:
        print(f"\n{'='*70}")
        print(f"CALIBRATING: {matcher_name}")
        print(f"{'='*70}")

        calibrator = ConfidenceCalibrator()

        # entity tier
        print(f"\n  entity tier ({len(entity_candidates)} candidates)...")
        results = calibrate_matcher_on_tier(
            matcher=matcher,
            matcher_name=matcher_name,
            db=db,
            tier=Tier.ENTITY,
            candidates=entity_candidates,
            calibrator=calibrator,
            max_samples=None,
        )
        for r in results:
            r["matcher"] = matcher_name
        all_results.extend(results)
        if results:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"    accuracy: {acc:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")

        # field tier
        print(f"\n  field tier ({len(field_candidates)} candidates)...")
        results = calibrate_matcher_on_tier(
            matcher=matcher,
            matcher_name=matcher_name,
            db=db,
            tier=Tier.FIELD,
            candidates=field_candidates,
            calibrator=calibrator,
            max_samples=300,
        )
        for r in results:
            r["matcher"] = matcher_name
        all_results.extend(results)
        if results:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"    accuracy: {acc:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")

        # content tier - from content_values table (NEW)
        print(f"\n  content tier ({len(db.content_values)} values, {len(content_candidates)} FHIR path candidates)...")
        results = calibrate_matcher_on_content(
            matcher=matcher,
            matcher_name=matcher_name,
            db=db,
            calibrator=calibrator,
            max_samples=500,
        )
        for r in results:
            r["matcher"] = matcher_name
        all_results.extend(results)
        if results:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"    accuracy: {acc:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")

        # build calibration curves
        calibrator.calibrate_all()
        calibrators[matcher_name] = calibrator

        # print calibration report
        print(f"\n  calibration reports for {matcher_name}:")
        calibrator.print_calibration_report(matcher_name)
        for tier in ["entity", "field", "content"]:
            key = f"{matcher_name}:{tier}"
            if key in calibrator.calibration_data:
                calibrator.print_calibration_report(key)

    # summary table
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)

    summary_rows = []
    for matcher_name, calibrator in calibrators.items():
        for tier in [None, "entity", "field", "content"]:
            key = f"{matcher_name}:{tier}" if tier else matcher_name
            stats = calibrator.get_statistics(key)
            if stats:
                summary_rows.append({
                    "matcher": matcher_name,
                    "tier": tier or "global",
                    "accuracy": stats.get("overall_accuracy", 0),
                    "mean_conf": stats.get("mean_confidence", 0),
                    "ece": stats.get("calibration_error", 0),
                    "n": stats.get("total_predictions", 0),
                })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    # save artifacts
    print("\n" + "="*70)
    print("SAVING ARTIFACTS")
    print("="*70)

    save_dir = Path(__file__).parent
    save_dir.mkdir(exist_ok=True)

    # save calibrators to pkl/ subdirectory
    pkl_dir = save_dir / "pkl"
    pkl_dir.mkdir(exist_ok=True)

    for matcher_name, calibrator in calibrators.items():
        pkl_path = pkl_dir / f"{matcher_name}_flat_calibrator.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(calibrator, f)
        print(f"  saved: {pkl_path}")

    # save summaries to summaries/ subdirectory
    summaries_dir = save_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)

    summary_path = summaries_dir / "flat_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  saved: {summary_path}")

    # save detailed results
    results_df = pd.DataFrame(all_results)
    results_path = summaries_dir / "flat_calibration_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  saved: {results_path}")

    # generate plots
    print("\n  generating plots...")
    plots_dir = save_dir / "calibration_logs"
    plots_dir.mkdir(exist_ok=True)

    for matcher_name, calibrator in calibrators.items():
        plot_path = plots_dir / f"{matcher_name}_flat_calibration.png"
        plot_tier_calibration(calibrator, matcher_name, plot_path)

    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)

    return calibrators, summary_df, results_df


def compare_embeddings():
    """run calibration with and without embeddings, show comparison."""
    print("\n" + "="*70)
    print("RUNNING WITHOUT EXPERT EMBEDDINGS")
    print("="*70)
    _, summary_no, _ = main(use_expert_embeddings=False)
    summary_no = summary_no.copy()
    summary_no["mode"] = "no_emb"

    print("\n" + "="*70)
    print("RUNNING WITH EXPERT EMBEDDINGS")
    print("="*70)
    _, summary_yes, _ = main(use_expert_embeddings=True)
    summary_yes = summary_yes.copy()
    summary_yes["mode"] = "with_emb"

    # comparison table
    print("\n" + "="*70)
    print("COMPARISON: WITHOUT vs WITH EMBEDDINGS")
    print("="*70)

    merged = summary_no.merge(summary_yes, on=["matcher", "tier"], suffixes=("_no", "_yes"))
    merged["acc_delta"] = merged["accuracy_yes"] - merged["accuracy_no"]
    merged["ece_delta"] = merged["ece_yes"] - merged["ece_no"]

    cols = ["matcher", "tier", "accuracy_no", "accuracy_yes", "acc_delta", "ece_no", "ece_yes", "ece_delta"]
    print("\n" + merged[cols].to_string(index=False))

    # save comparison
    save_dir = Path(__file__).parent / "summaries"
    merged.to_csv(save_dir / "embeddings_comparison.csv", index=False)
    print(f"\nsaved: {save_dir / 'embeddings_comparison.csv'}")


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--embeddings", is_flag=True, help="Use expert embeddings for BioBERT/Magneto")
    @click.option("--compare", is_flag=True, help="Run both modes and show comparison")
    def cli(embeddings, compare):
        if compare:
            compare_embeddings()
        else:
            main(use_expert_embeddings=embeddings)

    cli()