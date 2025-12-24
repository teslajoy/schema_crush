"""calibrate matchers on all available mapping rules (gdc + htan) with proper train/test split."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from schema_crush.knowledge.mapping_rules import load_gdc_rules, load_htan_rules, RuleDatabase
from schema_crush.tools.matchers import BioBERTMatcher, MagnetoMatcher, RuleMatcher
from schema_crush.orchestrator.calibration import create_calibrator_from_mappings
from schema_crush.orchestrator.schema_utils import load_fhir_schema_subset
import schema_crush


def plot_calibration_curves(calibrators, output_path="calibration_all_rules.png"):
    """generate reliability diagram for all matchers."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    matcher_names = ["biobert", "magneto", "rule"]  # use correct key for rulematcher
    display_names = ["BioBERT", "Magneto", "RuleMatcher"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    for idx, (matcher_name, display_name, color) in enumerate(zip(matcher_names, display_names, colors)):
        ax = axes[idx]
        calibrator = calibrators[matcher_name]

        if matcher_name not in calibrator.calibration_curves:
            continue

        bins = calibrator.calibration_curves[matcher_name]
        mean_confidences = []
        actual_accuracies = []
        sample_counts = []

        for bin_data in bins.values():
            if bin_data.sample_count > 0:
                mean_confidences.append(bin_data.predicted_confidence)
                actual_accuracies.append(bin_data.actual_accuracy)
                sample_counts.append(bin_data.sample_count)

        # plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)

        # plot actual calibration curve
        if mean_confidences:
            ax.plot(mean_confidences, actual_accuracies, 'o-',
                   color=color, linewidth=2.5, markersize=8)

            # add sample count as marker size variation
            max_samples = max(sample_counts)
            sizes = [100 + (count / max_samples) * 300 for count in sample_counts]
            ax.scatter(mean_confidences, actual_accuracies,
                      s=sizes, color=color, alpha=0.3, zorder=2)

        stats = calibrator.get_statistics(matcher_name)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted confidence", fontsize=12, fontweight='bold')
        ax.set_ylabel("actual accuracy", fontsize=12, fontweight='bold')
        ax.set_title(f"{display_name}\n"
                    f"acc: {stats['overall_accuracy']:.1%} | "
                    f"ece: {stats['calibration_error']:.3f}",
                    fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

        # # optional: show calibration quality labels
        # if stats['calibration_error'] < 0.05:
        #     interpretation = "well-calibrated"
        #     text_color = "green"
        # elif stats['calibration_error'] < 0.15:
        #     interpretation = "moderately calibrated"
        #     text_color = "orange"
        # else:
        #     interpretation = "poorly calibrated"
        #     text_color = "red"
        #
        # ax.text(0.98, 0.02, interpretation,
        #        transform=ax.transAxes,
        #        fontsize=10, verticalalignment='bottom',
        #        horizontalalignment='right',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=text_color, linewidth=2),
        #        color=text_color, fontweight='bold')

    plt.suptitle("confidence calibration: reliability diagrams (all gdc + htan rules)\n"
                "marker size proportional to sample count per bin",
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\ncalibration plot saved to: {output_path}")


def main():
    """calibrate all matchers with proper train/test split and generate summary report."""

    print("="*70)
    print("COMPREHENSIVE CALIBRATION: ALL GDC + HTAN MAPPINGS")
    print("="*70)

    # load fhir schema (optimized: only 6 relevant biomedical resources)
    print("\nloading fhir schema from linkml...")
    relevant_resources = [
        "Patient", "Specimen", "Condition",
        "ResearchStudy", "ResearchSubject", "Observation"
    ]
    fhir_schema = load_fhir_schema_subset(
        resources=relevant_resources,
        required_fields=True
    )
    total_fields = sum(len(fields) for fields in fhir_schema.values())
    print(f"  loaded {len(fhir_schema)} resources, {total_fields} total fields")
    print(f"  resources: {', '.join(fhir_schema.keys())}")

    # load all mapping rules
    print("\nloading mapping rules...")
    gdc_rules = load_gdc_rules()
    htan_rules = load_htan_rules()
    all_rules = gdc_rules + htan_rules

    print(f"  gdc rules: {len(gdc_rules)}")
    print(f"  htan rules: {len(htan_rules)}")
    print(f"  total: {len(all_rules)}")

    # train/test split (80/20)
    print("\ncreating train/test split (80/20)...")
    train_rules, test_rules = train_test_split(all_rules, test_size=0.2, random_state=42)
    print(f"  train: {len(train_rules)} rules ({len(train_rules)/len(all_rules):.1%})")
    print(f"  test: {len(test_rules)} rules ({len(test_rules)/len(all_rules):.1%})")

    # create rule database (ALL rules for rulematcher since it's a lookup table, not a learned model)
    print("\nbuilding rulematcher on ALL data (no train/test split for knowledge base)...")
    db = RuleDatabase()
    db.add_rules(all_rules)
    print(f"  rulematcher database: {len(db.rules)} rules")

    # initialize matchers
    print("\ninitializing matchers...")
    biobert = BioBERTMatcher()
    magneto = MagnetoMatcher()
    rule_matcher = RuleMatcher(db)
    print("  all matchers initialized")

    # calibrate on test set
    print("\n" + "="*70)
    print("CALIBRATION IN PROGRESS (on held-out test set)...")
    print("="*70)

    print(f"\n[1/3] calibrating biobert on test set ({len(test_rules)} rules)...")
    biobert_cal = create_calibrator_from_mappings(
        matcher=biobert,
        mapping_rules=test_rules,
        target_schema=fhir_schema,
        max_samples=None
    )

    print(f"\n[2/3] calibrating magneto on test set ({len(test_rules)} rules)...")
    magneto_cal = create_calibrator_from_mappings(
        matcher=magneto,
        mapping_rules=test_rules,
        target_schema=fhir_schema,
        max_samples=None
    )

    print(f"\n[3/3] calibrating rulematcher on test set ({len(test_rules)} rules)...")
    rule_cal = create_calibrator_from_mappings(
        matcher=rule_matcher,
        mapping_rules=test_rules,
        target_schema=fhir_schema,
        max_samples=None
    )

    # generate summary report
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY REPORT")
    print("="*70)

    results = []
    for matcher_name, calibrator in [
        ("BioBERT", biobert_cal),
        ("Magneto", magneto_cal),
        ("RuleMatcher", rule_cal)
    ]:
        # use correct key for get_statistics
        key = "biobert" if matcher_name == "BioBERT" else "magneto" if matcher_name == "Magneto" else "rule"
        stats = calibrator.get_statistics(key)

        if not stats:
            print(f"warning: no statistics available for {matcher_name}")
            continue

        results.append({
            "Matcher": matcher_name,
            "Train Size": len(train_rules) if matcher_name == "RuleMatcher" else "N/A",
            "Test Size": len(test_rules),
            "Accuracy": stats.get('overall_accuracy', 0.0),
            "Mean Confidence": stats.get('mean_confidence', 0.0),
            "ECE": stats.get('calibration_error', 0.0),
        })

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    # detailed reports
    print("\n" + "="*70)
    print("DETAILED CALIBRATION REPORTS")
    print("="*70)

    biobert_cal.print_calibration_report("biobert")
    magneto_cal.print_calibration_report("magneto")
    rule_cal.print_calibration_report("rule")  # use correct key

    # generate plots
    print("\n" + "="*70)
    print("GENERATING CALIBRATION PLOTS")
    print("="*70)

    # use package-relative path for plot output
    package_root = Path(importlib.resources.files(schema_crush.__name__).parent)
    calibrators_dir = package_root / "calibrators" / "calibration_logs"
    calibrators_dir.mkdir(parents=True, exist_ok=True)
    output_path = calibrators_dir / "calibration_all_rules.png"

    plot_calibration_curves(
        {
            "biobert": biobert_cal,
            "magneto": magneto_cal,
            "rule": rule_cal  # use correct key
        },
        output_path=str(output_path)
    )

    # key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    print(f"""
1. dataset split:
   - total rules: {len(all_rules)} (gdc: {len(gdc_rules)}, htan: {len(htan_rules)})
   - train set: {len(train_rules)} rules ({len(train_rules)/len(all_rules):.1%})
   - test set: {len(test_rules)} rules ({len(test_rules)/len(all_rules):.1%})

2. calibration quality:
   - biobert ece: {results[0]['ECE']:.3f}
   - magneto ece: {results[1]['ECE']:.3f}
   - rulematcher ece: {results[2]['ECE']:.3f}

3. accuracy rankings:
   1st: {sorted(results, key=lambda x: x['Accuracy'], reverse=True)[0]['Matcher']} ({sorted(results, key=lambda x: x['Accuracy'], reverse=True)[0]['Accuracy']:.1%})
   2nd: {sorted(results, key=lambda x: x['Accuracy'], reverse=True)[1]['Matcher']} ({sorted(results, key=lambda x: x['Accuracy'], reverse=True)[1]['Accuracy']:.1%})
   3rd: {sorted(results, key=lambda x: x['Accuracy'], reverse=True)[2]['Matcher']} ({sorted(results, key=lambda x: x['Accuracy'], reverse=True)[2]['Accuracy']:.1%})
""")

    # save artifacts to calibrators/ (script is already in calibrators/)
    save_path = Path(__file__).parent
    save_path.mkdir(exist_ok=True)

    # save summary csv to summaries/
    import pickle
    summaries_dir = save_path / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    df.to_csv(summaries_dir / "calibration_summary.csv", index=False)

    # save calibrators to pkl/ subdirectory
    pkl_dir = save_path / "pkl"
    pkl_dir.mkdir(exist_ok=True)

    with open(pkl_dir / "biobert_calibrator.pkl", "wb") as f:
        pickle.dump(biobert_cal, f)
    with open(pkl_dir / "magneto_calibrator.pkl", "wb") as f:
        pickle.dump(magneto_cal, f)
    with open(pkl_dir / "rule_calibrator.pkl", "wb") as f:
        pickle.dump(rule_cal, f)

    print(f"\nartifacts saved to: {save_path}/")
    print("  - pkl/biobert_calibrator.pkl")
    print("  - pkl/magneto_calibrator.pkl")
    print("  - pkl/rule_calibrator.pkl")
    print("  - summaries/calibration_summary.csv")

    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)

    return {
        "biobert": biobert_cal,
        "magneto": magneto_cal,
        "rule": rule_cal,
        "train_rules": train_rules,
        "test_rules": test_rules,
        "summary": df
    }


if __name__ == '__main__':
    calibrators = main()