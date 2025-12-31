"""generate calibration plots for context-aware backoff calibration system.

plots reliability diagrams showing:
- calibration curves per matcher (global)
- context-specific curves with backoff visualization
- key eligibility status and data distribution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Optional


def plot_calibration_curves(
    calibrator,
    matchers: List[str] = None,
    output_path: str = "calibration_plot.png"
):
    """generate reliability diagram for matchers.

    args:
        calibrator: ConfidenceCalibrator with calibration_curves
        matchers: list of matcher names to plot (default: biobert, magneto, rule)
        output_path: path to save plot
    """
    if matchers is None:
        matchers = ["biobert", "magneto", "rule"]

    fig, axes = plt.subplots(1, len(matchers), figsize=(5 * len(matchers), 5))
    if len(matchers) == 1:
        axes = [axes]

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]  # red, blue, green, purple

    for idx, matcher_name in enumerate(matchers):
        ax = axes[idx]
        color = colors[idx % len(colors)]

        # find the global key for this matcher
        global_key = f"{matcher_name}:*:*:*"
        legacy_key = matcher_name

        # try global key first, then legacy
        key = None
        if global_key in calibrator.calibration_curves:
            key = global_key
        elif legacy_key in calibrator.calibration_curves:
            key = legacy_key

        if key is None:
            ax.text(0.5, 0.5, f"No data for {matcher_name}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(matcher_name.upper())
            continue

        bins = calibrator.calibration_curves[key]

        # extract data
        mean_confidences = []
        actual_accuracies = []
        sample_counts = []

        for bin_data in bins.values():
            if bin_data.sample_count > 0:
                mean_confidences.append(bin_data.predicted_confidence)
                actual_accuracies.append(bin_data.actual_accuracy)
                sample_counts.append(bin_data.sample_count)

        # plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label="Perfect")

        # plot actual calibration curve
        if mean_confidences:
            ax.plot(mean_confidences, actual_accuracies, 'o-',
                   color=color, linewidth=2.5, markersize=8,
                   label=f"{matcher_name}")

            # add sample count as marker size variation
            max_samples = max(sample_counts) if sample_counts else 1
            sizes = [100 + (count / max_samples) * 300 for count in sample_counts]
            ax.scatter(mean_confidences, actual_accuracies,
                      s=sizes, color=color, alpha=0.3, zorder=2)

        # get statistics
        stats = calibrator.get_statistics(matcher_name)

        # check eligibility (handle legacy calibrators)
        if hasattr(calibrator, 'key_stats') and calibrator.key_stats:
            key_stats = calibrator.key_stats.get(key, {"n_total": 0, "n_pos": 0})
            eligible = calibrator._is_eligible(key)
        else:
            # legacy format - count samples from bins
            total = sum(b.sample_count for b in bins.values())
            key_stats = {"n_total": total, "n_pos": 0}
            eligible = total > 0

        # formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Confidence", fontsize=11)
        ax.set_ylabel("Actual Accuracy", fontsize=11)

        title = f"{matcher_name.upper()}\n"
        if stats:
            title += f"Acc: {stats.get('overall_accuracy', 0):.1%} | ECE: {stats.get('calibration_error', 0):.3f}"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_aspect('equal')

        # add eligibility indicator
        status = "Eligible" if eligible else f"Low data (n={key_stats['n_total']})"
        status_color = "green" if eligible else "orange"
        ax.text(0.05, 0.95, status,
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               color=status_color, fontweight='bold')

    plt.suptitle("Calibration Reliability Diagrams\n(marker size ~ sample count)",
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved: {output_path}")

    return fig


def plot_context_comparison(
    calibrator,
    matcher_name: str = "biobert",
    output_path: str = "calibration_context_comparison.png"
):
    """plot calibration curves for different contexts to show backoff behavior.

    args:
        calibrator: ConfidenceCalibrator with context-aware curves
        matcher_name: matcher to analyze
        output_path: path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # find all keys for this matcher
    matcher_keys = [k for k in calibrator.calibration_curves.keys()
                   if k.startswith(f"{matcher_name}:")]

    if not matcher_keys:
        print(f"no context keys found for {matcher_name}")
        return None

    # group keys by specificity
    global_keys = [k for k in matcher_keys if k.endswith(":*:*:*")]
    tier_keys = [k for k in matcher_keys if k.count(":") == 3 and k.endswith(":*:*") and not k.endswith(":*:*:*")]
    schema_keys = [k for k in matcher_keys if ":*:" in k and not k.endswith(":*:*")]
    specific_keys = [k for k in matcher_keys if "*" not in k.split(":")[-1] and "*" not in k.split(":")[-2]]

    # LEFT PLOT: global vs tier-specific
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label="Perfect")

    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    color_idx = 0

    # plot global
    for key in global_keys[:1]:
        _plot_curve(ax, calibrator, key, colors[color_idx], "Global")
        color_idx += 1

    # plot tier-specific
    for key in sorted(tier_keys)[:3]:
        tier = key.split(":")[1]
        _plot_curve(ax, calibrator, key, colors[color_idx], f"Tier: {tier}")
        color_idx += 1

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Confidence", fontsize=11)
    ax.set_ylabel("Actual Accuracy", fontsize=11)
    ax.set_title(f"{matcher_name.upper()}: Global vs Tier-Specific", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    # RIGHT PLOT: key eligibility heatmap
    ax = axes[1]

    # build data for all keys
    keys_data = []
    for key in sorted(matcher_keys):
        stats = calibrator.key_stats.get(key, {"n_total": 0, "n_pos": 0})
        eligible = calibrator._is_eligible(key)
        keys_data.append({
            "key": key.replace(f"{matcher_name}:", ""),
            "n_total": stats["n_total"],
            "n_pos": stats["n_pos"],
            "eligible": eligible
        })

    # sort by n_total descending
    keys_data.sort(key=lambda x: -x["n_total"])
    keys_data = keys_data[:15]  # top 15

    if keys_data:
        labels = [d["key"][:30] for d in keys_data]
        totals = [d["n_total"] for d in keys_data]
        positives = [d["n_pos"] for d in keys_data]
        eligible = [d["eligible"] for d in keys_data]

        y_pos = np.arange(len(labels))
        bar_colors = ['#2ECC71' if e else '#E74C3C' for e in eligible]

        ax.barh(y_pos, totals, color=bar_colors, alpha=0.7, label='Total')
        ax.barh(y_pos, positives, color='#3498DB', alpha=0.5, label='Positive')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Sample Count", fontsize=11)
        ax.set_title(f"{matcher_name.upper()}: Key Data Distribution\n(green=eligible, red=insufficient)", fontsize=11)
        ax.legend(loc='lower right', fontsize=9)

        # add threshold lines
        ax.axvline(x=calibrator.min_total, color='red', linestyle='--', alpha=0.5, label=f'min_total={calibrator.min_total}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved: {output_path}")

    return fig


def _plot_curve(ax, calibrator, key, color, label):
    """helper to plot a single calibration curve."""
    if key not in calibrator.calibration_curves:
        return

    bins = calibrator.calibration_curves[key]
    confs = []
    accs = []
    counts = []

    for bin_data in bins.values():
        if bin_data.sample_count > 0:
            confs.append(bin_data.predicted_confidence)
            accs.append(bin_data.actual_accuracy)
            counts.append(bin_data.sample_count)

    if confs:
        eligible = calibrator._is_eligible(key)
        linestyle = '-' if eligible else ':'
        alpha = 1.0 if eligible else 0.5

        ax.plot(confs, accs, 'o-', color=color, linewidth=2, markersize=6,
               label=label, linestyle=linestyle, alpha=alpha)


def plot_key_stats_summary(calibrator, output_path: str = "calibration_key_stats.png"):
    """plot summary of all calibration keys and their eligibility.

    args:
        calibrator: ConfidenceCalibrator
        output_path: path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # group keys by matcher
    matchers = {}
    for key, stats in calibrator.key_stats.items():
        matcher = key.split(":")[0]
        if matcher not in matchers:
            matchers[matcher] = []
        matchers[matcher].append((key, stats))

    # sort each matcher's keys by total
    for matcher in matchers:
        matchers[matcher].sort(key=lambda x: -x[1]["n_total"])

    # build combined data
    all_keys = []
    all_totals = []
    all_positives = []
    all_eligible = []
    all_colors = []

    color_map = {"biobert": "#E74C3C", "magneto": "#3498DB", "rule": "#2ECC71"}

    for matcher, items in matchers.items():
        for key, stats in items[:10]:  # top 10 per matcher
            short_key = key.replace(f"{matcher}:", f"{matcher[:3]}:")
            all_keys.append(short_key[:35])
            all_totals.append(stats["n_total"])
            all_positives.append(stats["n_pos"])
            all_eligible.append(calibrator._is_eligible(key))
            all_colors.append(color_map.get(matcher, "#666666"))

    if not all_keys:
        ax.text(0.5, 0.5, "No calibration data", ha='center', va='center')
        return fig

    y_pos = np.arange(len(all_keys))

    # plot bars
    bars = ax.barh(y_pos, all_totals, color=all_colors, alpha=0.7)

    # add eligibility markers
    for i, (eligible, bar) in enumerate(zip(all_eligible, bars)):
        if eligible:
            ax.plot(bar.get_width() + 50, i, 'g*', markersize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_keys, fontsize=8)
    ax.set_xlabel("Total Samples", fontsize=11)
    ax.set_title("Calibration Keys by Sample Count\n(* = eligible for calibration)", fontsize=12, fontweight='bold')

    # add threshold line
    ax.axvline(x=calibrator.min_total, color='red', linestyle='--', alpha=0.7,
              label=f'min_total threshold ({calibrator.min_total})')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved: {output_path}")

    return fig


def _plot_combined_key_stats(calibrators: Dict, output_path: str):
    """plot key stats from all calibrators combined."""
    fig, ax = plt.subplots(figsize=(12, 8))

    all_data = []
    color_map = {"biobert": "#E74C3C", "magneto": "#3498DB", "rule": "#2ECC71"}

    for matcher_name, calibrator in calibrators.items():
        if not hasattr(calibrator, 'key_stats'):
            continue
        for key, stats in calibrator.key_stats.items():
            # only show top-level keys (global, tier-specific)
            parts = key.split(":")
            if len(parts) <= 2 or (len(parts) == 4 and parts[2] == "*"):
                all_data.append({
                    "key": key[:40],
                    "n_total": stats["n_total"],
                    "n_pos": stats["n_pos"],
                    "matcher": matcher_name,
                    "eligible": calibrator._is_eligible(key)
                })

    if not all_data:
        ax.text(0.5, 0.5, "No key stats", ha='center', va='center')
        return

    # sort by total
    all_data.sort(key=lambda x: -x["n_total"])
    all_data = all_data[:20]

    y_pos = np.arange(len(all_data))
    colors = [color_map.get(d["matcher"], "#666") for d in all_data]
    bars = ax.barh(y_pos, [d["n_total"] for d in all_data], color=colors, alpha=0.7)

    for i, (d, bar) in enumerate(zip(all_data, bars)):
        if d["eligible"]:
            ax.plot(bar.get_width() + 20, i, 'g*', markersize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([d["key"] for d in all_data], fontsize=8)
    ax.set_xlabel("Sample Count")
    ax.set_title("Calibration Keys (* = eligible)")
    ax.axvline(x=300, color='red', linestyle='--', alpha=0.5, label='min_total=300')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved: {output_path}")


def main():
    """load calibrators and generate plots."""
    print("=" * 60)
    print("CALIBRATION PLOT GENERATOR")
    print("=" * 60)

    # load calibrators
    print("\nloading calibrators...")
    from schema_crush.orchestrator.calibration import load_calibrators

    try:
        calibrators = load_calibrators()
    except FileNotFoundError as e:
        print(f"error: {e}")
        print("\nrun training first to generate calibrators:")
        print("  python examples/demo_calibration.py")
        return

    if not calibrators:
        print("no calibrators found")
        return

    print(f"  loaded: {list(calibrators.keys())}")

    # check if any calibrator has new backoff-key attributes
    first_cal = list(calibrators.values())[0]
    has_key_stats = hasattr(first_cal, 'key_stats') and first_cal.key_stats

    # output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # generate plots
    print("\ngenerating plots...")

    # 1. main reliability diagrams (one plot per matcher using its own calibrator)
    fig, axes = plt.subplots(1, len(calibrators), figsize=(5 * len(calibrators), 5))
    if len(calibrators) == 1:
        axes = [axes]

    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    for idx, (matcher_name, calibrator) in enumerate(calibrators.items()):
        ax = axes[idx]
        color = colors[idx % len(colors)]

        # find the global key
        key = f"{matcher_name}:*:*:*" if f"{matcher_name}:*:*:*" in calibrator.calibration_curves else matcher_name
        if key not in calibrator.calibration_curves:
            ax.text(0.5, 0.5, f"No data", ha='center', va='center', transform=ax.transAxes)
            continue

        bins = calibrator.calibration_curves[key]
        confs, accs, counts = [], [], []
        for b in bins.values():
            if b.sample_count > 0:
                confs.append(b.predicted_confidence)
                accs.append(b.actual_accuracy)
                counts.append(b.sample_count)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
        if confs:
            ax.plot(confs, accs, 'o-', color=color, linewidth=2.5, markersize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Confidence")
        ax.set_ylabel("Actual Accuracy")
        stats = calibrator.get_statistics(matcher_name)
        ax.set_title(f"{matcher_name.upper()}\nAcc: {stats.get('overall_accuracy', 0):.1%} | ECE: {stats.get('calibration_error', 0):.3f}")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(str(output_dir / "calibration_reliability.png"), dpi=300, bbox_inches='tight')
    print(f"saved: {output_dir / 'calibration_reliability.png'}")

    # 2. context comparison for each matcher (using its own calibrator)
    if has_key_stats:
        for matcher_name, calibrator in calibrators.items():
            plot_context_comparison(
                calibrator,
                matcher_name=matcher_name,
                output_path=str(output_dir / f"calibration_{matcher_name}_contexts.png")
            )

        # 3. combined key stats from all calibrators
        _plot_combined_key_stats(calibrators, str(output_dir / "calibration_key_stats.png"))
    else:
        print("\nskipping context plots (need new calibrator format)")

    print("\n" + "=" * 60)
    print("PLOTS GENERATED")
    print("=" * 60)
    print(f"\nview plots in: {output_dir}")


if __name__ == '__main__':
    main()
