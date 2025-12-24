"""confidence calibration for matchers based on historical accuracy.

the problem:
- BioBERT gives 0.85 confidence but is only correct 5% of the time (overconfident)
- Magneto gives 0.53 confidence and is correct 25% of the time (well-calibrated)
- RuleMatcher gives 0.90 uniform but is only correct 15% of the time (was due to the db structure not the rules)

solution:
- use flat mapping database as calibration ground truth
- measure actual accuracy at each confidence level
- build calibration curves to adjust raw scores
- support tier-aware calibration (entity vs field vs content)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class CalibrationBin:
    """statistics for a confidence bin."""
    confidence_range: Tuple[float, float]
    predicted_confidence: float
    actual_accuracy: float
    sample_count: int


class ConfidenceCalibrator:
    """calibrate matcher confidence scores based on historical accuracy.

    supports tier-aware calibration where each tier (entity, field, content)
    has separate calibration curves.
    """

    def __init__(self, num_bins: int = 10):
        """initialize calibrator.

        args:
            num_bins: number of confidence bins (default 10 for 0.0-0.1, 0.1-0.2, ...)
        """
        self.num_bins = num_bins
        self.calibration_data = {}  # regular dict for pickling
        self.calibration_curves = {}

    def _make_key(self, matcher_name: str, tier: Optional[str] = None) -> str:
        """build consistent key for calibration data/curves.

        args:
            matcher_name: name of matcher ("biobert", "magneto", "rule")
            tier: optional tier (entity, field, content)

        returns:
            key string: "matcher_name:tier" if tier provided, else "matcher_name"
        """
        return f"{matcher_name}:{tier}" if tier else matcher_name

    def add_prediction(
        self,
        matcher_name: str,
        confidence: float,
        was_correct: bool,
        metadata: Dict[str, Any] = None,
        tier: Optional[str] = None,
    ):
        """record a prediction outcome for calibration.

        args:
            matcher_name: name of matcher ("biobert", "magneto", "rule")
            confidence: raw confidence score from matcher (0.0-1.0)
            was_correct: whether the prediction was correct
            metadata: optional metadata (source, target, ...)
            tier: optional tier for tier-aware calibration (entity, field, content)
        """
        key = self._make_key(matcher_name, tier)

        # initialize if needed (no defaultdict for pickling)
        if key not in self.calibration_data:
            self.calibration_data[key] = {
                "confidences": [],
                "outcomes": [],
                "metadata": [],
            }

        self.calibration_data[key]["confidences"].append(confidence)
        self.calibration_data[key]["outcomes"].append(1.0 if was_correct else 0.0)
        if metadata:
            self.calibration_data[key]["metadata"].append(metadata)

    def calibrate(self, key: str) -> Dict[str, CalibrationBin]:
        """build calibration curve for a key (matcher or matcher:tier).

        mathematical formulation:
            divide confidence range [0, 1] into B bins (default B=10)
            for bin i with range [b_i, b_{i+1}):
                - for i < B-1: collect predictions where b_i ≤ confidence < b_{i+1}
                - for i = B-1: collect predictions where b_{B-1} ≤ confidence ≤ 1.0 (includes 100% confidence)
                - predicted_confidence_i = mean(confidences in bin i)
                - actual_accuracy_i = mean(outcomes in bin i)
                                    = (# correct predictions in bin i) / (# samples in bin i)

        args:
            key: calibration key (matcher_name or matcher_name:tier)

        returns:
            dict mapping bin_id -> CalibrationBin with accuracy statistics
        """
        if key not in self.calibration_data:
            raise ValueError(f"no calibration data for key: {key}")

        confidences = np.array(self.calibration_data[key]["confidences"])
        outcomes = np.array(self.calibration_data[key]["outcomes"])

        bins = {}
        bin_size = 1.0 / self.num_bins

        for i in range(self.num_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size

            # find predictions in this confidence range
            # last bin includes upper boundary to handle confidence = 1.0
            if i == self.num_bins - 1:
                mask = (confidences >= bin_start) & (confidences <= bin_end)
            else:
                mask = (confidences >= bin_start) & (confidences < bin_end)
            bin_outcomes = outcomes[mask]

            if len(bin_outcomes) > 0:
                actual_accuracy = bin_outcomes.mean()
                avg_confidence = confidences[mask].mean()
            else:
                # empty bins handled in calibrate_score() by returning raw_score
                actual_accuracy = 0.0
                avg_confidence = (bin_start + bin_end) / 2

            bins[i] = CalibrationBin(
                confidence_range=(bin_start, bin_end),
                predicted_confidence=avg_confidence,
                actual_accuracy=actual_accuracy,
                sample_count=len(bin_outcomes)
            )

        self.calibration_curves[key] = bins
        return bins

    def calibrate_score(
        self,
        matcher_name: str,
        raw_score: float,
        tier: Optional[str] = None,
    ) -> float:
        """calibrate a raw confidence score using learned curve.

        mathematical formulation:
            given raw_score s, find bin i where s in [b_i, b_{i+1})
            calibrated_score = actual_accuracy_i

            this maps the raw confidence to the empirically observed accuracy
            for predictions at that confidence level.

        args:
            matcher_name: name of matcher
            raw_score: raw confidence score (0.0-1.0)
            tier: optional tier for tier-aware calibration

        returns:
            calibrated confidence score reflecting actual accuracy
        """
        # try tier-specific curve first, fallback to pooled
        key = self._make_key(matcher_name, tier)
        if key not in self.calibration_curves:
            key = matcher_name
        if key not in self.calibration_curves:
            return raw_score

        # find which bin this score falls into
        bin_idx = min(int(raw_score * self.num_bins), self.num_bins - 1)
        bin_data = self.calibration_curves[key].get(bin_idx)

        if bin_data and bin_data.sample_count > 0:
            return bin_data.actual_accuracy
        else:
            return raw_score

    def calibrate_all(self) -> None:
        """build calibration curves for all keys (matchers and tier-specific)."""
        for key in self.calibration_data:
            self.calibrate(key)

    def get_statistics(
        self,
        matcher_name: str,
        tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """get calibration statistics for a matcher.

        mathematical formulations:

        1. overall accuracy:
            accuracy = (1/N) * Σ outcomes_j
                     = (# correct predictions) / (total predictions)
            where outcomes_j ∈ {0, 1} for each prediction j

        2. Expected Calibration Error (ECE):
            ECE = Σ_i (n_i / N) * |conf_i - acc_i|

            where:
              i = bin index
              n_i = number of samples in bin i
              N = total number of samples
              conf_i = mean predicted confidence in bin i
              acc_i = actual accuracy in bin i (fraction correct)

            ECE measures the weighted average difference between predicted
            confidence and actual accuracy across all bins. Lower is better
            (0 = perfect calibration).

        args:
            matcher_name: name of matcher
            tier: optional tier for tier-specific statistics

        returns:
            dict with overall metrics and per-bin statistics
        """
        key = self._make_key(matcher_name, tier)

        if key not in self.calibration_data:
            return {}

        confidences = np.array(self.calibration_data[key]["confidences"])
        outcomes = np.array(self.calibration_data[key]["outcomes"])

        stats = {
            "total_predictions": len(outcomes),
            "overall_accuracy": outcomes.mean(),
            "mean_confidence": confidences.mean(),
            "calibration_error": 0.0,
            "bins": []
        }

        # calculate Expected Calibration Error (ECE)
        if key in self.calibration_curves:
            ece = 0.0
            total_samples = len(outcomes)

            for bin_data in self.calibration_curves[key].values():
                if bin_data.sample_count > 0:
                    bin_weight = bin_data.sample_count / total_samples
                    ece += bin_weight * abs(bin_data.predicted_confidence - bin_data.actual_accuracy)

                    stats["bins"].append({
                        "range": f"{bin_data.confidence_range[0]:.1f}-{bin_data.confidence_range[1]:.1f}",
                        "avg_confidence": f"{bin_data.predicted_confidence:.3f}",
                        "actual_accuracy": f"{bin_data.actual_accuracy:.3f}",
                        "samples": bin_data.sample_count,
                        "calibration_gap": f"{bin_data.predicted_confidence - bin_data.actual_accuracy:+.3f}"
                    })

            stats["calibration_error"] = ece

        return stats

    def print_calibration_report(
        self,
        matcher_name: str,
        tier: Optional[str] = None,
    ):
        """print calibration report.

        args:
            matcher_name: name of matcher
            tier: optional tier for tier-specific report
        """
        key = self._make_key(matcher_name, tier)
        stats = self.get_statistics(matcher_name, tier)

        if not stats:
            print(f"no calibration data for {key}")
            return

        print(f"\n{'='*70}")
        print(f"CALIBRATION REPORT: {key}")
        print(f"{'='*70}")

        print(f"\nOverall Statistics:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Overall accuracy: {stats['overall_accuracy']:.1%}")
        print(f"  Mean confidence: {stats['mean_confidence']:.3f}")
        print(f"  Expected Calibration Error (ECE): {stats['calibration_error']:.3f}")

        if stats['bins']:
            print(f"\nPer-Bin Analysis:")
            print(f"  {'Range':<12} {'Avg Conf':<12} {'Accuracy':<12} {'Samples':<10} {'Gap':<10}")
            print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

            for bin_data in stats['bins']:
                if bin_data['samples'] > 0:
                    print(f"  {bin_data['range']:<12} "
                          f"{bin_data['avg_confidence']:<12} "
                          f"{bin_data['actual_accuracy']:<12} "
                          f"{bin_data['samples']:<10} "
                          f"{bin_data['calibration_gap']:<10}")

        print(f"\nInterpretation:")
        if stats['calibration_error'] < 0.05:
            print(f"= Well-calibrated (ECE < 0.05)")
        elif stats['calibration_error'] < 0.15:
            print(f"= Moderately calibrated (0.05 ≤ ECE < 0.15)")
        else:
            print(f"= Poorly calibrated (ECE ≥ 0.15) - needs adjustment")

        print(f"{'='*70}\n")


def create_calibrator_from_flat_db(
    matcher,
    flat_db: "FlatMappingDatabase",
    target_candidates: List[str],
    max_samples: int = None,
    tier_aware: bool = True,
    include_content: bool = True,
) -> ConfidenceCalibrator:
    """create calibrator by evaluating matcher on flat mapping database.

    uses flat mappings as ground truth for calibration. supports tier-aware
    calibration where entity, field, and content tiers get separate curves.

    args:
        matcher: matcher instance (BioBERTMatcher, MagnetoMatcher, RuleMatcher)
        flat_db: FlatMappingDatabase with ground truth mappings
        target_candidates: list of candidate targets to score against
        max_samples: optional limit on number of sources to evaluate
        tier_aware: if true, build separate calibration curves per tier
        include_content: if true, also calibrate on content_values table

    returns:
        ConfidenceCalibrator with learned calibration curves
    """
    import random

    calibrator = ConfidenceCalibrator()
    matcher_name = matcher.__class__.__name__.lower().replace("matcher", "")

    # get all sources from flat db
    sources = list(flat_db.sources.values())

    if max_samples and len(sources) > max_samples:
        sources = random.sample(sources, max_samples)

    print(f"\ncalibrating {matcher_name} on {len(sources)} sources from flat db...")

    for i, src in enumerate(sources):
        # get ground truth destinations for this source
        dests = flat_db._dest_by_source.get(src.id, [])
        if not dests:
            continue

        ground_truths = {d.destination.lower() for d in dests}

        # get matcher predictions
        try:
            results = matcher.match(src.source, target_candidates)

            if results:
                top_target, top_score = results[0]
                was_correct = top_target.lower() in ground_truths

                # add to pooled calibration (all tiers) for fallback
                calibrator.add_prediction(
                    matcher_name=matcher_name,
                    confidence=top_score,
                    was_correct=was_correct,
                    metadata={
                        "source": src.source,
                        "context": src.source_context,
                        "predicted": top_target,
                        "ground_truths": list(ground_truths),
                    },
                )

                # add to tier-specific calibration
                if tier_aware:
                    calibrator.add_prediction(
                        matcher_name=matcher_name,
                        confidence=top_score,
                        was_correct=was_correct,
                        tier=src.tier.value,
                        metadata={
                            "source": src.source,
                            "context": src.source_context,
                            "predicted": top_target,
                            "ground_truths": list(ground_truths),
                        },
                    )

                if (i + 1) % 100 == 0:
                    print(f"  processed {i + 1}/{len(sources)} sources...")

        except Exception as e:
            print(f"  warning: failed to evaluate source {src.source}: {e}")
            continue

    # calibrate on content_values table (SNOMED code prediction)
    if include_content and flat_db.content_values:
        _calibrate_content_values(
            calibrator=calibrator,
            matcher=matcher,
            matcher_name=matcher_name,
            flat_db=flat_db,
            max_samples=max_samples,
        )

    # build all calibration curves
    calibrator.calibrate_all()

    return calibrator


def _calibrate_content_values(
    calibrator: ConfidenceCalibrator,
    matcher,
    matcher_name: str,
    flat_db: "FlatMappingDatabase",
    max_samples: int = None,
) -> None:
    """calibrate matcher on content_values table (source_value -> FHIR path/code).

    evaluates how well matcher predicts FHIR paths for content values like
    "Adenocarcinoma" -> "Condition.code" or "G1" -> "Observation.valueCodeableConcept".

    args:
        calibrator: ConfidenceCalibrator to add predictions to
        matcher: matcher instance
        matcher_name: name for calibration keys
        flat_db: FlatMappingDatabase with content_values
        max_samples: optional limit on samples
    """
    import random

    content_values = flat_db.content_values
    if not content_values:
        return

    if max_samples and len(content_values) > max_samples:
        content_values = random.sample(content_values, max_samples)

    # build candidate set from all unique FHIR paths in content_fhir_targets
    fhir_path_candidates = list(set(t.fhir_path for t in flat_db.content_fhir_targets))

    print(f"\ncalibrating {matcher_name} on {len(content_values)} content values...")
    print(f"  candidate FHIR paths: {len(fhir_path_candidates)}")

    for i, cv in enumerate(content_values):
        # get ground truth FHIR paths for this content value
        targets = [t for t in flat_db.content_fhir_targets if t.content_value_id == cv.id]
        if not targets:
            continue

        ground_truth_paths = {t.fhir_path.lower() for t in targets}

        # get matcher predictions
        try:
            results = matcher.match(cv.source_value, fhir_path_candidates)

            if results:
                top_path, top_score = results[0]
                was_correct = top_path.lower() in ground_truth_paths

                # add to pooled calibration
                calibrator.add_prediction(
                    matcher_name=matcher_name,
                    confidence=top_score,
                    was_correct=was_correct,
                    metadata={
                        "source_value": cv.source_value,
                        "category": cv.source_category,
                        "code": cv.code,
                        "predicted": top_path,
                        "ground_truths": list(ground_truth_paths),
                    },
                )

                # add to content tier calibration
                calibrator.add_prediction(
                    matcher_name=matcher_name,
                    confidence=top_score,
                    was_correct=was_correct,
                    tier="content",
                    metadata={
                        "source_value": cv.source_value,
                        "category": cv.source_category,
                        "code": cv.code,
                        "predicted": top_path,
                        "ground_truths": list(ground_truth_paths),
                    },
                )

                # add category-specific calibration (e.g., "content:histology")
                calibrator.add_prediction(
                    matcher_name=matcher_name,
                    confidence=top_score,
                    was_correct=was_correct,
                    tier=f"content:{cv.source_category}",
                    metadata={
                        "source_value": cv.source_value,
                        "category": cv.source_category,
                        "code": cv.code,
                        "predicted": top_path,
                        "ground_truths": list(ground_truth_paths),
                    },
                )

                if (i + 1) % 200 == 0:
                    print(f"  processed {i + 1}/{len(content_values)} content values...")

        except Exception as e:
            print(f"  warning: failed on content value {cv.source_value}: {e}")
            continue


# -----------------------------------------------------------------------------
# legacy calibrator creator - uses old hierarchical MappingRule objects
# -----------------------------------------------------------------------------
def create_calibrator_from_mappings(
    matcher,
    mapping_rules: List,
    target_schema: Dict[str, List[str]],
    max_samples: int = None,
) -> ConfidenceCalibrator:
    """create calibrator by evaluating matcher on hierarchical mapping rules.

    legacy function for backwards compatibility. prefer create_calibrator_from_flat_db.

    args:
        matcher: matcher instance (BioBERTMatcher, MagnetoMatcher, etc.)
        mapping_rules: list of MappingRule objects with ground truth
        target_schema: dict of {resource: [fields]} for candidate generation
        max_samples: optional limit on number of rules to evaluate

    returns:
        ConfidenceCalibrator with learned calibration curve
    """
    calibrator = ConfidenceCalibrator()
    matcher_name = matcher.__class__.__name__.lower().replace("matcher", "")

    # sample rules if max_samples specified
    if max_samples and len(mapping_rules) > max_samples:
        import random
        mapping_rules = random.sample(mapping_rules, max_samples)

    print(f"\ncalibrating {matcher_name} on {len(mapping_rules)} mapping rules...")

    for i, rule in enumerate(mapping_rules):
        source = rule.source_node
        ground_truth = f"{rule.target_resource}.{rule.field_mappings[0].field}" if rule.field_mappings else rule.target_resource

        # generate candidate targets from schema
        candidates = []
        for resource, fields in target_schema.items():
            candidates.append(resource)
            for field in fields:
                candidates.append(f"{resource}.{field}")

        # ensure ground truth is in candidates
        if ground_truth not in candidates:
            candidates.append(ground_truth)

        # get matcher predictions
        try:
            results = matcher.match(source, candidates)

            if results:
                top_target, top_score = results[0]
                was_correct = (top_target == ground_truth)

                calibrator.add_prediction(
                    matcher_name=matcher_name,
                    confidence=top_score,
                    was_correct=was_correct,
                    metadata={
                        "source": source,
                        "predicted": top_target,
                        "ground_truth": ground_truth,
                    },
                )

                if (i + 1) % 100 == 0:
                    print(f"  processed {i + 1}/{len(mapping_rules)} rules...")

        except Exception as e:
            print(f"  warning: failed to evaluate rule {i}: {e}")
            continue

    # build calibration curve
    calibrator.calibrate(matcher_name)

    return calibrator


def load_calibrators(
    calibrators_dir: str = None,
    use_flat: bool = True,
) -> Dict[str, 'ConfidenceCalibrator']:
    """load pre-trained calibrators from disk.

    args:
        calibrators_dir: path to calibrators directory (defaults to package calibrators/)
        use_flat: if true, prefer flat calibrators (*_flat_calibrator.pkl) over legacy

    returns:
        dict mapping {matcher_name: calibrator}
    """
    import pickle
    from pathlib import Path

    if calibrators_dir is None:
        # default to package calibrators/ directory
        import schema_crush
        package_dir = Path(schema_crush.__file__).parent.parent
        calibrators_dir = package_dir / "calibrators"
    else:
        calibrators_dir = Path(calibrators_dir)

    if not calibrators_dir.exists():
        raise FileNotFoundError(f"calibrators directory not found: {calibrators_dir}")

    # look in pkl/ subdirectory first, fallback to root for backwards compatibility
    pkl_dir = calibrators_dir / "pkl"
    search_dir = pkl_dir if pkl_dir.exists() else calibrators_dir

    calibrators = {}

    if use_flat:
        # prefer flat calibrators (tier-aware)
        for pkl_file in search_dir.glob("*_flat_calibrator.pkl"):
            matcher_name = pkl_file.stem.replace("_flat_calibrator", "")
            with open(pkl_file, 'rb') as f:
                calibrators[matcher_name] = pickle.load(f)

    # fallback to legacy calibrators if flat not found
    if not calibrators:
        for pkl_file in search_dir.glob("*_calibrator.pkl"):
            if "_flat_" not in pkl_file.stem:
                matcher_name = pkl_file.stem.replace("_calibrator", "")
                with open(pkl_file, 'rb') as f:
                    calibrators[matcher_name] = pickle.load(f)

    return calibrators


# -----------------------------------------------------------------------------
# legacy calibrator loader - loads old hierarchical calibrators
# -----------------------------------------------------------------------------
def load_legacy_calibrators(calibrators_dir: str = None) -> Dict[str, 'ConfidenceCalibrator']:
    """load legacy (non-flat) calibrators from disk.

    args:
        calibrators_dir: path to calibrators directory

    returns:
        dict mapping {matcher_name: calibrator}
    """
    return load_calibrators(calibrators_dir, use_flat=False)