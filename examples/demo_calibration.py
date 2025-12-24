"""demonstrate confidence calibration using existing mapping rules.

this shows how to:
1. load existing mapping rules (GDC + HTAN)
2. evaluate matchers on these known mappings
3. build calibration curves
4. use calibrated scores for better predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_crush.knowledge.mapping_rules import load_gdc_rules, load_htan_rules
from schema_crush.tools.matchers import BioBERTMatcher, MagnetoMatcher, RuleMatcher
from schema_crush.orchestrator.calibration import ConfidenceCalibrator, create_calibrator_from_mappings
from schema_crush.orchestrator.schema_utils import load_fhir_schema_subset


# load FHIR schema from LinkML definition (all fields + required fields prioritized)
print("loading FHIR schema from LinkML...")
FHIR_SCHEMA = load_fhir_schema_subset(required_fields=True)
total_fields = sum(len(fields) for fields in FHIR_SCHEMA.values())
print(f"loaded {len(FHIR_SCHEMA)} resources, {total_fields} total fields (required fields prioritized)")


def demo_biobert_calibration():
    """calibrate BioBERT matcher on known mappings."""

    print("="*70)
    print("BIOBERT CALIBRATION")
    print("="*70)

    # load mapping rules
    print("\nloading mapping rules...")
    gdc_rules = load_gdc_rules()
    print(f"loaded {len(gdc_rules)} GDC rules")

    # create matcher
    print("\ninitializing BioBERT matcher...")
    biobert = BioBERTMatcher()

    # calibrate (use subset for speed)
    calibrator = create_calibrator_from_mappings(
        matcher=biobert,
        mapping_rules=gdc_rules[:50],  # use first 50 rules for demo
        target_schema=FHIR_SCHEMA,
        max_samples=50
    )

    # show calibration report
    calibrator.print_calibration_report("biobert")

    # demonstrate calibration
    print("\nDemonstrating Calibration:")
    print("-"*70)

    raw_scores = [0.50, 0.70, 0.85, 0.90, 0.95]
    print(f"{'Raw Score':<15} {'Calibrated Score':<20} {'Meaning':<40}")
    print(f"{'-'*15} {'-'*20} {'-'*40}")

    for raw_score in raw_scores:
        calibrated = calibrator.calibrate_score("biobert", raw_score)
        print(f"{raw_score:<15.2f} {calibrated:<20.2f} "
              f"{'(actual accuracy at this confidence level)'}")

    return calibrator


def demo_magneto_calibration():
    """calibrate Magneto matcher on known mappings."""

    print("\n\n" + "="*70)
    print("MAGNETO CALIBRATION")
    print("="*70)

    # load mapping rules
    print("\nloading mapping rules...")
    gdc_rules = load_gdc_rules()

    # create matcher
    print("\ninitializing Magneto matcher...")
    magneto = MagnetoMatcher()

    # calibrate
    calibrator = create_calibrator_from_mappings(
        matcher=magneto,
        mapping_rules=gdc_rules[:50],
        target_schema=FHIR_SCHEMA,
        max_samples=50
    )

    # show calibration report
    calibrator.print_calibration_report("magneto")

    return calibrator


def demo_rule_calibration():
    """calibrate Rule matcher on known mappings."""

    print("\n\n" + "="*70)
    print("RULE MATCHER CALIBRATION")
    print("="*70)

    # load mapping rules
    print("\nloading mapping rules...")
    gdc_rules = load_gdc_rules()
    htan_rules = load_htan_rules()

    # create matcher with rule database
    from schema_crush.knowledge.mapping_rules import RuleDatabase
    db = RuleDatabase()
    db.add_rules(gdc_rules)
    db.add_rules(htan_rules)

    print(f"rule database: {len(db.rules)} total rules")

    print("\ninitializing Rule matcher...")
    rule_matcher = RuleMatcher(db)

    # calibrate
    calibrator = create_calibrator_from_mappings(
        matcher=rule_matcher,
        mapping_rules=gdc_rules[:50],
        target_schema=FHIR_SCHEMA,
        max_samples=50
    )

    # show calibration report
    calibrator.print_calibration_report("rule")

    return calibrator


def compare_calibrations():
    """compare all three matchers before and after calibration."""

    print("\n\n" + "="*70)
    print("CALIBRATION COMPARISON")
    print("="*70)

    print("""
Expected Results (based on architecture.md evaluation):

Matcher   | Raw Performance      | After Calibration
----------|---------------------|-------------------
BioBERT   | 5% accuracy at 0.90 | Scores adjusted down to reflect real accuracy
          | OVERCONFIDENT       | e.g., 0.85 → 0.15 (actual accuracy)
          |                     |
Magneto   | 25% accuracy at 0.53| Already well-calibrated
          | WELL-CALIBRATED     | Minor adjustments only
          |                     |
Rule      | 15% accuracy at 0.90| Scores adjusted based on rule quality
          | UNIFORM SCORES      | Differentiate high vs low quality rules

Key Insight:
- BioBERT will show LARGE calibration gaps (predicted vs actual)
- Magneto will show SMALL calibration gaps (already calibrated)
- Rule matcher needs to learn which rules are high quality

Usage in Production:
- Raw scores are misleading
- Always use calibrated scores for decision thresholds
- Auto-accept threshold should be based on calibrated scores
  (e.g., calibrated > 0.95 means truly 95%+ accurate)
""")


def demonstrate_production_usage():
    """show how to use calibrated scores in production."""

    print("\n\n" + "="*70)
    print("PRODUCTION USAGE EXAMPLE")
    print("="*70)

    print("""
Workflow with Calibration:

1. TRAINING PHASE (one-time):
   - Evaluate matchers on 1,280 known mappings
   - Build calibration curves for each matcher
   - Save calibration models to disk

2. PRODUCTION MAPPING:
   - Load calibrated models
   - For each field mapping task:
     a) Get raw scores from matchers
     b) Apply calibration: calibrated = calibrator.calibrate_score(matcher, raw)
     c) Make decision based on CALIBRATED scores

3. DECISION THRESHOLDS (using calibrated scores):
   - calibrated >= 0.95: auto-accept (truly 95%+ accurate)
   - 0.85 <= calibrated < 0.95: human review (HITL)
   - calibrated < 0.85: reject or escalate

4. CONTINUOUS LEARNING:
   - When humans validate/correct predictions:
     - Add to calibration dataset
     - Periodically re-calibrate
     - Improves over time

Example Code:

```python
# at startup: load calibrators
biobert_calibrator = load_calibrator("biobert_calibration.pkl")
magneto_calibrator = load_calibrator("magneto_calibration.pkl")

# for each mapping task:
biobert_raw = biobert.match(source, candidates)[0][1]
magneto_raw = magneto.match(source, candidates)[0][1]

biobert_calibrated = biobert_calibrator.calibrate_score("biobert", biobert_raw)
magneto_calibrated = magneto_calibrator.calibrate_score("magneto", magneto_raw)

# combine calibrated scores
final_score = 0.4 * biobert_calibrated + 0.6 * magneto_calibrated

# make decision
if final_score >= 0.95:
    auto_accept()
elif final_score >= 0.85:
    request_human_review()
else:
    reject_or_escalate()
```
""")


if __name__ == '__main__':
    # demo 1: calibrate BioBERT
    biobert_cal = demo_biobert_calibration()

    # demo 2: calibrate Magneto
    magneto_cal = demo_magneto_calibration()

    # demo 3: calibrate Rule matcher
    rule_cal = demo_rule_calibration()

    # demo 4: compare all three
    compare_calibrations()

    # demo 5: show production usage
    demonstrate_production_usage()

    print("\n" + "="*70)
    print("CALIBRATION DEMO COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Run calibration on FULL dataset (1,280 rules)")
    print("2. Save calibrators to disk (pickle)")
    print("3. Integrate into ClaudeAgent workflow")
    print("4. Test on real HTAN/TCGA data")
    print("="*70)