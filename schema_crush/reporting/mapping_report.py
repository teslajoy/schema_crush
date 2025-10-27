"""mapping report generator for schema matching results."""

from typing import Dict, List, Any
from pathlib import Path
import json
from datetime import datetime


class MappingReport:
    """generate and save schema matching reports."""

    def __init__(self, output_dir: str = "results"):
        """
        initialize report generator.

        args:
            output_dir: directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: Dict[str, Any],
        source_name: str,
        target_name: str,
        embedders_used: List[str],
        embedder_weights: Dict[str, float],
        confidence_thresholds: Dict[str, float]
    ) -> Dict[str, str]:
        """
        generate complete mapping report.

        args:
            results: output from pearl agent
            source_name: source schema name
            target_name: target schema name
            embedders_used: list of embedder names
            embedder_weights: embedder weights used
            confidence_thresholds: confidence thresholds used

        returns:
            dict with paths to generated files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{source_name}_to_{target_name}_{timestamp}"

        # generate all report files
        files = {}
        files['mappings'] = self._save_mappings(results, base_name)
        files['scores'] = self._save_scores(results, base_name)
        files['reasoning'] = self._save_reasoning(results, base_name)
        files['metadata'] = self._save_metadata(
            results, base_name, source_name, target_name,
            embedders_used, embedder_weights, confidence_thresholds
        )
        files['hitl'] = self._save_hitl_queue(results, base_name)

        # generate summary
        files['summary'] = self._save_summary(results, base_name, files)

        return files

    def _save_mappings(self, results: Dict[str, Any], base_name: str) -> str:
        """save final mappings in simple format."""
        output_path = self.output_dir / f"{base_name}_mappings.json"

        mappings = []
        for match in results['matches']:
            mappings.append({
                'source_entity': match['source_entity'],
                'source_field': match['source_field'],
                'target_entity': match['target_entity'],
                'target_field': match['target_field'],
                'score': match['score'],
                'confidence': match['confidence'],
                'tier': match['tier']
            })

        with open(output_path, 'w') as f:
            json.dump(mappings, f, indent=2)

        return str(output_path)

    def _save_scores(self, results: Dict[str, Any], base_name: str) -> str:
        """save detailed scores including individual embedder contributions."""
        output_path = self.output_dir / f"{base_name}_scores.json"

        scores = []
        for match in results['matches']:
            scores.append({
                'match': f"{match['source_entity']}.{match['source_field']} -> {match['target_entity']}.{match['target_field']}",
                'combined_score': match['score'],
                'embedder_scores': match['embedder_scores'],
                'confidence': match['confidence'],
                'tier': match['tier']
            })

        with open(output_path, 'w') as f:
            json.dump(scores, f, indent=2)

        return str(output_path)

    def _save_reasoning(self, results: Dict[str, Any], base_name: str) -> str:
        """save reasoning log from agents."""
        output_path = self.output_dir / f"{base_name}_reasoning.txt"

        with open(output_path, 'w') as f:
            f.write("schema matching reasoning log\n")
            f.write("=" * 60 + "\n\n")

            for i, reasoning in enumerate(results.get('reasoning_log', []), 1):
                f.write(f"{i}. {reasoning}\n")

        return str(output_path)

    def _save_metadata(
        self,
        results: Dict[str, Any],
        base_name: str,
        source_name: str,
        target_name: str,
        embedders_used: List[str],
        embedder_weights: Dict[str, float],
        confidence_thresholds: Dict[str, float]
    ) -> str:
        """save metadata about the matching run."""
        output_path = self.output_dir / f"{base_name}_metadata.json"

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'source_schema': source_name,
            'target_schema': target_name,
            'embedders': embedders_used,
            'embedder_weights': embedder_weights,
            'confidence_thresholds': confidence_thresholds,
            'metrics': results.get('metrics', {}),
            'tiers_executed': list(results.get('matches_by_tier', {}).keys()),
            'total_matches': len(results['matches']),
            'iteration': results.get('iteration', 0)
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return str(output_path)

    def _save_hitl_queue(self, results: Dict[str, Any], base_name: str) -> str:
        """save hitl queue for human review."""
        output_path = self.output_dir / f"{base_name}_hitl_queue.json"

        hitl_items = []
        for match in results.get('hitl_queue', []):
            hitl_items.append({
                'match_id': match['match_id'],
                'source': f"{match['source_entity']}.{match['source_field']}",
                'target': f"{match['target_entity']}.{match['target_field']}",
                'score': match['score'],
                'confidence': match['confidence'],
                'embedder_scores': match['embedder_scores'],
                'reasoning': match.get('reasoning', '')
            })

        with open(output_path, 'w') as f:
            json.dump(hitl_items, f, indent=2)

        return str(output_path)

    def _save_summary(
        self,
        results: Dict[str, Any],
        base_name: str,
        files: Dict[str, str]
    ) -> str:
        """save human-readable summary."""
        output_path = self.output_dir / f"{base_name}_summary.txt"

        matches = results['matches']
        metrics = results.get('metrics', {})

        with open(output_path, 'w') as f:
            f.write("schema matching summary\n")
            f.write("=" * 60 + "\n\n")

            # overview
            f.write("overview:\n")
            f.write(f"  total matches: {len(matches)}\n")
            f.write(f"  requires hitl: {len(results.get('hitl_queue', []))}\n")
            f.write(f"  iteration: {results.get('iteration', 0)}\n\n")

            # confidence distribution
            high = [m for m in matches if m['confidence'] == 'high']
            medium = [m for m in matches if m['confidence'] == 'medium']
            low = [m for m in matches if m['confidence'] == 'low']

            f.write("confidence distribution:\n")
            f.write(f"  high (>=95%): {len(high)}\n")
            f.write(f"  medium (85-95%): {len(medium)}\n")
            f.write(f"  low (<85%): {len(low)}\n\n")

            # tier distribution
            by_tier = {}
            for match in matches:
                tier = match['tier']
                by_tier[tier] = by_tier.get(tier, 0) + 1

            f.write("tier distribution:\n")
            for tier, count in by_tier.items():
                f.write(f"  {tier}: {count}\n")
            f.write("\n")

            # top matches
            f.write("top 10 matches:\n")
            sorted_matches = sorted(matches, key=lambda m: m['score'], reverse=True)
            for i, match in enumerate(sorted_matches[:10], 1):
                f.write(f"  {i}. {match['source_entity']}.{match['source_field']} -> ")
                f.write(f"{match['target_entity']}.{match['target_field']}\n")
                f.write(f"     score: {match['score']:.4f}, confidence: {match['confidence']}, tier: {match['tier']}\n")
            f.write("\n")

            # files generated
            f.write("generated files:\n")
            for file_type, file_path in files.items():
                if file_type != 'summary':
                    f.write(f"  {file_type}: {file_path}\n")

        return str(output_path)