#!/usr/bin/env python
"""map csv columns to fhir fields using claude agent."""

import argparse
import csv
import json
import sys
from pathlib import Path

ENTITY_RESOURCES = {
    "patient": ["Patient", "Condition", "Observation"],
    "sample": ["Specimen", "Observation"],
    "file": ["DocumentReference", "Observation"],
}

def main():
    parser = argparse.ArgumentParser(description='Map CSV columns to FHIR fields')
    parser.add_argument('csv_path', help='Path to CSV file')
    parser.add_argument('--entity', '-e', choices=['patient', 'sample', 'file'],
                        required=True, help='Entity type: patient, sample, or file')
    parser.add_argument('--output', '-o', help='Output JSON path (default: stdout)')
    parser.add_argument('--limit', '-n', type=int, help='Limit columns to map')
    parser.add_argument('--no-profile', action='store_true',
                        help='Disable CSV profiling (skip use-case detection)')
    parser.add_argument('--no-llm-profile', action='store_true',
                        help='Use heuristic profiling only (no LLM call for profiler)')
    args = parser.parse_args()

    # check file exists
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # read csv headers
    with open(csv_path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        # get sample values (first 5 rows)
        sample_values = {h: [] for h in headers}
        for i, row in enumerate(reader):
            if i >= 5:
                break
            for j, val in enumerate(row):
                if j < len(headers) and val:
                    sample_values[headers[j]].append(val)

    print(f"found {len(headers)} columns in {csv_path.name}", file=sys.stderr)
    print(f"entity: {args.entity} -> {ENTITY_RESOURCES[args.entity]}", file=sys.stderr)

    # profile csv for context-aware mapping
    csv_profile = None
    if not args.no_profile:
        from schema_crush.orchestrator.agents.csv_profiler import CSVProfiler
        use_llm = not args.no_llm_profile
        profiler = CSVProfiler(use_llm=use_llm)
        csv_profile = profiler.profile(headers, sample_values)
        print(f"csv profile: use_case={csv_profile.use_case}, entity={csv_profile.primary_entity}", file=sys.stderr)
        print(f"analysis: {csv_profile.analysis_purpose}", file=sys.stderr)
        if csv_profile.column_groups:
            print(f"column groups: {list(csv_profile.column_groups.keys())}", file=sys.stderr)

    # limit if requested
    if args.limit:
        headers = headers[:args.limit]

    # initialize agent
    from schema_crush.orchestrator.agents import ClaudeAgent
    agent = ClaudeAgent(task="field")

    # get target resources for this entity
    target_resources = ENTITY_RESOURCES[args.entity]

    # map each column
    results = []
    for col in headers:
        samples = sample_values.get(col, [])[:3]
        context = {
            "entity": args.entity,
            "target_resources": target_resources,
            "sample_values": samples,
            "csv_profile": csv_profile,  # pass profile for context-aware mapping
        }

        print(f"mapping: {col}...", file=sys.stderr)
        proposals = agent.propose_mappings(col, candidate_targets=None, context=context)

        if proposals:
            p = proposals[0]
            results.append({
                "source": col,
                "target": p.target_field,
                "confidence": round(p.confidence, 3),
                "reasoning": p.reasoning[:200] if p.reasoning else ""
            })
        else:
            results.append({"source": col, "target": None, "confidence": 0})

    # output
    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"saved to {args.output}", file=sys.stderr)
    else:
        print(output)

if __name__ == '__main__':
    main()
