"""loader for fhir aggregator ndjson data.

extracts patterns from real fhir data:
- field paths actually used
- coding systems and codes
- reference patterns between resources

uses streaming approach - never loads all records into memory.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Iterator
from collections import defaultdict


@dataclass
class FhirFieldUsage:
    """tracks how a fhir field is used across records."""
    path: str                           # ex. "Specimen.type.coding.code"
    resource_type: str                  # ex. "Specimen"
    count: int = 0                      # how many records use this field
    sample_values: list = field(default_factory=list)  # sample values (max 10)

    def add_value(self, value):
        self.count += 1
        if len(self.sample_values) < 10 and value not in self.sample_values:
            self.sample_values.append(value)


@dataclass
class FhirCodingUsage:
    """tracks coding system usage."""
    system: str                         # ex. "https://cadsr.cancer.gov/sample_type"
    code: str                           # ex. "3111302"
    display: str = ""                   # ex. "Blood Derived Normal"
    resource_type: str = ""             # ex. "Specimen"
    field_path: str = ""                # ex. "Specimen.type.coding"
    count: int = 0


@dataclass
class FhirReferenceUsage:
    """tracks reference patterns between resources."""
    source_resource: str                # ex. "Specimen"
    source_field: str                   # ex. "subject"
    target_resource: str                # ex. "Patient"
    count: int = 0


@dataclass
class SourceDestinationPair:
    """extracted source->destination mapping pair."""
    source_term: str                    # ex. "sample_id", "Blood Derived Normal"
    source_schema: str                  # ex. "gdc", "cadsr"
    destination_path: str               # ex. "Specimen.identifier", "Specimen.type.coding"
    destination_resource: str           # ex. "Specimen"
    pair_type: str                      # "field" or "content" (legacy)
    tier: str = "field"                 # "entity", "field", or "content"
    dest_system: str = ""               # coding system if content
    dest_code: str = ""                 # code if content
    count: int = 0

    def __post_init__(self):
        """infer tier from pair_type and destination_path."""
        if self.pair_type == "content":
            self.tier = "content"
        elif '.' not in self.destination_path or self.destination_path == self.destination_resource:
            # just resource name, no field path
            self.tier = "entity"
        else:
            self.tier = "field"


class FhirAggregatorLoader:
    """streams fhir aggregator ndjson files and extracts patterns.

    never loads all data into memory - processes line by line.
    """

    def __init__(self, base_path: str):
        """initialize loader.

        args:
            base_path: path to fhir aggregator data (ex. /path/to/all/)
        """
        self.base_path = Path(base_path)

        # aggregated stats (small - just counts and samples)
        self.field_usage: dict[str, FhirFieldUsage] = {}
        self.coding_usage: dict[str, FhirCodingUsage] = {}
        self.reference_usage: dict[str, FhirReferenceUsage] = {}
        self.resource_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # source->destination pairs (the key training data!)
        self.mapping_pairs: dict[str, SourceDestinationPair] = {}

    def list_sources(self) -> list[str]:
        """list available data sources."""
        sources = []
        for d in self.base_path.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                sources.append(d.name)
        return sorted(sources)

    def list_resources(self, source: str) -> list[str]:
        """list available resource types for a source."""
        meta_path = self.base_path / source / "META"
        if not meta_path.exists():
            return []

        resources = []
        for f in meta_path.glob("*.ndjson"):
            resources.append(f.stem)
        return sorted(resources)

    def count_records(self, source: str, resource_type: str) -> int:
        """count records in a file without loading all."""
        ndjson_path = self.base_path / source / "META" / f"{resource_type}.ndjson"
        if not ndjson_path.exists():
            return 0

        count = 0
        with open(ndjson_path, 'r') as f:
            for _ in f:
                count += 1
        return count

    def stream_records(self, source: str, resource_type: str,
                       limit: Optional[int] = None,
                       sample_every: int = 1) -> Iterator[dict]:
        """stream ndjson records one at a time.

        args:
            source: source name (ex. "GDC")
            resource_type: fhir resource type (ex. "Specimen")
            limit: max records to read (None = all)
            sample_every: sample every nth record (1 = all, 10 = every 10th)

        yields:
            parsed json records (one at a time, not stored in memory)
        """
        ndjson_path = self.base_path / source / "META" / f"{resource_type}.ndjson"
        if not ndjson_path.exists():
            return

        count = 0
        yielded = 0
        with open(ndjson_path, 'r') as f:
            for line in f:
                count += 1

                # sampling
                if sample_every > 1 and count % sample_every != 0:
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    yield record
                    yielded += 1

                    if limit and yielded >= limit:
                        break
                except json.JSONDecodeError:
                    continue

    def normalize_path(self, path: str) -> str:
        """normalize path by removing array indices.

        ex. "Specimen.type.coding.0.code" -> "Specimen.type.coding.code"
        """
        parts = path.split('.')
        normalized = []
        for part in parts:
            if not part.isdigit():
                normalized.append(part)
        return '.'.join(normalized)

    def extract_paths_from_record(self, record: dict, prefix: str = "") -> Iterator[tuple[str, any]]:
        """recursively extract field paths from a single record.

        yields (path, value) tuples - processes in place, no storage.
        """
        if isinstance(record, dict):
            for key, value in record.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    yield from self.extract_paths_from_record(value, new_prefix)
                else:
                    yield (new_prefix, value)
        elif isinstance(record, list):
            for i, item in enumerate(record):
                new_prefix = f"{prefix}.{i}"
                if isinstance(item, (dict, list)):
                    yield from self.extract_paths_from_record(item, new_prefix)
                else:
                    yield (new_prefix, item)

    def extract_codings_from_record(self, record: dict, resource_type: str) -> Iterator[FhirCodingUsage]:
        """extract coding entries from a single record."""
        def find_codings(obj, path=""):
            if isinstance(obj, dict):
                if 'system' in obj and 'code' in obj:
                    yield FhirCodingUsage(
                        system=obj.get('system', ''),
                        code=obj.get('code', ''),
                        display=obj.get('display', ''),
                        resource_type=resource_type,
                        field_path=self.normalize_path(path),
                        count=1
                    )
                else:
                    for key, value in obj.items():
                        new_path = f"{path}.{key}" if path else key
                        yield from find_codings(value, new_path)
            elif isinstance(obj, list):
                for item in obj:
                    yield from find_codings(item, path)

        yield from find_codings(record, resource_type)

    def extract_refs_from_record(self, record: dict, resource_type: str) -> Iterator[FhirReferenceUsage]:
        """extract reference patterns from a single record."""
        def find_refs(obj, path=""):
            if isinstance(obj, dict):
                if 'reference' in obj:
                    ref_value = obj['reference']
                    if isinstance(ref_value, str) and '/' in ref_value:
                        target_resource = ref_value.split('/')[0]
                        field_name = path.split('.')[-1] if path else "unknown"
                        yield FhirReferenceUsage(
                            source_resource=resource_type,
                            source_field=field_name,
                            target_resource=target_resource,
                            count=1
                        )
                else:
                    for key, value in obj.items():
                        new_path = f"{path}.{key}" if path else key
                        yield from find_refs(value, new_path)
            elif isinstance(obj, list):
                for item in obj:
                    yield from find_refs(item, path)

        yield from find_refs(record, resource_type)

    def extract_source_term_from_url(self, url: str) -> tuple[str, str]:
        """extract source term and schema from a system URL.

        ex. "https://gdc.cancer.gov/sample_id" -> ("sample_id", "gdc")
        ex. "https://cadsr.cancer.gov/sample_type" -> ("sample_type", "cadsr")
        ex. "http://snomed.info/sct" -> ("", "snomed")

        returns:
            (source_term, source_schema) tuple
        """
        if not url:
            return ("", "")

        # extract schema from domain
        schema = ""
        if "gdc.cancer.gov" in url:
            schema = "gdc"
        elif "cadsr.cancer.gov" in url:
            schema = "cadsr"
        elif "ncit.nci.nih.gov" in url:
            schema = "ncit"
        elif "snomed.info" in url:
            schema = "snomed"
        elif "loinc.org" in url:
            schema = "loinc"
        elif "hl7.org" in url:
            schema = "hl7"
        elif "unitsofmeasure.org" in url:
            schema = "ucum"
        else:
            # try to extract from domain
            parts = url.replace("https://", "").replace("http://", "").split("/")
            if parts:
                domain = parts[0].split(".")[0]
                schema = domain

        # extract source term from path (last segment)
        source_term = ""
        if "/" in url:
            path_parts = url.rstrip("/").split("/")
            last_part = path_parts[-1] if path_parts else ""
            # skip if it's just a domain or standard path
            if last_part and last_part not in ["sct", "CodeSystem", "ValueSet", ""]:
                source_term = last_part

        return (source_term, schema)

    def extract_mapping_pairs_from_record(self, record: dict, resource_type: str, data_source: str) -> Iterator[SourceDestinationPair]:
        """extract source->destination pairs from a fhir record.

        sources come from:
        1. identifier[].system URL paths -> field tier
        2. coding[].system URL paths -> field tier
        3. coding[].display values -> content tier

        destinations are the fhir paths where these appear.

        args:
            record: fhir record dict
            resource_type: ex. "Specimen", "Patient"
            data_source: directory name ex. "GDC", "HTAN" - used as schema
        """
        def find_pairs(obj, path=""):
            if isinstance(obj, dict):
                # check for identifier with system
                if 'system' in obj and 'value' in obj and 'coding' not in path.lower():
                    system_url = obj.get('system', '')
                    source_term, _ = self.extract_source_term_from_url(system_url)
                    if source_term:
                        dest_path = self.normalize_path(path) if path else f"{resource_type}.identifier"
                        yield SourceDestinationPair(
                            source_term=source_term,
                            source_schema=data_source.lower(),  # use directory name
                            destination_path=dest_path,
                            destination_resource=resource_type,
                            pair_type="field",
                            count=1
                        )

                # check for coding with system/code/display
                if 'system' in obj and 'code' in obj:
                    system_url = obj.get('system', '')
                    code = obj.get('code', '')
                    display = obj.get('display', '')
                    source_term, _ = self.extract_source_term_from_url(system_url)
                    dest_path = self.normalize_path(path) if path else f"{resource_type}.code.coding"

                    # field mapping: system URL path -> fhir path
                    if source_term:
                        yield SourceDestinationPair(
                            source_term=source_term,
                            source_schema=data_source.lower(),  # use directory name
                            destination_path=dest_path,
                            destination_resource=resource_type,
                            pair_type="field",
                            dest_system=system_url,
                            count=1
                        )

                    # content mapping: display value -> fhir code
                    if display:
                        yield SourceDestinationPair(
                            source_term=display,
                            source_schema=data_source.lower(),  # use directory name
                            destination_path=dest_path,
                            destination_resource=resource_type,
                            pair_type="content",
                            dest_system=system_url,
                            dest_code=code,
                            count=1
                        )

                # recurse into other fields
                for key, value in obj.items():
                    if key not in ['system', 'code', 'display', 'value']:
                        new_path = f"{path}.{key}" if path else key
                        yield from find_pairs(value, new_path)

            elif isinstance(obj, list):
                for item in obj:
                    yield from find_pairs(item, path)

        yield from find_pairs(record, resource_type)

    def process_source(self, source: str, sample_size: int = 1000, sample_every: int = 1) -> None:
        """process a source and update aggregated stats.

        streams records - never loads all into memory.

        args:
            source: source name (ex. "GDC")
            sample_size: max records per resource type
            sample_every: sample every nth record
        """
        for resource_type in self.list_resources(source):
            count = 0

            for record in self.stream_records(source, resource_type,
                                               limit=sample_size,
                                               sample_every=sample_every):
                count += 1

                # extract and aggregate field paths
                for path, value in self.extract_paths_from_record(record):
                    norm_path = self.normalize_path(path)
                    if norm_path not in self.field_usage:
                        self.field_usage[norm_path] = FhirFieldUsage(
                            path=norm_path,
                            resource_type=resource_type
                        )
                    self.field_usage[norm_path].add_value(str(value)[:100])

                # extract and aggregate codings
                for coding in self.extract_codings_from_record(record, resource_type):
                    key = f"{coding.system}|{coding.code}"
                    if key not in self.coding_usage:
                        self.coding_usage[key] = coding
                    else:
                        self.coding_usage[key].count += 1

                # extract and aggregate references
                for ref in self.extract_refs_from_record(record, resource_type):
                    key = f"{ref.source_resource}.{ref.source_field}->{ref.target_resource}"
                    if key not in self.reference_usage:
                        self.reference_usage[key] = ref
                    else:
                        self.reference_usage[key].count += 1

                # extract source->destination mapping pairs (pass source dir as schema)
                for pair in self.extract_mapping_pairs_from_record(record, resource_type, source):
                    key = f"{pair.source_schema}:{pair.source_term}->{pair.destination_path}"
                    if key not in self.mapping_pairs:
                        self.mapping_pairs[key] = pair
                    else:
                        self.mapping_pairs[key].count += 1

            self.resource_counts[source][resource_type] = count

    def process_all(self, sample_size: int = 1000, sample_every: int = 1) -> None:
        """process all sources.

        args:
            sample_size: max records per resource type per source
            sample_every: sample every nth record (ex. 10 = every 10th)
        """
        for source in self.list_sources():
            print(f"processing {source}...")
            self.process_source(source, sample_size, sample_every)

    def get_field_paths(self, min_count: int = 1) -> dict[str, int]:
        """get field paths with counts, filtered by min count."""
        paths = {
            usage.path: usage.count
            for usage in self.field_usage.values()
            if usage.count >= min_count
        }
        return dict(sorted(paths.items(), key=lambda x: -x[1]))

    def get_coding_systems(self) -> dict[str, int]:
        """get coding systems with counts."""
        systems = defaultdict(int)
        for usage in self.coding_usage.values():
            systems[usage.system] += usage.count
        return dict(sorted(systems.items(), key=lambda x: -x[1]))

    def get_references(self) -> dict[str, int]:
        """get reference patterns with counts."""
        refs = {}
        for usage in self.reference_usage.values():
            key = f"{usage.source_resource}.{usage.source_field}->{usage.target_resource}"
            refs[key] = usage.count
        return dict(sorted(refs.items(), key=lambda x: -x[1]))

    def get_mapping_pairs(self, pair_type: str = None, tier: str = None, min_count: int = 1) -> list[SourceDestinationPair]:
        """get source->destination mapping pairs.

        args:
            pair_type: filter by "field" or "content" (None = all) - legacy
            tier: filter by "entity", "field", or "content" (None = all)
            min_count: minimum count threshold

        returns:
            list of SourceDestinationPair sorted by count
        """
        pairs = []
        for p in self.mapping_pairs.values():
            if p.count < min_count:
                continue
            if pair_type is not None and p.pair_type != pair_type:
                continue
            if tier is not None and p.tier != tier:
                continue
            pairs.append(p)
        return sorted(pairs, key=lambda x: -x.count)

    def get_tier_summary(self) -> dict:
        """get summary of mapping pairs by tier.

        returns:
            dict with counts per tier
        """
        summary = {"entity": 0, "field": 0, "content": 0}
        for p in self.mapping_pairs.values():
            summary[p.tier] = summary.get(p.tier, 0) + 1
        return summary

    def export_summary(self, output_path: str) -> None:
        """export summary to json."""
        summary = {
            "sources": list(self.resource_counts.keys()),
            "field_paths": self.get_field_paths(min_count=5),
            "coding_systems": self.get_coding_systems(),
            "references": self.get_references(),
            "resource_counts": {
                source: dict(counts)
                for source, counts in self.resource_counts.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def to_flat_db_patterns(self) -> dict:
        """convert to flat mapping database compatible format.

        returns:
            dict with destinations, coding_systems, references, mapping_pairs
        """
        return {
            "destinations": [
                {
                    "path": usage.path,
                    "resource": usage.resource_type,
                    "count": usage.count,
                    "samples": usage.sample_values[:5]
                }
                for usage in sorted(self.field_usage.values(), key=lambda x: -x.count)
            ],
            "coding_systems": [
                {
                    "system": usage.system,
                    "code": usage.code,
                    "display": usage.display,
                    "resource": usage.resource_type,
                    "field_path": usage.field_path,
                    "count": usage.count
                }
                for usage in sorted(self.coding_usage.values(), key=lambda x: -x.count)
            ],
            "references": [
                {
                    "source_resource": usage.source_resource,
                    "source_field": usage.source_field,
                    "target_resource": usage.target_resource,
                    "count": usage.count
                }
                for usage in sorted(self.reference_usage.values(), key=lambda x: -x.count)
            ],
            "mapping_pairs": [
                {
                    "source_term": pair.source_term,
                    "source_schema": pair.source_schema,
                    "destination_path": pair.destination_path,
                    "destination_resource": pair.destination_resource,
                    "pair_type": pair.pair_type,
                    "dest_system": pair.dest_system,
                    "dest_code": pair.dest_code,
                    "count": pair.count
                }
                for pair in sorted(self.mapping_pairs.values(), key=lambda x: -x.count)
            ]
        }

    def merge_into_flat_db(self, db) -> dict:
        """merge extracted mappings into a FlatMappingDatabase.

        args:
            db: FlatMappingDatabase instance

        returns:
            dict with counts of added sources, destinations, content_values
        """
        from schema_crush.mappings.flat import Source, Destination, ContentValue, ContentFhirTarget, Tier

        added = {"sources": 0, "destinations": 0, "content_values": 0}

        for pair in self.get_mapping_pairs(min_count=2):
            # determine tier
            if pair.pair_type == "content":
                tier = Tier.CONTENT
            else:
                tier = Tier.FIELD

            # create source id
            source_id = f"{pair.source_schema}:{pair.source_term}"

            # check if source already exists
            if source_id not in db.sources:
                source = Source(
                    id=source_id,
                    source=pair.source_term,
                    source_schema=pair.source_schema,
                    tier=tier,
                    source_context=""
                )
                db.add_source(source)
                added["sources"] += 1

            # add destination if not duplicate
            existing_dests = db._dest_by_source.get(source_id, [])
            dest_exists = any(d.destination == pair.destination_path for d in existing_dests)

            if not dest_exists:
                dest = Destination(
                    source_id=source_id,
                    destination=pair.destination_path,
                    dest_system=pair.dest_system,
                    dest_code=pair.dest_code,
                    dest_display=pair.source_term if pair.pair_type == "content" else ""
                )
                db.add_destination(dest)
                added["destinations"] += 1

            # for content tier, also add to content_values
            if pair.pair_type == "content" and pair.dest_code:
                # check if already exists
                existing = [cv for cv in db.content_values
                            if cv.source_value.lower() == pair.source_term.lower()
                            and cv.code == pair.dest_code]
                if not existing:
                    cv = ContentValue(
                        source_value=pair.source_term,
                        source_category="fhir_aggregator",
                        code=pair.dest_code,
                        system=pair.dest_system,
                        display=pair.source_term
                    )
                    cv_id = db.add_content_value(cv)

                    # add fhir target
                    target = ContentFhirTarget(
                        content_value_id=cv_id,
                        fhir_path=pair.destination_path,
                        fhir_resource=pair.destination_resource,
                        context="fhir_aggregator"
                    )
                    db.add_content_fhir_target(target)
                    added["content_values"] += 1

        return added


def main():
    """demo: process fhir aggregator data and print summary."""
    import sys

    if len(sys.argv) < 2:
        print("usage: python fhir_aggregator_loader.py /path/to/fhir-aggregator/all")
        sys.exit(1)

    base_path = sys.argv[1]
    loader = FhirAggregatorLoader(base_path)

    print(f"sources: {loader.list_sources()}")
    print()

    # process with sampling (100 records per resource, every 10th record)
    loader.process_all(sample_size=100, sample_every=10)

    print("\n=== top 20 field paths ===")
    for path, count in list(loader.get_field_paths().items())[:20]:
        print(f"  {path}: {count}")

    print("\n=== coding systems ===")
    for system, count in list(loader.get_coding_systems().items())[:10]:
        print(f"  {system}: {count}")

    print("\n=== reference patterns ===")
    for ref, count in list(loader.get_references().items())[:15]:
        print(f"  {ref}: {count}")

    # export
    output_path = "fhir_aggregator_summary.json"
    loader.export_summary(output_path)
    print(f"\nexported to {output_path}")


if __name__ == "__main__":
    main()