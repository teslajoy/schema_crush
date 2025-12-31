"""mapping normalization across schemas.

two types of patterns to handle:

1. MULTI-DESTINATION (valid, not a conflict):
   tumor_stage -> [Condition.stage.summary, Observation.valueCodeableConcept]
   these are intentional - one source can map to multiple FHIR locations.

2. ACTUAL CONFLICTS (multiple sources -> same destination):
   case_id -> Patient.identifier
   participant_id -> Patient.identifier
   subject_id -> Patient.identifier
   ambiguous for reverse lookup - which source term is canonical?

3. CONTEXT-AWARE (not a conflict):
   submitter_id (case) -> Patient.identifier
   submitter_id (file) -> DocumentReference.identifier
   context disambiguates - no conflict.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from .flat import FlatMappingDatabase, Source, Destination, Tier


@dataclass
class DestinationConflict:
    """conflict where multiple source terms map to the same destination."""
    destination: str
    destination_resource: str
    source_terms: dict[str, dict]  # source_term -> {schema, context, count}
    total_occurrences: int = 0
    canonical_source: Optional[str] = None  # resolved canonical source term
    resolution_method: Optional[str] = None

    @property
    def sources(self) -> list[str]:
        return sorted(self.source_terms.keys())

    @property
    def is_resolved(self) -> bool:
        return self.canonical_source is not None

    def to_dict(self) -> dict:
        return {
            "destination": self.destination,
            "resource": self.destination_resource,
            "source_terms": self.source_terms,
            "total_occurrences": self.total_occurrences,
            "canonical_source": self.canonical_source,
            "resolution_method": self.resolution_method,
        }


@dataclass
class MultiDestination:
    """valid multi-destination mapping (not a conflict)."""
    source_term: str
    source_schema: str
    source_context: str
    destinations: list[str]  # multiple valid FHIR paths

    def to_dict(self) -> dict:
        return {
            "source_term": self.source_term,
            "schema": self.source_schema,
            "context": self.source_context,
            "destinations": self.destinations,
        }


# keep old class for backwards compatibility
@dataclass
class MappingConflict:
    """DEPRECATED: use DestinationConflict instead."""
    source_term: str
    tier: Tier
    schema_destinations: dict[str, set[str]]
    total_occurrences: int = 0
    canonical_destination: Optional[str] = None
    resolution_method: Optional[str] = None

    @property
    def schemas(self) -> list[str]:
        return sorted(self.schema_destinations.keys())

    @property
    def all_destinations(self) -> set[str]:
        all_dests = set()
        for dests in self.schema_destinations.values():
            all_dests.update(dests)
        return all_dests

    @property
    def is_resolved(self) -> bool:
        return self.canonical_destination is not None

    def to_dict(self) -> dict:
        return {
            "source_term": self.source_term,
            "tier": self.tier.value,
            "schemas": self.schemas,
            "destinations": {s: list(d) for s, d in self.schema_destinations.items()},
            "total_occurrences": self.total_occurrences,
            "canonical_destination": self.canonical_destination,
            "resolution_method": self.resolution_method,
        }


@dataclass
class NormalizationReport:
    """report of conflicts and resolutions."""
    conflicts: list[MappingConflict] = field(default_factory=list)
    resolved: list[MappingConflict] = field(default_factory=list)
    total_sources: int = 0
    conflicting_sources: int = 0

    @property
    def resolution_rate(self) -> float:
        if not self.conflicts:
            return 1.0
        return len(self.resolved) / len(self.conflicts)

    def to_dict(self) -> dict:
        return {
            "total_sources": self.total_sources,
            "conflicting_sources": self.conflicting_sources,
            "conflict_rate": f"{100 * self.conflicting_sources / max(1, self.total_sources):.1f}%",
            "resolution_rate": f"{100 * self.resolution_rate:.1f}%",
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolved": [c.to_dict() for c in self.resolved],
        }

    def print_summary(self):
        print("=" * 70)
        print("MAPPING NORMALIZATION REPORT")
        print("=" * 70)
        print(f"total sources:      {self.total_sources}")
        print(f"conflicting:        {self.conflicting_sources} ({100 * self.conflicting_sources / max(1, self.total_sources):.1f}%)")
        print(f"resolved:           {len(self.resolved)}")
        print(f"unresolved:         {len(self.conflicts) - len(self.resolved)}")
        print()

        if self.conflicts:
            print("CONFLICTS DETECTED:")
            print("-" * 70)
            for c in self.conflicts[:20]:  # show first 20
                status = "[RESOLVED]" if c.is_resolved else "[PENDING]"
                print(f"\n{status} {c.source_term} ({c.tier.value})")
                for schema, dests in sorted(c.schema_destinations.items()):
                    print(f"  {schema}: {', '.join(sorted(dests))}")
                if c.is_resolved:
                    print(f"  -> canonical: {c.canonical_destination} ({c.resolution_method})")

            if len(self.conflicts) > 20:
                print(f"\n... and {len(self.conflicts) - 20} more conflicts")


class MappingNormalizer:
    """detects and resolves cross-schema mapping conflicts."""

    # schema priority for resolution (higher = more authoritative)
    DEFAULT_PRIORITY = {
        "gdc": 10,
        "htan": 9,
        "cda": 8,
        "icgc": 7,
        "gtex": 6,
        "cellosaurus": 5,
        "1000genome": 4,
    }

    # destination equivalence groups (semantically equivalent)
    EQUIVALENCE_GROUPS = [
        {"patient.identifier", "researchsubject.identifier"},
        {"patient.extension.valuestring", "extension.extension:uscoreraceextension"},
    ]

    def __init__(self, db: FlatMappingDatabase, schema_priority: dict = None):
        """initialize normalizer.

        args:
            db: flat mapping database
            schema_priority: optional custom schema priority (higher = more authoritative)
        """
        self.db = db
        self.schema_priority = schema_priority or self.DEFAULT_PRIORITY

    def detect_destination_conflicts(self, tier: Optional[Tier] = None) -> list[DestinationConflict]:
        """detect ACTUAL conflicts: multiple source terms -> same destination.

        these are ambiguous for reverse lookup (given Patient.identifier,
        which source term was it? case_id? participant_id? subject_id?)

        args:
            tier: optional tier filter (entity, field, content)

        returns:
            list of DestinationConflict objects
        """
        # group by destination (lowercase for case-insensitive)
        dest_to_sources: dict[str, dict[str, dict]] = defaultdict(dict)

        for src in self.db.sources.values():
            if tier and src.tier != tier:
                continue

            dests = self.db._dest_by_source.get(src.id, [])
            for d in dests:
                dest_lower = d.destination.lower()
                source_key = f"{src.source.lower()}|{src.source_schema}|{src.source_context}"

                if source_key not in dest_to_sources[dest_lower]:
                    dest_to_sources[dest_lower][source_key] = {
                        "source_term": src.source,
                        "schema": src.source_schema,
                        "context": src.source_context,
                        "count": 0,
                    }
                dest_to_sources[dest_lower][source_key]["count"] += 1

        # find conflicts (multiple DIFFERENT source terms for same destination)
        conflicts = []
        for dest, sources in dest_to_sources.items():
            # group by unique source term (ignoring schema/context)
            unique_terms = set(info["source_term"].lower() for info in sources.values())

            if len(unique_terms) > 1:
                # extract resource from destination
                resource = dest.split(".")[0] if "." in dest else dest

                # build source_terms dict
                source_terms = {}
                for info in sources.values():
                    term = info["source_term"]
                    if term not in source_terms:
                        source_terms[term] = {
                            "schemas": [],
                            "contexts": [],
                            "count": 0,
                        }
                    if info["schema"] not in source_terms[term]["schemas"]:
                        source_terms[term]["schemas"].append(info["schema"])
                    if info["context"] and info["context"] not in source_terms[term]["contexts"]:
                        source_terms[term]["contexts"].append(info["context"])
                    source_terms[term]["count"] += info["count"]

                conflict = DestinationConflict(
                    destination=dest,
                    destination_resource=resource,
                    source_terms=source_terms,
                    total_occurrences=sum(info["count"] for info in sources.values()),
                )
                conflicts.append(conflict)

        return sorted(conflicts, key=lambda c: -c.total_occurrences)

    def get_multi_destinations(self, tier: Optional[Tier] = None) -> list[MultiDestination]:
        """get valid multi-destination mappings (one source -> multiple destinations).

        these are NOT conflicts - they're intentional mappings where one source
        term can map to multiple FHIR locations (e.g., tumor_stage ->
        [Condition.stage.summary, Observation.valueCodeableConcept]).

        args:
            tier: optional tier filter

        returns:
            list of MultiDestination objects
        """
        multi = []

        for src in self.db.sources.values():
            if tier and src.tier != tier:
                continue

            dests = self.db._dest_by_source.get(src.id, [])
            if len(dests) > 1:
                multi.append(MultiDestination(
                    source_term=src.source,
                    source_schema=src.source_schema,
                    source_context=src.source_context,
                    destinations=[d.destination for d in dests],
                ))

        return sorted(multi, key=lambda m: -len(m.destinations))

    def resolve_destination_conflict(
        self,
        conflict: DestinationConflict,
        method: str = "frequency"
    ) -> str:
        """resolve destination conflict by picking canonical source term.

        args:
            conflict: DestinationConflict to resolve
            method: "frequency" (most common) or "priority" (highest priority schema)

        returns:
            canonical source term
        """
        if method == "frequency":
            # pick source term with highest count
            return max(conflict.source_terms.keys(),
                      key=lambda t: conflict.source_terms[t]["count"])

        elif method == "priority":
            # pick source term from highest priority schema
            best_term = None
            best_priority = -1
            for term, info in conflict.source_terms.items():
                for schema in info["schemas"]:
                    priority = self.schema_priority.get(schema, 0)
                    if priority > best_priority:
                        best_priority = priority
                        best_term = term
            return best_term or list(conflict.source_terms.keys())[0]

        return list(conflict.source_terms.keys())[0]

    def detect_conflicts(self, tier: Optional[Tier] = None) -> list[MappingConflict]:
        """detect sources with divergent destinations across schemas.

        args:
            tier: optional tier filter (entity, field, content)

        returns:
            list of MappingConflict objects
        """
        # group by source term (lowercase for case-insensitive matching)
        term_to_schemas: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        term_to_tier: dict[str, Tier] = {}
        term_counts: dict[str, int] = defaultdict(int)

        for src in self.db.sources.values():
            if tier and src.tier != tier:
                continue

            term_lower = src.source.lower()
            term_to_tier[term_lower] = src.tier

            dests = self.db._dest_by_source.get(src.id, [])
            for d in dests:
                term_to_schemas[term_lower][src.source_schema].add(d.destination.lower())
                term_counts[term_lower] += 1

        # find conflicts (same term, different destinations across schemas)
        conflicts = []
        for term, schema_dests in term_to_schemas.items():
            if len(schema_dests) <= 1:
                continue  # only one schema, no conflict

            # collect all unique destinations
            all_dests = set()
            for dests in schema_dests.values():
                all_dests.update(dests)

            # check if destinations actually differ
            # (same term in multiple schemas with SAME destination is not a conflict)
            if len(all_dests) > 1:
                conflict = MappingConflict(
                    source_term=term,
                    tier=term_to_tier[term],
                    schema_destinations=dict(schema_dests),
                    total_occurrences=term_counts[term],
                )
                conflicts.append(conflict)

        return sorted(conflicts, key=lambda c: -c.total_occurrences)

    def resolve_by_priority(self, conflict: MappingConflict) -> str:
        """resolve conflict by picking destination from highest priority schema.

        args:
            conflict: MappingConflict to resolve

        returns:
            canonical destination path
        """
        best_schema = None
        best_priority = -1

        for schema in conflict.schemas:
            priority = self.schema_priority.get(schema, 0)
            if priority > best_priority:
                best_priority = priority
                best_schema = schema

        if best_schema:
            # pick first destination from best schema
            dests = conflict.schema_destinations[best_schema]
            return sorted(dests)[0]

        return sorted(conflict.all_destinations)[0]

    def resolve_by_frequency(self, conflict: MappingConflict) -> str:
        """resolve conflict by picking most common destination.

        args:
            conflict: MappingConflict to resolve

        returns:
            canonical destination path
        """
        # count how many schemas use each destination
        dest_counts = defaultdict(int)
        for dests in conflict.schema_destinations.values():
            for d in dests:
                dest_counts[d] += 1

        # return most common
        return max(dest_counts.keys(), key=lambda d: dest_counts[d])

    def resolve_by_equivalence(self, conflict: MappingConflict) -> Optional[str]:
        """resolve conflict if destinations are semantically equivalent.

        args:
            conflict: MappingConflict to resolve

        returns:
            canonical destination if equivalent, None otherwise
        """
        all_dests = conflict.all_destinations

        for equiv_group in self.EQUIVALENCE_GROUPS:
            if all_dests.issubset(equiv_group):
                # all destinations are equivalent, pick first alphabetically
                return sorted(all_dests)[0]

        return None

    def auto_resolve(
        self,
        conflicts: list[MappingConflict],
        method: str = "priority"
    ) -> list[MappingConflict]:
        """auto-resolve conflicts using specified method.

        args:
            conflicts: list of conflicts to resolve
            method: resolution method ("priority", "frequency", "equivalence")

        returns:
            list of resolved conflicts
        """
        resolved = []

        for conflict in conflicts:
            # try equivalence first (if destinations are semantically same)
            equiv_dest = self.resolve_by_equivalence(conflict)
            if equiv_dest:
                conflict.canonical_destination = equiv_dest
                conflict.resolution_method = "equivalence"
                resolved.append(conflict)
                continue

            # use specified method
            if method == "priority":
                conflict.canonical_destination = self.resolve_by_priority(conflict)
                conflict.resolution_method = "priority"
            elif method == "frequency":
                conflict.canonical_destination = self.resolve_by_frequency(conflict)
                conflict.resolution_method = "frequency"

            resolved.append(conflict)

        return resolved

    def generate_report(
        self,
        tier: Optional[Tier] = None,
        auto_resolve: bool = True,
        method: str = "priority"
    ) -> NormalizationReport:
        """generate normalization report with conflicts and resolutions.

        args:
            tier: optional tier filter
            auto_resolve: whether to auto-resolve conflicts
            method: resolution method if auto_resolve

        returns:
            NormalizationReport
        """
        # count total sources
        total = sum(1 for s in self.db.sources.values() if tier is None or s.tier == tier)

        # detect conflicts
        conflicts = self.detect_conflicts(tier)

        # auto-resolve if requested
        resolved = []
        if auto_resolve and conflicts:
            resolved = self.auto_resolve(conflicts, method)

        report = NormalizationReport(
            conflicts=conflicts,
            resolved=resolved,
            total_sources=total,
            conflicting_sources=len(set(c.source_term for c in conflicts)),
        )

        return report

    def apply_canonical_mappings(
        self,
        conflicts: list[MappingConflict],
        update_db: bool = False
    ) -> dict:
        """apply canonical mappings to database.

        for each resolved conflict, creates a canonical source entry that
        points to the resolved destination.

        args:
            conflicts: list of resolved conflicts
            update_db: if True, actually modify the database

        returns:
            dict with stats on changes made
        """
        stats = {"canonical_sources_added": 0, "destinations_added": 0}

        for conflict in conflicts:
            if not conflict.is_resolved:
                continue

            # create canonical source id
            canonical_id = f"canonical:{conflict.source_term}"

            if update_db and canonical_id not in self.db.sources:
                from .flat import Source, Destination

                # add canonical source
                src = Source(
                    id=canonical_id,
                    source=conflict.source_term,
                    source_schema="canonical",
                    tier=conflict.tier,
                    source_context=""
                )
                self.db.add_source(src)
                stats["canonical_sources_added"] += 1

                # add canonical destination
                dest = Destination(
                    source_id=canonical_id,
                    destination=conflict.canonical_destination
                )
                self.db.add_destination(dest)
                stats["destinations_added"] += 1

        return stats

    def export_conflicts_for_hitl(self, conflicts: list[MappingConflict], output_path: str):
        """export conflicts to JSON for HITL review in mapping workbench.

        args:
            conflicts: list of conflicts
            output_path: path to write JSON
        """
        import json

        hitl_items = []
        for conflict in conflicts:
            hitl_items.append({
                "id": f"conflict:{conflict.source_term}",
                "source_term": conflict.source_term,
                "tier": conflict.tier.value,
                "schemas": conflict.schemas,
                "options": [
                    {
                        "destination": dest,
                        "schemas": [s for s, dests in conflict.schema_destinations.items() if dest in dests],
                        "is_suggested": dest == conflict.canonical_destination,
                    }
                    for dest in sorted(conflict.all_destinations)
                ],
                "auto_resolution": conflict.canonical_destination,
                "auto_method": conflict.resolution_method,
                "status": "resolved" if conflict.is_resolved else "pending",
            })

        with open(output_path, 'w') as f:
            json.dump({"conflicts": hitl_items, "total": len(hitl_items)}, f, indent=2)


def detect_and_report(db: FlatMappingDatabase, tier: Optional[Tier] = None) -> NormalizationReport:
    """convenience function to detect conflicts and print report.

    args:
        db: flat mapping database
        tier: optional tier filter

    returns:
        NormalizationReport
    """
    normalizer = MappingNormalizer(db)
    report = normalizer.generate_report(tier=tier, auto_resolve=True, method="priority")
    report.print_summary()
    return report


# known fhir resources for validation
FHIR_RESOURCES = {
    'Patient', 'Specimen', 'Condition', 'Observation', 'DocumentReference',
    'ResearchStudy', 'ResearchSubject', 'MedicationAdministration', 'ServiceRequest',
    'Device', 'Identifier', 'Extension', 'Group', 'ImagingStudy', 'Organization',
    'Attachment', 'Medication', 'DiagnosticReport', 'ResearchStudyRecruitment',
    'ResearchStudyProgressStatus', 'Encounter', 'Procedure', 'Task', 'Practitioner',
    'CodeableConcept', 'Coding', 'Reference', 'Period', 'Quantity', 'Range',
}

# context to focus_ref mapping
CONTEXT_TO_FOCUS = {
    # patient-related contexts
    'patient': 'Patient',
    'demographic': 'Patient',
    'demographics': 'Patient',
    'case': 'Patient',
    'cases': 'Patient',
    'subject': 'Patient',
    'participant': 'Patient',
    'followup': 'Patient',
    'follow_up': 'Patient',
    'exposure': 'Patient',
    'familyhistory': 'Patient',
    'family_history': 'Patient',

    # specimen-related contexts
    'specimen': 'Specimen',
    'sample': 'Specimen',
    'samples': 'Specimen',
    'biospecimen': 'Specimen',
    'biospecimens': 'Specimen',
    'portion': 'Specimen',
    'portions': 'Specimen',
    'aliquot': 'Specimen',
    'aliquots': 'Specimen',
    'analyte': 'Specimen',
    'analytes': 'Specimen',
    'slide': 'Specimen',
    'slides': 'Specimen',

    # file-related contexts
    'file': 'DocumentReference',
    'files': 'DocumentReference',
    'document': 'DocumentReference',
    'attachment': 'DocumentReference',

    # diagnosis-related contexts
    'diagnosis': 'Condition',
    'diagnoses': 'Condition',
    'condition': 'Condition',

    # treatment-related contexts
    'treatment': 'MedicationAdministration',
    'treatments': 'MedicationAdministration',
    'therapy': 'MedicationAdministration',
}


def is_valid_fhir_destination(dest: str) -> bool:
    """check if destination is a valid fhir path (not a pass-through).

    args:
        dest: destination string

    returns:
        true if valid fhir path
    """
    if not dest:
        return False

    # check if starts with a known fhir resource
    first_part = dest.split('.')[0].split(':')[0]
    if first_part in FHIR_RESOURCES:
        return True

    # check for extension pattern
    if dest.startswith('Extension') or ':extension' in dest.lower():
        return True

    return False


def infer_focus_ref(context: str, source_term: str = "") -> str:
    """infer focus_ref from context and source term.

    args:
        context: source context (e.g., "demographic", "samples")
        source_term: source field name

    returns:
        focus_ref string ("Patient", "Specimen", "DocumentReference")
    """
    ctx_lower = context.lower() if context else ""

    # direct context match
    if ctx_lower in CONTEXT_TO_FOCUS:
        return CONTEXT_TO_FOCUS[ctx_lower]

    # check if context contains known keywords
    for keyword, focus in CONTEXT_TO_FOCUS.items():
        if keyword in ctx_lower:
            return focus

    # check source term for hints
    term_lower = source_term.lower()
    if any(kw in term_lower for kw in ['specimen', 'sample', 'tissue', 'tumor', 'biopsy']):
        return 'Specimen'
    if any(kw in term_lower for kw in ['file', 'document', 'data_', 'format']):
        return 'DocumentReference'
    if any(kw in term_lower for kw in ['patient', 'age', 'gender', 'race', 'ethnicity']):
        return 'Patient'

    # default to specimen for htan (most common), patient for others
    return 'Specimen'


def fix_passthrough_mappings(db: FlatMappingDatabase, dry_run: bool = True) -> dict:
    """fix pass-through mappings by converting to observation.component.

    pass-through mappings are destinations that:
    - equal the source term
    - are garbage (short codes, numeric ids)
    - don't start with a valid fhir resource

    these get converted to observation.component with appropriate focus_ref.

    args:
        db: flat mapping database
        dry_run: if true, only report what would change (don't modify)

    returns:
        dict with stats and list of changes
    """
    changes = []
    stats = {
        "total_checked": 0,
        "pass_throughs_found": 0,
        "fixed": 0,
        "by_focus": defaultdict(int),
    }

    for src in db.sources.values():
        dests = db._dest_by_source.get(src.id, [])

        for dest in dests:
            stats["total_checked"] += 1

            # check if this is a pass-through
            if is_valid_fhir_destination(dest.destination):
                continue

            # check if it's garbage
            is_passthrough = (
                dest.destination.lower() == src.source.lower() or  # exact match
                len(dest.destination) < 3 or  # too short
                dest.destination.replace('-', '').replace('_', '').isdigit() or  # numeric
                dest.destination.startswith('NCIT_') or  # ncit codes
                not any(c.isalpha() for c in dest.destination)  # no letters
            )

            if not is_passthrough and not is_valid_fhir_destination(dest.destination):
                # unknown destination - also convert
                is_passthrough = True

            if is_passthrough:
                stats["pass_throughs_found"] += 1

                # infer focus_ref
                focus_ref = infer_focus_ref(src.source_context, src.source)
                stats["by_focus"][focus_ref] += 1

                old_dest = dest.destination
                new_dest = "Observation.component"

                changes.append({
                    "source_id": src.id,
                    "source_term": src.source,
                    "context": src.source_context,
                    "schema": src.source_schema,
                    "old_destination": old_dest,
                    "new_destination": new_dest,
                    "focus_ref": focus_ref,
                })

                if not dry_run:
                    # update destination
                    dest.destination = new_dest
                    dest.focus_ref = focus_ref
                    stats["fixed"] += 1

    return {"stats": dict(stats), "changes": changes}


def create_calibration_ground_truth(
    curated_json_path: str = None,
    fhir_ndjson_path: str = None,
) -> dict:
    """create ground truth datasets for calibration metrics.

    args:
        curated_json_path: path to gdc mapping json directory
        fhir_ndjson_path: path to fhir aggregator data directory

    returns:
        dict with "curated" and "fhir_actual" ground truth
    """
    from pathlib import Path

    ground_truth = {
        "curated": [],  # from case.json, file.json, project.json
        "fhir_actual": [],  # from actual fhir ndjson
    }

    # load curated gdc mappings
    if curated_json_path is None:
        curated_json_path = Path(__file__).parent.parent / "data/resources/gdc_mapping"
    else:
        curated_json_path = Path(curated_json_path)

    if curated_json_path.exists():
        import json
        for json_file in curated_json_path.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)

            entity_type = json_file.stem

            # extract mappings
            mappings = data.get("mappings", [])
            for m in mappings:
                src_info = m.get("source", {})
                dst_info = m.get("destination", {})

                src_name = src_info.get("name", "")
                dst_name = dst_info.get("name", "")

                if src_name and dst_name:
                    # extract just the field name from nested paths
                    src_field = src_name.split(".")[-1] if "." in src_name else src_name

                    ground_truth["curated"].append({
                        "source_term": src_field,
                        "source_full": src_name,
                        "destination": dst_name,
                        "entity": entity_type,
                        "schema": "gdc",
                    })

    # load fhir actual patterns (if path provided)
    if fhir_ndjson_path:
        fhir_path = Path(fhir_ndjson_path)
        if fhir_path.exists():
            # use fhiraggregatorloader to extract patterns
            from schema_crush.loaders.fhir_aggregator_loader import FhirAggregatorLoader

            loader = FhirAggregatorLoader(str(fhir_path))
            loader.process_all(sample_size=500, sample_every=5)

            for pair in loader.get_mapping_pairs(min_count=3):
                ground_truth["fhir_actual"].append({
                    "source_term": pair.source_term,
                    "destination": pair.destination_path,
                    "schema": pair.source_schema,
                    "pair_type": pair.pair_type,
                    "count": pair.count,
                })

    return ground_truth


def compute_calibration_metrics(
    db: FlatMappingDatabase,
    ground_truth: dict,
    source: str = "curated",
) -> dict:
    """compute calibration metrics against ground truth.

    handles one-to-many mappings properly:
    - if expected dest is in flat db's multi-destinations -> exact match
    - if expected resource matches any flat db resource -> resource match

    args:
        db: flat mapping database to evaluate
        ground_truth: dict from create_calibration_ground_truth
        source: "curated" or "fhir_actual"

    returns:
        dict with accuracy, coverage, and detailed results
    """
    gt_entries = ground_truth.get(source, [])
    if not gt_entries:
        return {"error": f"no ground truth for source: {source}"}

    results = {
        "source": source,
        "total_gt_entries": len(gt_entries),
        "exact_matches": 0,
        "resource_matches": 0,  # right resource, wrong field
        "no_match": 0,
        "missing_destinations": [],  # fhir actual has dest not in flat db
        "mismatches": [],
        "terminology_differences": [],
    }

    for gt in gt_entries:
        source_term = gt["source_term"]
        expected_dest = gt["destination"]

        # lookup in flat db - try with and without schema filter
        lookups = db.lookup(source_term, schema=gt.get("schema"))
        if not lookups:
            # try without schema filter (cross-schema lookup)
            lookups = db.lookup(source_term)

        if not lookups:
            results["no_match"] += 1
            results["mismatches"].append({
                "source_term": source_term,
                "expected": expected_dest,
                "predicted": None,
                "reason": "no_lookup_result",
            })
            continue

        # get ALL destinations for this source (one-to-many)
        predicted_dests = [d.destination for _, d in lookups]

        # exact match: expected dest is one of the multi-destinations
        if any(p.lower() == expected_dest.lower() for p in predicted_dests):
            results["exact_matches"] += 1
            continue

        # resource match: expected resource matches any predicted resource
        expected_resource = expected_dest.split(".")[0].lower()
        predicted_resources = [p.split(".")[0].lower() for p in predicted_dests]

        if expected_resource in predicted_resources:
            results["resource_matches"] += 1
            # this is a missing destination - flat db has resource but not this specific path
            results["missing_destinations"].append({
                "source_term": source_term,
                "expected": expected_dest,
                "flat_db_has": [p for p in predicted_dests if p.split(".")[0].lower() == expected_resource],
            })
            continue

        # complete mismatch - different resource entirely
        results["mismatches"].append({
            "source_term": source_term,
            "expected": expected_dest,
            "predicted": predicted_dests,
            "reason": "wrong_resource",
        })

    # compute metrics
    total = results["total_gt_entries"]
    results["exact_accuracy"] = results["exact_matches"] / total if total > 0 else 0
    results["resource_accuracy"] = (results["exact_matches"] + results["resource_matches"]) / total if total > 0 else 0
    results["coverage"] = (total - results["no_match"]) / total if total > 0 else 0

    return results


def print_calibration_report(metrics: dict):
    """print calibration metrics report."""
    print("=" * 70)
    print(f"CALIBRATION METRICS: {metrics.get('source', 'unknown').upper()}")
    print("=" * 70)

    print(f"\nground truth entries: {metrics.get('total_gt_entries', 0)}")
    print(f"exact matches:        {metrics.get('exact_matches', 0)}")
    print(f"resource matches:     {metrics.get('resource_matches', 0)} (missing specific path)")
    print(f"no match:             {metrics.get('no_match', 0)}")

    print(f"\nexact accuracy:    {100 * metrics.get('exact_accuracy', 0):.1f}%")
    print(f"resource accuracy: {100 * metrics.get('resource_accuracy', 0):.1f}%")
    print(f"coverage:          {100 * metrics.get('coverage', 0):.1f}%")

    # show missing destinations (one-to-many gaps)
    missing = metrics.get("missing_destinations", [])
    if missing:
        print(f"\n--- missing destinations ({len(missing)}) ---")
        print("(flat db has resource but missing this specific fhir path)")
        for m in missing[:10]:
            print(f"  {m['source_term'][:25]:25}")
            print(f"    fhir actual: {m['expected']}")
            print(f"    flat db has: {m['flat_db_has'][:2]}")

    # show mismatches
    mismatches = metrics.get("mismatches", [])
    if mismatches:
        print(f"\n--- mismatches ({len(mismatches)}) ---")
        for m in mismatches[:10]:
            print(f"  {m['source_term'][:25]:25} ({m['reason']})")
            print(f"    expected:  {m['expected']}")
            print(f"    predicted: {m['predicted']}")

    print("=" * 70)