"""two-table mapping structure optimized for fast lookup and training export."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import csv
import json
import sqlite3
from pathlib import Path


class Tier(str, Enum):
    ENTITY = "entity"
    FIELD = "field"
    CONTENT = "content"


@dataclass
class Source:
    """unique source term from a schema (GDC, HTAN, BTS, etc.)"""
    id: str                          # "gdc:sample_id", "htan:gender.male"
    source: str                      # "sample_id", "male", "Biospecimen"
    source_schema: str               # "gdc", "htan", "bts"
    tier: Tier                       # entity, field, content
    source_context: str = ""         # parent: "demographic" for gender, "sample" for sample_id

    def training_key(self) -> str:
        """source text for embedding training."""
        if self.source_context:
            return f"{self.source_context}.{self.source}"
        return self.source

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "source_schema": self.source_schema,
            "tier": self.tier.value,
            "source_context": self.source_context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        return cls(
            id=data["id"],
            source=data["source"],
            source_schema=data["source_schema"],
            tier=Tier(data["tier"]),
            source_context=data.get("source_context", ""),
        )


@dataclass
class Destination:
    """single FHIR destination for a source (one-to-many = multiple Destinations)."""
    source_id: str                   # FK to Source.id
    destination: str                 # "Patient.identifier", "Observation.valueQuantity"

    # FHIR coding (for content tier)
    dest_system: str = ""            # "http://snomed.info/sct"
    dest_code: str = ""              # "254626006"
    dest_display: str = ""           # "Lung adenocarcinoma"

    # FHIR units (for quantities)
    unit: str = ""                   # "mm", "kg/m2"
    unit_system: str = ""            # "http://unitsofmeasure.org"
    unit_code: str = ""              # "mm"

    # FHIR references
    subject_ref: str = ""            # "Patient", "Group"
    focus_ref: str = ""              # "Specimen", "Condition"
    specimen_ref: str = ""           # "Specimen"

    def training_value(self) -> str:
        """destination text for embedding training."""
        parts = [self.destination]

        if self.dest_system:
            parts.append(f"system={self.dest_system}|code={self.dest_code}")
        if self.dest_display:
            parts.append(f"display={self.dest_display}")
        if self.unit:
            parts.append(f"unit={self.unit}")
        if self.subject_ref:
            parts.append(f"subject->Reference({self.subject_ref})")
        if self.focus_ref:
            parts.append(f"focus->Reference({self.focus_ref})")
        if self.specimen_ref:
            parts.append(f"specimen->Reference({self.specimen_ref})")

        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "destination": self.destination,
            "dest_system": self.dest_system,
            "dest_code": self.dest_code,
            "dest_display": self.dest_display,
            "unit": self.unit,
            "unit_system": self.unit_system,
            "unit_code": self.unit_code,
            "subject_ref": self.subject_ref,
            "focus_ref": self.focus_ref,
            "specimen_ref": self.specimen_ref,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Destination":
        return cls(
            source_id=data["source_id"],
            destination=data["destination"],
            dest_system=data.get("dest_system", ""),
            dest_code=data.get("dest_code", ""),
            dest_display=data.get("dest_display", ""),
            unit=data.get("unit", ""),
            unit_system=data.get("unit_system", ""),
            unit_code=data.get("unit_code", ""),
            subject_ref=data.get("subject_ref", ""),
            focus_ref=data.get("focus_ref", ""),
            specimen_ref=data.get("specimen_ref", ""),
        )


@dataclass
class ContentValue:
    """coded value mapping: source string -> FHIR code (SNOMED, etc)."""
    id: int = 0                      # auto-increment PK
    source_value: str = ""           # "Adenocarcinoma", "G1", "Breast Cancer"
    source_category: str = ""        # "histology", "staging_grade", "condition"
    code: str = ""                   # "443961001"
    system: str = "http://snomed.info/sct"  # default SNOMED CT
    display: str = ""                # "Malignant adenomatous neoplasm (disorder)"
    is_primary: bool = True          # primary code when multiple codes exist

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_value": self.source_value,
            "source_category": self.source_category,
            "code": self.code,
            "system": self.system,
            "display": self.display,
            "is_primary": self.is_primary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentValue":
        return cls(
            id=data.get("id", 0),
            source_value=data["source_value"],
            source_category=data["source_category"],
            code=data["code"],
            system=data.get("system", "http://snomed.info/sct"),
            display=data.get("display", ""),
            is_primary=data.get("is_primary", True),
        )


@dataclass
class ContentFhirTarget:
    """FHIR path target for a content value (many-to-many)."""
    id: int = 0                      # auto-increment PK
    content_value_id: int = 0        # FK to ContentValue.id
    fhir_path: str = ""              # "Condition.code", "Observation.valueCodeableConcept"
    fhir_resource: str = ""          # "Condition", "Observation"
    context: str = ""                # "histology", "staging", "diagnosis"
    is_primary: bool = True          # primary path when multiple paths exist

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content_value_id": self.content_value_id,
            "fhir_path": self.fhir_path,
            "fhir_resource": self.fhir_resource,
            "context": self.context,
            "is_primary": self.is_primary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentFhirTarget":
        return cls(
            id=data.get("id", 0),
            content_value_id=data["content_value_id"],
            fhir_path=data["fhir_path"],
            fhir_resource=data.get("fhir_resource", ""),
            context=data.get("context", ""),
            is_primary=data.get("is_primary", True),
        )


@dataclass
class FlatMappingDatabase:
    """two-table storage with O(1) lookup and training export."""

    # storage
    sources: dict[str, Source] = field(default_factory=dict)
    destinations: list[Destination] = field(default_factory=list)

    # content value mappings (normalized)
    content_values: list[ContentValue] = field(default_factory=list)
    content_fhir_targets: list[ContentFhirTarget] = field(default_factory=list)

    # indexes for O(1) lookup
    _source_by_term: dict[str, list[str]] = field(default_factory=dict)  # source -> [ids]
    _source_by_term_context: dict[tuple, str] = field(default_factory=dict)  # (source, context) -> id
    _dest_by_source: dict[str, list[Destination]] = field(default_factory=dict)  # source_id -> [dests]

    def add_source(self, source: Source) -> None:
        """add source and update indexes."""
        self.sources[source.id] = source

        # index by source term (handles duplicates across contexts)
        if source.source not in self._source_by_term:
            self._source_by_term[source.source] = []
        self._source_by_term[source.source].append(source.id)

        # index by (source, context) for exact match
        key = (source.source, source.source_context)
        self._source_by_term_context[key] = source.id

    def add_destination(self, dest: Destination) -> None:
        """add destination and update index."""
        self.destinations.append(dest)

        if dest.source_id not in self._dest_by_source:
            self._dest_by_source[dest.source_id] = []
        self._dest_by_source[dest.source_id].append(dest)

    def lookup(self, source_term: str, context: Optional[str] = None) -> list[tuple[Source, Destination]]:
        """O(1) lookup: get all mappings for a source term.

        args:
            source_term: source field name (e.g., "sample_id")
            context: optional context for disambiguation (e.g., "sample")

        returns:
            list of (Source, Destination) tuples
        """
        if context:
            # exact match with context
            source_id = self._source_by_term_context.get((source_term, context))
            if not source_id:
                return []
            source = self.sources[source_id]
            dests = self._dest_by_source.get(source_id, [])
            return [(source, d) for d in dests]
        else:
            # all sources matching term
            source_ids = self._source_by_term.get(source_term, [])
            results = []
            for source_id in source_ids:
                source = self.sources[source_id]
                dests = self._dest_by_source.get(source_id, [])
                results.extend((source, d) for d in dests)
            return results

    def lookup_by_destination(self, destination: str) -> list[tuple[Source, Destination]]:
        """reverse lookup: find sources that map to a destination."""
        results = []
        dest_lower = destination.lower()
        for dest in self.destinations:
            if dest.destination.lower() == dest_lower:
                source = self.sources.get(dest.source_id)
                if source:
                    results.append((source, dest))
        return results

    def export_training_pairs(self) -> list[tuple[str, str]]:
        """flat pairs for embedding training."""
        pairs = []
        for dest in self.destinations:
            source = self.sources.get(dest.source_id)
            if source:
                pairs.append((source.training_key(), dest.training_value()))
        return pairs

    def export_training_csv(self, path: str) -> None:
        """export to CSV for training."""
        pairs = self.export_training_pairs()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "destination"])
            writer.writerows(pairs)

    def filter_by_tier(self, tier: Tier) -> list[tuple[Source, Destination]]:
        """get all mappings for a tier."""
        results = []
        for source_id, dests in self._dest_by_source.items():
            source = self.sources.get(source_id)
            if source and source.tier == tier:
                results.extend((source, d) for d in dests)
        return results

    def filter_by_schema(self, schema: str) -> list[tuple[Source, Destination]]:
        """get all mappings for a schema."""
        results = []
        for source_id, dests in self._dest_by_source.items():
            source = self.sources.get(source_id)
            if source and source.source_schema == schema:
                results.extend((source, d) for d in dests)
        return results

    def count(self) -> dict[str, int]:
        """return counts."""
        return {
            "sources": len(self.sources),
            "destinations": len(self.destinations),
        }

    def stats(self) -> dict:
        """return detailed statistics."""
        schemas = set(s.source_schema for s in self.sources.values())
        tiers = set(s.tier.value for s in self.sources.values())
        return {
            "counts": self.count(),
            "schemas": sorted(schemas),
            "tiers": sorted(tiers),
        }

    def save(self, path: Path) -> None:
        """save to JSON."""
        data = {
            "sources": [s.to_dict() for s in self.sources.values()],
            "destinations": [d.to_dict() for d in self.destinations],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """load from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        for s in data.get("sources", []):
            self.add_source(Source.from_dict(s))
        for d in data.get("destinations", []):
            self.add_destination(Destination.from_dict(d))

    def clear(self) -> None:
        """clear all data and indexes."""
        self.sources.clear()
        self.destinations.clear()
        self._source_by_term.clear()
        self._source_by_term_context.clear()
        self._dest_by_source.clear()

    def add_content_value(self, cv: ContentValue) -> int:
        """add content value and return its id."""
        cv.id = len(self.content_values) + 1
        self.content_values.append(cv)
        return cv.id

    def add_content_fhir_target(self, target: ContentFhirTarget) -> None:
        """add FHIR target for a content value."""
        target.id = len(self.content_fhir_targets) + 1
        self.content_fhir_targets.append(target)

    def lookup_content(self, source_value: str, category: Optional[str] = None) -> list[tuple[ContentValue, list[ContentFhirTarget]]]:
        """lookup content value mappings."""
        results = []
        for cv in self.content_values:
            if cv.source_value.lower() == source_value.lower():
                if category and cv.source_category != category:
                    continue
                targets = [t for t in self.content_fhir_targets if t.content_value_id == cv.id]
                results.append((cv, targets))
        return results

    def lookup_content_by_fhir_path(self, fhir_path: str) -> list[tuple[ContentValue, ContentFhirTarget]]:
        """find all content values that map to a FHIR path."""
        results = []
        for target in self.content_fhir_targets:
            if target.fhir_path.lower() == fhir_path.lower():
                cv = next((c for c in self.content_values if c.id == target.content_value_id), None)
                if cv:
                    results.append((cv, target))
        return results

    def save_sqlite(self, path: Path) -> None:
        """save to SQLite database for inspection and querying.

        creates tables:
        - sources: id, source, source_schema, tier, source_context
        - destinations: source_id (FK), destination, dest_system, dest_code, ...
        - content_values: coded value mappings
        - content_fhir_targets: FHIR paths for content values
        """
        path = Path(path)
        if path.exists():
            path.unlink()

        conn = sqlite3.connect(path)
        cur = conn.cursor()

        # create sources table
        cur.execute("""
            CREATE TABLE sources (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_schema TEXT NOT NULL,
                tier TEXT NOT NULL,
                source_context TEXT
            )
        """)

        # create destinations table
        cur.execute("""
            CREATE TABLE destinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                destination TEXT NOT NULL,
                dest_system TEXT,
                dest_code TEXT,
                dest_display TEXT,
                unit TEXT,
                unit_system TEXT,
                unit_code TEXT,
                subject_ref TEXT,
                focus_ref TEXT,
                specimen_ref TEXT,
                FOREIGN KEY (source_id) REFERENCES sources(id)
            )
        """)

        # create content_values table
        cur.execute("""
            CREATE TABLE content_values (
                id INTEGER PRIMARY KEY,
                source_value TEXT NOT NULL,
                source_category TEXT NOT NULL,
                code TEXT NOT NULL,
                system TEXT,
                display TEXT,
                is_primary INTEGER DEFAULT 1
            )
        """)

        # create content_fhir_targets table
        cur.execute("""
            CREATE TABLE content_fhir_targets (
                id INTEGER PRIMARY KEY,
                content_value_id INTEGER NOT NULL,
                fhir_path TEXT NOT NULL,
                fhir_resource TEXT,
                context TEXT,
                is_primary INTEGER DEFAULT 1,
                FOREIGN KEY (content_value_id) REFERENCES content_values(id)
            )
        """)

        # create indexes
        cur.execute("CREATE INDEX idx_sources_source ON sources(source)")
        cur.execute("CREATE INDEX idx_sources_tier ON sources(tier)")
        cur.execute("CREATE INDEX idx_sources_schema ON sources(source_schema)")
        cur.execute("CREATE INDEX idx_dest_source_id ON destinations(source_id)")
        cur.execute("CREATE INDEX idx_dest_destination ON destinations(destination)")
        cur.execute("CREATE INDEX idx_cv_source_value ON content_values(source_value)")
        cur.execute("CREATE INDEX idx_cv_category ON content_values(source_category)")
        cur.execute("CREATE INDEX idx_cv_code ON content_values(code)")
        cur.execute("CREATE INDEX idx_cft_content_value_id ON content_fhir_targets(content_value_id)")
        cur.execute("CREATE INDEX idx_cft_fhir_path ON content_fhir_targets(fhir_path)")

        # insert sources
        for src in self.sources.values():
            cur.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                (src.id, src.source, src.source_schema, src.tier.value, src.source_context)
            )

        # insert destinations
        for dest in self.destinations:
            cur.execute(
                """INSERT INTO destinations
                   (source_id, destination, dest_system, dest_code, dest_display,
                    unit, unit_system, unit_code, subject_ref, focus_ref, specimen_ref)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (dest.source_id, dest.destination, dest.dest_system, dest.dest_code,
                 dest.dest_display, dest.unit, dest.unit_system, dest.unit_code,
                 dest.subject_ref, dest.focus_ref, dest.specimen_ref)
            )

        # insert content_values
        for cv in self.content_values:
            cur.execute(
                """INSERT INTO content_values
                   (id, source_value, source_category, code, system, display, is_primary)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (cv.id, cv.source_value, cv.source_category, cv.code,
                 cv.system, cv.display, 1 if cv.is_primary else 0)
            )

        # insert content_fhir_targets
        for target in self.content_fhir_targets:
            cur.execute(
                """INSERT INTO content_fhir_targets
                   (id, content_value_id, fhir_path, fhir_resource, context, is_primary)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (target.id, target.content_value_id, target.fhir_path,
                 target.fhir_resource, target.context, 1 if target.is_primary else 0)
            )

        conn.commit()
        conn.close()

    def load_sqlite(self, path: Path) -> None:
        """load from SQLite database."""
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # load sources
        cur.execute("SELECT * FROM sources")
        for row in cur.fetchall():
            self.add_source(Source(
                id=row["id"],
                source=row["source"],
                source_schema=row["source_schema"],
                tier=Tier(row["tier"]),
                source_context=row["source_context"] or "",
            ))

        # load destinations
        cur.execute("SELECT * FROM destinations")
        for row in cur.fetchall():
            self.add_destination(Destination(
                source_id=row["source_id"],
                destination=row["destination"],
                dest_system=row["dest_system"] or "",
                dest_code=row["dest_code"] or "",
                dest_display=row["dest_display"] or "",
                unit=row["unit"] or "",
                unit_system=row["unit_system"] or "",
                unit_code=row["unit_code"] or "",
                subject_ref=row["subject_ref"] or "",
                focus_ref=row["focus_ref"] or "",
                specimen_ref=row["specimen_ref"] or "",
            ))

        # load content_values (if table exists)
        try:
            cur.execute("SELECT * FROM content_values")
            for row in cur.fetchall():
                cv = ContentValue(
                    id=row["id"],
                    source_value=row["source_value"],
                    source_category=row["source_category"],
                    code=row["code"],
                    system=row["system"] or "http://snomed.info/sct",
                    display=row["display"] or "",
                    is_primary=bool(row["is_primary"]),
                )
                self.content_values.append(cv)
        except sqlite3.OperationalError:
            pass  # table doesn't exist in older db

        # load content_fhir_targets (if table exists)
        try:
            cur.execute("SELECT * FROM content_fhir_targets")
            for row in cur.fetchall():
                target = ContentFhirTarget(
                    id=row["id"],
                    content_value_id=row["content_value_id"],
                    fhir_path=row["fhir_path"],
                    fhir_resource=row["fhir_resource"] or "",
                    context=row["context"] or "",
                    is_primary=bool(row["is_primary"]),
                )
                self.content_fhir_targets.append(target)
        except sqlite3.OperationalError:
            pass  # table doesn't exist in older db

        conn.close()