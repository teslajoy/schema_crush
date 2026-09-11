#!/usr/bin/env python3
"""cleanup invalid mappings identified by audit.

issues fixed:
1. content-tier entries where destination = source (no FHIR path)
2. field-tier entries with invalid destinations (ResearchStudy, cases_submitter_id)
3. orphan sources after destination cleanup
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "schema_crush" / "data" / "db" / "flat_mappings.db"


def cleanup_content_tier_self_mappings(db_path: Path, dry_run: bool = True):
    """delete content-tier entries where destination = source (useless mappings)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # find entries where destination equals source and tier is content
    cur.execute("""
        SELECT d.id, s.source, d.destination, s.source_schema
        FROM sources s
        JOIN destinations d ON s.id = d.source_id
        WHERE s.tier = 'content' AND d.destination = s.source
    """)
    rows = cur.fetchall()

    print(f"\n1. CONTENT-TIER SELF-MAPPINGS (destination = source)")
    print(f"   found {len(rows)} entries to delete")

    if rows and not dry_run:
        cur.execute("""
            DELETE FROM destinations
            WHERE id IN (
                SELECT d.id FROM destinations d
                JOIN sources s ON d.source_id = s.id
                WHERE s.tier = 'content' AND d.destination = s.source
            )
        """)
        print(f"   deleted {cur.rowcount} destination entries")
        conn.commit()
    elif rows:
        for row in rows[:10]:
            print(f"      {row[1]} ({row[3]}) -> {row[2]}")
        if len(rows) > 10:
            print(f"      ... and {len(rows) - 10} more")

    conn.close()
    return len(rows)


def cleanup_content_tier_no_dot(db_path: Path, dry_run: bool = True):
    """delete content-tier entries where destination has no dot (not a FHIR path)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # find content entries without dots that aren't already caught by self-mapping
    cur.execute("""
        SELECT d.id, s.source, d.destination, s.source_schema
        FROM sources s
        JOIN destinations d ON s.id = d.source_id
        WHERE s.tier = 'content'
          AND d.destination NOT LIKE '%.%'
          AND d.destination != s.source
    """)
    rows = cur.fetchall()

    print(f"\n2. CONTENT-TIER NO-DOT DESTINATIONS (not FHIR paths)")
    print(f"   found {len(rows)} entries to delete")

    if rows and not dry_run:
        cur.execute("""
            DELETE FROM destinations
            WHERE id IN (
                SELECT d.id FROM destinations d
                JOIN sources s ON d.source_id = s.id
                WHERE s.tier = 'content'
                  AND d.destination NOT LIKE '%.%'
                  AND d.destination != s.source
            )
        """)
        print(f"   deleted {cur.rowcount} destination entries")
        conn.commit()
    elif rows:
        for row in rows[:10]:
            print(f"      {row[1]} ({row[3]}) -> {row[2]}")
        if len(rows) > 10:
            print(f"      ... and {len(rows) - 10} more")

    conn.close()
    return len(rows)


def fix_field_tier_research_study(db_path: Path, dry_run: bool = True):
    """fix field-tier entries with 'ResearchStudy' destination (should be entity tier)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT s.id, s.source, s.tier, d.destination
        FROM sources s
        JOIN destinations d ON s.id = d.source_id
        WHERE s.tier = 'field' AND d.destination = 'ResearchStudy'
    """)
    rows = cur.fetchall()

    print(f"\n3. FIELD-TIER 'ResearchStudy' DESTINATIONS")
    print(f"   found {len(rows)} entries (should be entity tier)")

    if rows and not dry_run:
        # update tier from field to entity
        cur.execute("""
            UPDATE sources SET tier = 'entity'
            WHERE tier = 'field' AND id IN (
                SELECT s.id FROM sources s
                JOIN destinations d ON s.id = d.source_id
                WHERE d.destination = 'ResearchStudy'
            )
        """)
        print(f"   updated {cur.rowcount} sources to entity tier")
        conn.commit()
    elif rows:
        for row in rows:
            print(f"      {row[1]} (tier={row[2]}) -> {row[3]}")

    conn.close()
    return len(rows)


def fix_cases_submitter_id(db_path: Path, dry_run: bool = True):
    """fix 'cases_submitter_id' destination to proper FHIR path."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT d.id, s.source, d.destination, s.source_context
        FROM sources s
        JOIN destinations d ON s.id = d.source_id
        WHERE d.destination = 'cases_submitter_id'
    """)
    rows = cur.fetchall()

    print(f"\n4. 'cases_submitter_id' DESTINATIONS")
    print(f"   found {len(rows)} entries to fix -> Patient.identifier")

    if rows and not dry_run:
        cur.execute("""
            UPDATE destinations
            SET destination = 'Patient.identifier'
            WHERE destination = 'cases_submitter_id'
        """)
        print(f"   updated {cur.rowcount} destinations")
        conn.commit()
    elif rows:
        for row in rows:
            print(f"      {row[1]} (context={row[3]}) -> {row[2]}")

    conn.close()
    return len(rows)


def fix_custom_fhir_paths(db_path: Path, dry_run: bool = True):
    """fix custom FHIR paths like Observation.DocumentReference.xxx."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # find custom paths
    cur.execute("""
        SELECT DISTINCT d.destination, COUNT(*)
        FROM destinations d
        WHERE d.destination LIKE 'Observation.DocumentReference.%'
           OR d.destination LIKE 'Observation.%.%' AND d.destination NOT LIKE 'Observation.value%'
           AND d.destination NOT LIKE 'Observation.code%'
           AND d.destination NOT LIKE 'Observation.component%'
           AND d.destination NOT LIKE 'Observation.identifier%'
        GROUP BY d.destination
    """)
    rows = cur.fetchall()

    print(f"\n5. CUSTOM FHIR PATHS (Observation.DocumentReference.xxx)")
    print(f"   found {len(rows)} distinct patterns")

    if rows:
        for row in rows[:10]:
            print(f"      {row[0]}: {row[1]} entries")

    # for now just report - these need manual review
    conn.close()
    return len(rows)


def cleanup_orphan_sources(db_path: Path, dry_run: bool = True):
    """delete sources with no destinations."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT s.id, s.source, s.source_schema
        FROM sources s
        LEFT JOIN destinations d ON s.id = d.source_id
        WHERE d.id IS NULL
    """)
    orphans = cur.fetchall()

    print(f"\n6. ORPHAN SOURCES (no destinations)")
    print(f"   found {len(orphans)} orphan sources")

    if orphans and not dry_run:
        cur.execute("""
            DELETE FROM sources
            WHERE id NOT IN (SELECT DISTINCT source_id FROM destinations)
        """)
        print(f"   deleted {cur.rowcount} orphan sources")
        conn.commit()
    elif orphans:
        for row in orphans[:5]:
            print(f"      {row[1]} ({row[2]})")
        if len(orphans) > 5:
            print(f"      ... and {len(orphans) - 5} more")

    conn.close()
    return len(orphans)


def show_summary(db_path: Path):
    """show current DB state."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM sources")
    sources = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM destinations")
    dests = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM destinations WHERE destination LIKE '%.%'")
    valid_paths = cur.fetchone()[0]

    cur.execute("""
        SELECT tier, COUNT(*) FROM sources GROUP BY tier ORDER BY tier
    """)
    tiers = cur.fetchall()

    print(f"\n{'=' * 60}")
    print("DATABASE STATE")
    print(f"{'=' * 60}")
    print(f"  sources:      {sources}")
    print(f"  destinations: {dests}")
    print(f"  valid paths:  {valid_paths}")
    print(f"  tiers: {dict(tiers)}")

    conn.close()


def run_cleanup(db_path: Path, dry_run: bool = True):
    """run all cleanup steps."""
    print("=" * 60)
    print(f"CLEANUP {'(DRY RUN)' if dry_run else '(APPLYING CHANGES)'}")
    print("=" * 60)

    total_fixed = 0

    total_fixed += cleanup_content_tier_self_mappings(db_path, dry_run)
    total_fixed += cleanup_content_tier_no_dot(db_path, dry_run)
    total_fixed += fix_field_tier_research_study(db_path, dry_run)
    total_fixed += fix_cases_submitter_id(db_path, dry_run)
    fix_custom_fhir_paths(db_path, dry_run)  # report only

    if not dry_run:
        cleanup_orphan_sources(db_path, dry_run)

    show_summary(db_path)

    if dry_run:
        print(f"\n[DRY RUN] would fix {total_fixed} entries")
        print("run with --apply to make changes")

    return total_fixed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup invalid mappings")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")

    args = parser.parse_args()

    db_path = Path(args.db)
    dry_run = not args.apply

    run_cleanup(db_path, dry_run)