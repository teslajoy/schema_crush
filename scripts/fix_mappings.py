#!/usr/bin/env python3
"""fix invalid mappings in flat_mappings.db.

provides tools to:
- update destinations (single or batch)
- delete bad mappings
- add new mappings
- apply fixes from CSV
"""

import sqlite3
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "schema_crush" / "data" / "db" / "flat_mappings.db"


def update_destination(db_path: Path, old_dest: str, new_dest: str, dry_run: bool = True):
    """update all occurrences of a destination path.

    args:
        old_dest: current destination path
        new_dest: new destination path
        dry_run: if True, show what would change without changing
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # find affected rows
    cur.execute("""
        SELECT d.id, s.source, s.source_schema, d.destination
        FROM destinations d
        JOIN sources s ON d.source_id = s.id
        WHERE d.destination = ?
    """, (old_dest,))

    rows = cur.fetchall()

    if not rows:
        print(f"no destinations found matching: {old_dest}")
        conn.close()
        return 0

    print(f"found {len(rows)} destinations to update:")
    for row in rows[:10]:
        print(f"  {row[1]} ({row[2]}) -> {old_dest}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")

    if dry_run:
        print(f"\n[DRY RUN] would update to: {new_dest}")
        print("run with --apply to make changes")
    else:
        cur.execute("UPDATE destinations SET destination = ? WHERE destination = ?",
                   (new_dest, old_dest))
        conn.commit()
        print(f"\nupdated {cur.rowcount} rows")

    conn.close()
    return len(rows)


def delete_by_destination(db_path: Path, dest_pattern: str, dry_run: bool = True):
    """delete mappings by destination pattern.

    args:
        dest_pattern: SQL LIKE pattern (use % for wildcard)
        dry_run: if True, show what would be deleted
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # find affected rows
    cur.execute("""
        SELECT d.id, s.id as source_id, s.source, s.source_schema, d.destination
        FROM destinations d
        JOIN sources s ON d.source_id = s.id
        WHERE d.destination LIKE ?
    """, (dest_pattern,))

    rows = cur.fetchall()

    if not rows:
        print(f"no destinations found matching: {dest_pattern}")
        conn.close()
        return 0

    print(f"found {len(rows)} destinations to delete:")
    for row in rows[:20]:
        print(f"  {row[2]} ({row[3]}) -> {row[4]}")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more")

    if dry_run:
        print(f"\n[DRY RUN] would delete these mappings")
        print("run with --apply to make changes")
    else:
        # delete destinations
        cur.execute("DELETE FROM destinations WHERE destination LIKE ?", (dest_pattern,))
        deleted_dests = cur.rowcount

        # optionally delete orphan sources
        cur.execute("""
            DELETE FROM sources
            WHERE id NOT IN (SELECT DISTINCT source_id FROM destinations)
        """)
        deleted_sources = cur.rowcount

        conn.commit()
        print(f"\ndeleted {deleted_dests} destinations, {deleted_sources} orphan sources")

    conn.close()
    return len(rows)


def delete_by_schema(db_path: Path, schema: str, dry_run: bool = True):
    """delete all mappings for a schema.

    args:
        schema: source_schema to delete (e.g., 'cda', 'icgc')
        dry_run: if True, show what would be deleted
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) FROM sources WHERE source_schema = ?
    """, (schema,))
    source_count = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM destinations d
        JOIN sources s ON d.source_id = s.id
        WHERE s.source_schema = ?
    """, (schema,))
    dest_count = cur.fetchone()[0]

    if source_count == 0:
        print(f"no sources found for schema: {schema}")
        conn.close()
        return 0

    print(f"schema '{schema}':")
    print(f"  sources: {source_count}")
    print(f"  destinations: {dest_count}")

    if dry_run:
        print(f"\n[DRY RUN] would delete all {schema} mappings")
        print("run with --apply to make changes")
    else:
        # delete destinations first (FK constraint)
        cur.execute("""
            DELETE FROM destinations
            WHERE source_id IN (SELECT id FROM sources WHERE source_schema = ?)
        """, (schema,))

        # delete sources
        cur.execute("DELETE FROM sources WHERE source_schema = ?", (schema,))

        conn.commit()
        print(f"\ndeleted {schema} schema completely")

    conn.close()
    return source_count


def apply_fixes_from_csv(db_path: Path, csv_path: Path, dry_run: bool = True):
    """apply fixes from a CSV file.

    CSV format:
        old_destination,new_destination,action
        Observation.code.nci_tumor_grade,Observation.valueCodeableConcept,update
        Some.bad.path,,delete

    actions:
        update: change old_destination to new_destination
        delete: remove mappings with old_destination
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    updates = 0
    deletes = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            old_dest = row.get("old_destination", "").strip()
            new_dest = row.get("new_destination", "").strip()
            action = row.get("action", "update").strip().lower()

            if not old_dest:
                continue

            if action == "delete" or (action == "update" and not new_dest):
                # delete
                if dry_run:
                    cur.execute("SELECT COUNT(*) FROM destinations WHERE destination = ?", (old_dest,))
                    cnt = cur.fetchone()[0]
                    print(f"[DELETE] {old_dest}: {cnt} rows")
                    deletes += cnt
                else:
                    cur.execute("DELETE FROM destinations WHERE destination = ?", (old_dest,))
                    deletes += cur.rowcount

            elif action == "update" and new_dest:
                # update
                if dry_run:
                    cur.execute("SELECT COUNT(*) FROM destinations WHERE destination = ?", (old_dest,))
                    cnt = cur.fetchone()[0]
                    print(f"[UPDATE] {old_dest} -> {new_dest}: {cnt} rows")
                    updates += cnt
                else:
                    cur.execute("UPDATE destinations SET destination = ? WHERE destination = ?",
                               (new_dest, old_dest))
                    updates += cur.rowcount

    if not dry_run:
        conn.commit()
        # clean up orphan sources
        cur.execute("""
            DELETE FROM sources
            WHERE id NOT IN (SELECT DISTINCT source_id FROM destinations)
        """)
        conn.commit()

    conn.close()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  updates: {updates}")
    print(f"  deletes: {deletes}")

    return updates + deletes


def add_mapping(
    db_path: Path,
    source: str,
    destination: str,
    schema: str,
    tier: str = "field",
    context: str = "",
    dry_run: bool = True
):
    """add a new mapping."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    source_id = f"{schema}:{source}"
    if context:
        source_id = f"{schema}:{context}.{source}"

    # check if source exists
    cur.execute("SELECT id FROM sources WHERE id = ?", (source_id,))
    existing = cur.fetchone()

    if dry_run:
        if existing:
            print(f"[DRY RUN] would add destination to existing source: {source_id}")
        else:
            print(f"[DRY RUN] would create new source: {source_id}")
        print(f"  source: {source}")
        print(f"  destination: {destination}")
        print(f"  schema: {schema}")
        print(f"  tier: {tier}")
        print(f"  context: {context}")
    else:
        if not existing:
            cur.execute("""
                INSERT INTO sources (id, source, source_schema, tier, source_context)
                VALUES (?, ?, ?, ?, ?)
            """, (source_id, source, schema, tier, context))

        cur.execute("""
            INSERT INTO destinations (source_id, destination)
            VALUES (?, ?)
        """, (source_id, destination))

        conn.commit()
        print(f"added: {source} -> {destination}")

    conn.close()


def show_source(db_path: Path, source_term: str):
    """show all mappings for a source term."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT s.id, s.source, s.source_schema, s.tier, s.source_context,
               d.destination, d.dest_system, d.dest_code
        FROM sources s
        JOIN destinations d ON s.id = d.source_id
        WHERE s.source = ? OR s.source LIKE ?
        ORDER BY s.source_schema
    """, (source_term, f"%{source_term}%"))

    rows = cur.fetchall()

    if not rows:
        print(f"no mappings found for: {source_term}")
    else:
        print(f"mappings for '{source_term}':\n")
        for row in rows:
            print(f"  {row['id']}")
            print(f"    source: {row['source']}")
            print(f"    schema: {row['source_schema']}")
            print(f"    context: {row['source_context']}")
            print(f"    tier: {row['tier']}")
            print(f"    destination: {row['destination']}")
            if row['dest_system']:
                print(f"    system: {row['dest_system']}")
            print()

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix flat_mappings.db")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # update command
    update_parser = subparsers.add_parser("update", help="Update destination path")
    update_parser.add_argument("old_dest", help="Current destination")
    update_parser.add_argument("new_dest", help="New destination")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete by destination pattern")
    delete_parser.add_argument("pattern", help="Destination pattern (use %% for wildcard)")

    # delete-schema command
    del_schema_parser = subparsers.add_parser("delete-schema", help="Delete all mappings for a schema")
    del_schema_parser.add_argument("schema", help="Schema to delete (e.g., cda, icgc)")

    # apply-csv command
    csv_parser = subparsers.add_parser("apply-csv", help="Apply fixes from CSV file")
    csv_parser.add_argument("csv_file", help="Path to fixes CSV")

    # add command
    add_parser = subparsers.add_parser("add", help="Add a mapping")
    add_parser.add_argument("source", help="Source term")
    add_parser.add_argument("destination", help="Destination path")
    add_parser.add_argument("--schema", required=True, help="Source schema")
    add_parser.add_argument("--tier", default="field", help="Tier (entity/field/content)")
    add_parser.add_argument("--context", default="", help="Source context")

    # show command
    show_parser = subparsers.add_parser("show", help="Show mappings for a source")
    show_parser.add_argument("source", help="Source term to look up")

    args = parser.parse_args()

    db_path = Path(args.db)
    dry_run = not args.apply

    if args.command == "update":
        update_destination(db_path, args.old_dest, args.new_dest, dry_run)

    elif args.command == "delete":
        delete_by_destination(db_path, args.pattern, dry_run)

    elif args.command == "delete-schema":
        delete_by_schema(db_path, args.schema, dry_run)

    elif args.command == "apply-csv":
        apply_fixes_from_csv(db_path, Path(args.csv_file), dry_run)

    elif args.command == "add":
        add_mapping(db_path, args.source, args.destination, args.schema,
                   args.tier, args.context, dry_run)

    elif args.command == "show":
        show_source(db_path, args.source)

    else:
        parser.print_help()