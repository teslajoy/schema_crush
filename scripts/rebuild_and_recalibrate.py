#!/usr/bin/env python3
"""rebuild flat_mappings.db and recalibrate matchers.

steps:
1. backup current db
2. rebuild from JSON sources (optional)
3. recalibrate all matchers
4. copy calibrators to package location
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "schema_crush" / "data" / "db" / "flat_mappings.db"
CALIBRATORS_SRC = Path(__file__).parent.parent / "calibrators" / "pkl"
CALIBRATORS_DST = Path(__file__).parent.parent / "schema_crush" / "data" / "calibrators" / "pkl"


def backup_db():
    """backup current database."""
    if not DB_PATH.exists():
        print("no database to backup")
        return None

    backup_path = DB_PATH.parent / f"flat_mappings.db.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(DB_PATH, backup_path)
    print(f"backed up to: {backup_path}")
    return backup_path


def rebuild_db():
    """rebuild database from JSON sources."""
    from schema_crush.mappings.flat_loader import load_flat_mappings

    print("rebuilding database from JSON sources...")
    db = load_flat_mappings(force_rebuild=True)
    stats = db.stats()

    print(f"  sources: {stats['counts']['sources']}")
    print(f"  destinations: {stats['counts']['destinations']}")
    print(f"  schemas: {stats['schemas']}")

    return db


def recalibrate(use_expert_embeddings: bool = False):
    """run calibration on current database."""
    print("\nrunning calibration...")
    print("=" * 70)

    # import and run calibration
    calibrators_dir = Path(__file__).parent.parent / "calibrators"
    sys.path.insert(0, str(calibrators_dir))

    from calibrate_flat_mappings import main as calibrate_main

    calibrators, summary_df, results_df = calibrate_main(use_expert_embeddings=use_expert_embeddings)

    return calibrators, summary_df


def copy_calibrators():
    """copy calibrators to package location."""
    print("\ncopying calibrators to package...")

    CALIBRATORS_DST.mkdir(parents=True, exist_ok=True)

    for pkl_file in CALIBRATORS_SRC.glob("*.pkl"):
        dst = CALIBRATORS_DST / pkl_file.name
        shutil.copy(pkl_file, dst)
        print(f"  {pkl_file.name}")

    print(f"copied to: {CALIBRATORS_DST}")


def show_stats():
    """show current db and calibrator stats."""
    import sqlite3

    print("CURRENT STATE")
    print("=" * 70)

    # db stats
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM sources")
        sources = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM destinations")
        dests = cur.fetchone()[0]

        cur.execute("SELECT source_schema, COUNT(*) FROM sources GROUP BY source_schema")
        schemas = dict(cur.fetchall())

        conn.close()

        print(f"\nDatabase: {DB_PATH}")
        print(f"  sources: {sources}")
        print(f"  destinations: {dests}")
        print(f"  schemas: {schemas}")
    else:
        print(f"\nDatabase: NOT FOUND at {DB_PATH}")

    # calibrator stats
    print(f"\nCalibrators (source): {CALIBRATORS_SRC}")
    if CALIBRATORS_SRC.exists():
        for pkl in CALIBRATORS_SRC.glob("*.pkl"):
            size = pkl.stat().st_size
            mtime = datetime.fromtimestamp(pkl.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {pkl.name}: {size:,} bytes ({mtime})")
    else:
        print("  NOT FOUND")

    print(f"\nCalibrators (package): {CALIBRATORS_DST}")
    if CALIBRATORS_DST.exists():
        for pkl in CALIBRATORS_DST.glob("*.pkl"):
            size = pkl.stat().st_size
            mtime = datetime.fromtimestamp(pkl.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {pkl.name}: {size:,} bytes ({mtime})")
    else:
        print("  NOT FOUND")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild DB and recalibrate")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild DB from JSON sources")
    parser.add_argument("--recalibrate", action="store_true", help="Run calibration")
    parser.add_argument("--copy-calibrators", action="store_true", help="Copy calibrators to package")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup")
    parser.add_argument("--expert-embeddings", action="store_true", help="Use expert embeddings")
    parser.add_argument("--stats", action="store_true", help="Show current stats")
    parser.add_argument("--all", action="store_true", help="Do everything: backup, rebuild, recalibrate, copy")

    args = parser.parse_args()

    if args.stats:
        show_stats()
        sys.exit(0)

    if args.all:
        args.rebuild = True
        args.recalibrate = True
        args.copy_calibrators = True

    if not any([args.rebuild, args.recalibrate, args.copy_calibrators]):
        parser.print_help()
        sys.exit(0)

    # backup first
    if not args.no_backup and (args.rebuild or args.recalibrate):
        backup_db()

    # rebuild db
    if args.rebuild:
        rebuild_db()

    # recalibrate
    if args.recalibrate:
        recalibrate(use_expert_embeddings=args.expert_embeddings)

    # copy calibrators
    if args.copy_calibrators:
        copy_calibrators()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)