"""feedback store for HITL decisions - enables learning loop."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


# default path relative to this file: schema_crush/data/db/feedback.db
DEFAULT_FEEDBACK_DB = Path(__file__).parent.parent / "data" / "db" / "feedback.db"


class Decision(str, Enum):
    ACCEPT = "accept"      # proposed mapping was correct
    REJECT = "reject"      # proposed mapping was wrong, no correction provided
    CORRECT = "correct"    # proposed mapping was wrong, correction provided


@dataclass
class Feedback:
    """single HITL feedback record."""
    id: int
    source: str                      # source term (ex, "sample_id")
    source_context: str              # context (ex., "demographic")
    proposed_target: str             # what the system proposed
    ground_truth: Optional[str]      # correct target (if CORRECT decision)
    decision: Decision               # accept/reject/correct
    matcher: str                     # which matcher made the proposal
    tier: str                        # entity/field/content
    confidence: float                # confidence at time of decision
    timestamp: datetime              # when decision was made
    user_id: str                     # who made the decision (for multi-user)
    notes: str                       # optional notes


class FeedbackStore:
    """SQLite-backed store for HITL feedback."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_FEEDBACK_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                source_context TEXT DEFAULT '',
                proposed_target TEXT NOT NULL,
                ground_truth TEXT,
                decision TEXT NOT NULL,
                matcher TEXT NOT NULL,
                tier TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT DEFAULT 'default',
                notes TEXT DEFAULT ''
            )
        """)

        # indexes for common queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_source ON feedback(source)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_decision ON feedback(decision)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_matcher ON feedback(matcher)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tier ON feedback(tier)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")

        conn.commit()
        conn.close()

    def record(
        self,
        source: str,
        proposed_target: str,
        decision: Decision | str,
        matcher: str,
        tier: str,
        confidence: float,
        ground_truth: Optional[str] = None,
        source_context: str = "",
        user_id: str = "default",
        notes: str = "",
    ) -> int:
        """record a HITL decision. returns feedback id."""
        if isinstance(decision, str):
            decision = Decision(decision)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """INSERT INTO feedback
               (source, source_context, proposed_target, ground_truth, decision,
                matcher, tier, confidence, timestamp, user_id, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source,
                source_context,
                proposed_target,
                ground_truth,
                decision.value,
                matcher,
                tier,
                confidence,
                datetime.utcnow().isoformat(),
                user_id,
                notes,
            )
        )

        feedback_id = cur.lastrowid
        conn.commit()
        conn.close()
        return feedback_id

    def get_all(self, limit: int = 1000) -> list[Feedback]:
        """get all feedback records."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )

        results = [self._row_to_feedback(row) for row in cur.fetchall()]
        conn.close()
        return results

    def get_corrections(self) -> list[Feedback]:
        """get all corrections (for retraining)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM feedback WHERE decision = ? ORDER BY timestamp",
            (Decision.CORRECT.value,)
        )

        results = [self._row_to_feedback(row) for row in cur.fetchall()]
        conn.close()
        return results

    def get_by_matcher(self, matcher: str) -> list[Feedback]:
        """get feedback for a specific matcher."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM feedback WHERE matcher = ? ORDER BY timestamp",
            (matcher,)
        )

        results = [self._row_to_feedback(row) for row in cur.fetchall()]
        conn.close()
        return results

    def get_by_tier(self, tier: str) -> list[Feedback]:
        """get feedback for a specific tier."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM feedback WHERE tier = ? ORDER BY timestamp",
            (tier,)
        )

        results = [self._row_to_feedback(row) for row in cur.fetchall()]
        conn.close()
        return results

    def stats(self) -> dict:
        """get feedback statistics."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # total counts by decision
        cur.execute("""
            SELECT decision, COUNT(*) as count
            FROM feedback
            GROUP BY decision
        """)
        by_decision = {row[0]: row[1] for row in cur.fetchall()}

        # counts by matcher
        cur.execute("""
            SELECT matcher, decision, COUNT(*) as count
            FROM feedback
            GROUP BY matcher, decision
        """)
        by_matcher = {}
        for row in cur.fetchall():
            matcher, decision, count = row
            if matcher not in by_matcher:
                by_matcher[matcher] = {}
            by_matcher[matcher][decision] = count

        # counts by tier
        cur.execute("""
            SELECT tier, decision, COUNT(*) as count
            FROM feedback
            GROUP BY tier, decision
        """)
        by_tier = {}
        for row in cur.fetchall():
            tier, decision, count = row
            if tier not in by_tier:
                by_tier[tier] = {}
            by_tier[tier][decision] = count

        # total
        cur.execute("SELECT COUNT(*) FROM feedback")
        total = cur.fetchone()[0]

        conn.close()

        return {
            "total": total,
            "by_decision": by_decision,
            "by_matcher": by_matcher,
            "by_tier": by_tier,
        }

    def export_for_calibration(self) -> list[dict]:
        """export feedback in format suitable for calibrator retraining.

        returns list of:
            {matcher, tier, confidence, was_correct}
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM feedback")

        results = []
        for row in cur.fetchall():
            # accept = correct, reject/correct = wrong
            was_correct = row["decision"] == Decision.ACCEPT.value
            results.append({
                "matcher": row["matcher"],
                "tier": row["tier"],
                "confidence": row["confidence"],
                "was_correct": was_correct,
                "source": row["source"],
                "source_context": row["source_context"],
            })

        conn.close()
        return results

    def export_new_mappings(self) -> list[dict]:
        """export corrections as new mappings for knowledge base.

        returns list of:
            {source, source_context, target, tier}
        """
        corrections = self.get_corrections()
        return [
            {
                "source": f.source,
                "source_context": f.source_context,
                "target": f.ground_truth,
                "tier": f.tier,
            }
            for f in corrections
            if f.ground_truth  # only if correction was provided
        ]

    def _row_to_feedback(self, row: sqlite3.Row) -> Feedback:
        """convert sqlite row to Feedback dataclass."""
        return Feedback(
            id=row["id"],
            source=row["source"],
            source_context=row["source_context"] or "",
            proposed_target=row["proposed_target"],
            ground_truth=row["ground_truth"],
            decision=Decision(row["decision"]),
            matcher=row["matcher"],
            tier=row["tier"],
            confidence=row["confidence"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            user_id=row["user_id"] or "default",
            notes=row["notes"] or "",
        )

    def clear(self):
        """clear all feedback (use with caution)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM feedback")
        conn.commit()
        conn.close()