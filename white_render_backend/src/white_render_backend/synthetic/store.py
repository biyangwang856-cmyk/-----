"""Durable single-host task ledger. Paid submissions have a separate intent record.

The SQLite file belongs on local disk shared by API and worker on the SAME host,
not NFS. Dataset workers never see this queue. Each write uses BEGIN IMMEDIATE;
lease tokens fence filesystem publication and the paid submission decision.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from ..domain import require
from ..util import canonical, digest, safe_id, timestamp


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                  project TEXT NOT NULL, kind TEXT NOT NULL, id TEXT NOT NULL,
                  payload TEXT NOT NULL, created_at TEXT NOT NULL,
                  PRIMARY KEY(project,kind,id));
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, project TEXT NOT NULL, dedupe TEXT UNIQUE NOT NULL,
                  fingerprint TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL,
                  phase TEXT NOT NULL, result TEXT NOT NULL, error TEXT, cancel INTEGER NOT NULL DEFAULT 0,
                  lease TEXT, lease_until REAL NOT NULL DEFAULT 0, next_run REAL NOT NULL DEFAULT 0,
                  attempt INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS synthetic_claim ON jobs(state,next_run,created_at);
                CREATE TABLE IF NOT EXISTS attempts (
                  job_id TEXT PRIMARY KEY, project TEXT NOT NULL, intent_id TEXT NOT NULL,
                  phase TEXT NOT NULL, task_id TEXT, payload TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS quote_uses (quote_id TEXT PRIMARY KEY, job_id TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        value = dict(row)
        for name in ("payload", "result", "error"):
            if name in value and value[name] is not None:
                value[name] = json.loads(value[name])
        return value

    def put(self, project, kind, identity, payload):
        safe_id(project); safe_id(identity)
        with self.connect(True) as db:
            self._put(db, project, kind, identity, payload)
        return self.get(project, kind, identity)

    @staticmethod
    def _put(db, project, kind, identity, payload):
        text = canonical(payload)
        old = db.execute("SELECT payload FROM objects WHERE project=? AND kind=? AND id=?",
                         (project, kind, identity)).fetchone()
        require(old is None or old["payload"] == text, "IMMUTABLE_ASSET", "Existing record differs", status=409)
        if old is None:
            db.execute("INSERT INTO objects VALUES(?,?,?,?,?)", (project, kind, identity, text, timestamp()))

    def get(self, project, kind, identity):
        safe_id(project); safe_id(identity)
        with self.connect() as db:
            row = db.execute("SELECT * FROM objects WHERE project=? AND kind=? AND id=?",
                             (project, kind, identity)).fetchone()
        require(row is not None, "NOT_FOUND", f"{kind}/{identity}", status=404)
        return self.decode(row)

    def list(self, project, kind):
        safe_id(project)
        with self.connect() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM objects WHERE project=? AND kind=? ORDER BY created_at DESC", (project, kind))]

    def submit(self, project, payload, key, *, quote_id=None):
        safe_id(project)
        require(isinstance(key, str) and 1 <= len(key) <= 200, "IDEMPOTENCY_KEY", "Provide Idempotency-Key (1–200 characters)")
        fingerprint = digest(payload)
        dedupe = digest([project, key])
        identity = "synjob_" + uuid4().hex
        with self.connect(True) as db:
            old = db.execute("SELECT * FROM jobs WHERE dedupe=?", (dedupe,)).fetchone()
            if old:
                require(old["fingerprint"] == fingerprint, "IDEMPOTENCY_CONFLICT", "Key reused with different request", status=409)
                return self.decode(old)
            if quote_id:
                quote = db.execute("SELECT payload FROM objects WHERE project=? AND kind='quote' AND id=?",
                                   (project, quote_id)).fetchone()
                require(quote is not None, "NOT_FOUND", "Quote not found", status=404)
                value = json.loads(quote["payload"])
                require(value["expires_at_epoch"] > time.time(), "QUOTE_EXPIRED", "Create a fresh quote", status=409)
                require(db.execute("SELECT 1 FROM quote_uses WHERE quote_id=?", (quote_id,)).fetchone() is None,
                        "QUOTE_USED", "Each quote authorizes only one generation job", status=409)
                db.execute("INSERT INTO quote_uses VALUES(?,?)", (quote_id, identity))
            now = timestamp()
            db.execute("INSERT INTO jobs(id,project,dedupe,fingerprint,payload,state,phase,result,created_at,updated_at) "
                       "VALUES(?,?,?,?,?,'queued','queued','{}',?,?)",
                       (identity, project, dedupe, fingerprint, canonical(payload), now, now))
        return self.job(project, identity)

    def job(self, project, identity):
        safe_id(project); safe_id(identity)
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE project=? AND id=?", (project, identity)).fetchone()
        require(row is not None, "NOT_FOUND", "Job not found", status=404)
        return self.decode(row)

    def jobs(self, project):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM jobs WHERE project=? ORDER BY created_at DESC", (project,))]

    def claim(self, lease_seconds=90):
        now = time.time()
        with self.connect(True) as db:
            db.execute("UPDATE jobs SET state='cancelled',lease=NULL,updated_at=? WHERE cancel=1 AND "
                       "(state='queued' OR (state='running' AND lease_until<?))", (timestamp(), now))
            row = db.execute("SELECT * FROM jobs WHERE cancel=0 AND "
                             "((state='queued' AND next_run<=?) OR (state='running' AND lease_until<?)) "
                             "ORDER BY next_run,created_at LIMIT 1", (now, now)).fetchone()
            if not row:
                return None
            token = uuid4().hex
            db.execute("UPDATE jobs SET state='running',lease=?,lease_until=?,attempt=attempt+1,updated_at=? WHERE id=?",
                       (token, now + lease_seconds, timestamp(), row["id"]))
            result = self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())
        return result

    @staticmethod
    def _owned(db, job, *, allow_cancel=False):
        row = db.execute("SELECT * FROM jobs WHERE id=? AND project=?", (job["id"], job["project"])).fetchone()
        require(row and row["state"] == "running" and row["lease"] == job["lease"] and row["lease_until"] > time.time(),
                "LEASE_LOST", "Worker no longer owns this job", status=409)
        if not allow_cancel:
            require(not row["cancel"], "CANCELLED", "Cancellation requested", status=409)
        return row

    def pulse(self, job, phase=None):
        with self.connect(True) as db:
            self._owned(db, job)
            db.execute("UPDATE jobs SET lease_until=?,phase=COALESCE(?,phase),updated_at=? WHERE id=?",
                       (time.time()+90, phase, timestamp(), job["id"]))

    def publish(self, job, asset_id, asset, result_key):
        with self.connect(True) as db:
            row = self._owned(db, job)
            self._put(db, job["project"], "asset", asset_id, asset)
            result = json.loads(row["result"])
            result[result_key] = asset_id
            db.execute("UPDATE jobs SET result=?,updated_at=? WHERE id=?", (canonical(result), timestamp(), job["id"]))

    def add_result(self, job, **values):
        with self.connect(True) as db:
            row = self._owned(db, job)
            result = json.loads(row["result"])
            result.update(values)
            db.execute("UPDATE jobs SET result=?,updated_at=? WHERE id=?", (canonical(result), timestamp(), job["id"]))

    def finish(self, job, *, state="succeeded", error=None, phase=None):
        with self.connect(True) as db:
            row = self._owned(db, job, allow_cancel=True)
            if row["cancel"]:
                state = "cancelled"
            db.execute("UPDATE jobs SET state=?,phase=?,error=?,lease=NULL,lease_until=0,updated_at=? WHERE id=?",
                       (state, phase or state, canonical(error) if error else None, timestamp(), job["id"]))

    def defer(self, job, seconds=5):
        with self.connect(True) as db:
            self._owned(db, job)
            db.execute("UPDATE jobs SET state='queued',phase='provider_pending',lease=NULL,lease_until=0,next_run=?,updated_at=? WHERE id=?",
                       (time.time()+seconds, timestamp(), job["id"]))

    def attempt(self, project, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM attempts WHERE project=? AND job_id=?", (project, job_id)).fetchone()
        return self.decode(row)

    def begin_submission(self, job, audit):
        """Persist BEFORE the only chargeable POST. Any abandoned intent is uncertain."""
        with self.connect(True) as db:
            self._owned(db, job)
            old = db.execute("SELECT * FROM attempts WHERE job_id=?", (job["id"],)).fetchone()
            require(old is None, "SUBMISSION_EXISTS", "Cannot submit this job again", status=409)
            intent = uuid4().hex
            db.execute("INSERT INTO attempts VALUES(?,?,?,'intent',NULL,?,?)",
                       (job["id"], job["project"], intent, canonical(audit), timestamp()))
            db.execute("UPDATE jobs SET phase='submitting',updated_at=? WHERE id=?", (timestamp(), job["id"]))
        return intent

    def receipt(self, project, job_id, intent_id, phase, task_id=None, details=None):
        """Record an external receipt even if cancellation/lease expiry happened during POST."""
        with self.connect(True) as db:
            old = db.execute("SELECT * FROM attempts WHERE job_id=? AND project=?", (job_id, project)).fetchone()
            require(old and old["intent_id"] == intent_id, "INTENT_MISMATCH", "Submission intent differs", status=409)
            require(not old["task_id"] or not task_id or old["task_id"] == task_id,
                    "TASK_ID_CONFLICT", "Provider task already bound", status=409)
            payload = json.loads(old["payload"])
            payload.update(details or {})
            db.execute("UPDATE attempts SET phase=?,task_id=COALESCE(?,task_id),payload=?,updated_at=? WHERE job_id=?",
                       (phase, task_id, canonical(payload), timestamp(), job_id))

    def cancel(self, project, job_id):
        with self.connect(True) as db:
            row = db.execute("SELECT * FROM jobs WHERE project=? AND id=?", (project, job_id)).fetchone()
            require(row is not None, "NOT_FOUND", "Job not found", status=404)
            require(row["state"] in {"running", "queued", "submission_unknown"}, "JOB_STATE", "Job cannot be cancelled", status=409)
            db.execute("UPDATE jobs SET cancel=1,state=?,updated_at=? WHERE id=?",
                       ("running" if row["state"] == "running" else "cancelled", timestamp(), job_id))
        return self.job(project, job_id)

    def retry(self, project, job_id):
        with self.connect(True) as db:
            row = db.execute("SELECT * FROM jobs WHERE project=? AND id=?", (project, job_id)).fetchone()
            require(row is not None, "NOT_FOUND", "Job not found", status=404)
            require(row["state"] in {"failed", "cancelled"}, "JOB_STATE", "Only failed/cancelled jobs can resume", status=409)
            attempt = db.execute("SELECT * FROM attempts WHERE job_id=?", (job_id,)).fetchone()
            if attempt:
                require(attempt["task_id"] and attempt["phase"] not in {"failed", "rejected"},
                        "PAID_RETRY_FORBIDDEN", "Unknown submission needs reconciliation; failed generation needs a new quote/job", status=409)
            db.execute("UPDATE jobs SET state='queued',phase='resuming',cancel=0,error=NULL,lease=NULL,lease_until=0,next_run=0,updated_at=? WHERE id=?",
                       (timestamp(), job_id))
        return self.job(project, job_id)

    def reconcile(self, project, job_id, task_id, note):
        with self.connect(True) as db:
            row = db.execute("SELECT * FROM jobs WHERE project=? AND id=?", (project, job_id)).fetchone()
            attempt = db.execute("SELECT * FROM attempts WHERE project=? AND job_id=?", (project, job_id)).fetchone()
            require(row and attempt, "NOT_FOUND", "No submission to reconcile", status=404)
            require(row["state"] in {"submission_unknown", "cancelled", "failed"} and not attempt["task_id"],
                    "JOB_STATE", "Only an unbound, inactive submission can be reconciled", status=409)
            payload = json.loads(attempt["payload"])
            payload["manual_reconciliation"] = {"note": note, "at": timestamp(), "input_match": "operator_attested"}
            db.execute("UPDATE attempts SET task_id=?,phase='submitted',payload=?,updated_at=? WHERE job_id=?",
                       (task_id, canonical(payload), timestamp(), job_id))
            db.execute("UPDATE jobs SET state='queued',phase='reconciled',cancel=0,error=NULL,next_run=0,updated_at=? WHERE id=?",
                       (timestamp(), job_id))
        return self.job(project, job_id)
