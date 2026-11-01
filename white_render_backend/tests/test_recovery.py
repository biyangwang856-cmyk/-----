"""Crashes/cancellation around the paid request boundary and local publication."""
import pytest
from white_render_backend.domain import EngineError
from white_render_backend.synthetic.store import Store


def create(tmp_path):
    store = Store(tmp_path/"tasks.sqlite")
    store.submit("p",{"mode":"scene_and_render"},"job")
    return store, store.claim()


def test_cancel_during_post_still_saves_receipt_and_resume_only_queries(tmp_path):
    store, job = create(tmp_path)
    intent = store.begin_submission(job,{"source_asset_id":"source"})
    store.cancel("p",job["id"])
    store.receipt("p",job["id"],intent,"submitted",task_id="123")
    with pytest.raises(EngineError,match="CANCELLED"):store.defer(job)
    store.finish(job,state="failed")
    assert store.job("p",job["id"])["state"] == "cancelled"
    store.retry("p",job["id"])
    resumed = store.claim()
    assert store.attempt("p",resumed["id"])["task_id"] == "123"
    with pytest.raises(EngineError,match="SUBMISSION_EXISTS"):store.begin_submission(resumed,{})


def test_cancel_after_intent_without_receipt_cannot_create_new_paid_attempt(tmp_path):
    store, job = create(tmp_path)
    store.begin_submission(job,{})
    store.cancel("p",job["id"])
    store.finish(job,state="submission_unknown")
    with pytest.raises(EngineError,match="PAID_RETRY_FORBIDDEN"):store.retry("p",job["id"])
    assert store.claim() is None


def test_expired_worker_cannot_publish_and_missing_source_survives_recovery(tmp_path):
    store, job = create(tmp_path)
    store.publish(job,"asset_source",{"kind":"source"},"source_asset_id")
    with store.connect(True) as db:db.execute("UPDATE jobs SET lease_until=0 WHERE id=?",(job["id"],))
    replacement = store.claim()
    assert replacement["result"]["source_asset_id"] == "asset_source"
    with pytest.raises(EngineError,match="LEASE_LOST"):store.publish(job,"asset_stale",{},"render_asset_id")
    with pytest.raises(EngineError,match="NOT_FOUND"):store.get("p","asset","asset_stale")
    store.finish(replacement)
    assert store.job("p",job["id"])["state"] == "succeeded"


def test_quote_reservation_is_atomic_across_repeated_clicks(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import time
    store = Store(tmp_path/"tasks.sqlite")
    store.put("p","quote","quote_1",{"expires_at_epoch":time.time()+100})
    def submit(i):
        try:return store.submit("p",{"mode":"scene_and_render"},str(i),quote_id="quote_1")["id"]
        except EngineError as exc:return exc.code
    with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(submit,range(4)))
    assert values.count("QUOTE_USED") == 3
    assert sum(v.startswith("synjob_") for v in values) == 1


def test_expired_quote_blocks_new_job_but_idempotent_receipt_still_reads(tmp_path):
    import time
    store = Store(tmp_path/"tasks.sqlite")
    store.put("p","quote","quote_1",{"expires_at_epoch":time.time()+100})
    first = store.submit("p",{"mode":"scene_and_render"},"one",quote_id="quote_1")
    with store.connect(True) as db:
        db.execute("UPDATE objects SET payload=? WHERE id='quote_1'",('{"expires_at_epoch":0}',))
    assert store.submit("p",{"mode":"scene_and_render"},"one",quote_id="quote_1")["id"] == first["id"]
    with pytest.raises(EngineError,match="QUOTE_EXPIRED"):store.submit("p",{},"two",quote_id="quote_1")
