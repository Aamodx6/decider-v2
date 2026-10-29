# Cell 13 - the monitor: poll describe_training_job for every recorded job that is InProgress or
# Stopping, until each is terminal; then download its artifacts under runs/<job>/ with the
# sketchy-file skip (a file already on disk with the same ContentLength is not re-downloaded);
# then CHECK final_step vs budget. A Completed run that ends off-budget fails HERE with a
# diagnostic and the receipt of where to look - that is the manager cell's whole reason to exist.
# SageMaker has no delete API: Stopping -> Stopped is the terminal form of "killed it ourselves".
import datetime

POLL_SECONDS = 60

def describe(name):
    return sm.describe_training_job(TrainingJobName=name)

def key_prefix(uri):
    """s3://bucket/key/... -> key/...   list_objects_v2 takes a KEY prefix, not a URI."""
    if uri.startswith("s3://"):
        bucket, _sep, key = uri[len("s3://"):].partition("/")
        assert bucket == BUCKET, f"prefix bucket {bucket} != {BUCKET}"
        return key
    return uri

def download_prefix(prefix, local_dir):
    """Copy every object under prefix to local_dir (skip: present + same size). Returns (new, total)."""
    prefix = key_prefix(prefix)
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        keys.extend(page.get("Contents", []))
    n_new = 0
    base = prefix.rstrip("/")
    for obj in keys:
        rel = obj["Key"][len(base):].lstrip("/")
        dst = local_dir / rel
        if dst.exists() and dst.stat().st_size == obj["Size"]:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, obj["Key"], str(dst))
        n_new += 1
    return n_new, len(keys)

pending = [n for n in state["jobs"]
           if describe(n)["TrainingJobStatus"] in ("InProgress", "Stopping")]

while pending:
    for name in list(pending):
        d = describe(name)
        status = d["TrainingJobStatus"]
        state["jobs"][name].update({
            "status": status,
            "secondary_status": d.get("SecondaryStatus", ""),
            "checked": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        if status == "Failed":
            print(f"[FAIL] {name}: {d.get('FailureReason', '')}")
            save_state()
            raise SystemExit(f"job {name} Failed - see its CloudWatch log; the run cannot continue")
        if status in ("Completed", "Stopped"):
            pending.remove(name)
    save_state()
    if not pending:
        break
    print("waiting:", ", ".join(n for n in pending), "----------------------")
    time.sleep(POLL_SECONDS)

# downloads + the final_step check, per terminal job
TERMINALS = [n for n, meta in state["jobs"].items()
             if describe(n)["TrainingJobStatus"] in ("Completed", "Stopped", "Failed")]
for name in sorted(TERMINALS):
    meta = state["jobs"][name]
    # Read the live status: the polling loop above only revisits jobs it saw as InProgress, so a
    # job that failed between launch and this pass still carries its recorded "InProgress" here.
    live = describe(name)["TrainingJobStatus"]
    meta["status"] = live
    if meta["arm"].startswith("eval-"):
        print(f"[eval] {name}: eval job - its result json is pulled inside cell 14")
        continue
    local = RUNS_LOCAL / name
    new_out, tot_out = download_prefix(f"{S3_RUNS}/{name}/output/", local / "output")
    new_ckpt, tot_ckpt = download_prefix(f"{S3_CKP}/{meta['base_job_name']}/{name}/checkpoints/", local / "checkpoints")
    print(f"[download] {name}/{meta['arm']}: output {new_out}/{tot_out} new, ckpt {new_ckpt}/{tot_ckpt} new")

    summary_path = local / "output" / "data" / "summary.json"
    if live == "Stopped":
        print(f"[stopped] {name}: partial run (our own stop or spot) - artifacts kept, no budget verdict")
        continue
    if live == "Failed":
        print(f"[FAIL] {name}: {describe(name).get('FailureReason', '')} - the job failed before writing "
              "summary.json; read its CloudWatch log (preflight/argv errors land there first)")
        continue
    if not summary_path.exists():
        raise SystemExit(f"NOT RESOLVED: {name} says Completed but runs/{name}/output/data/summary.json "
                         "did not download - check the S3 output prefix and this kernel's S3 permissions")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final_step, total_steps = summary.get("final_step"), summary.get("total_steps")
    expected = meta["budget"]["steps"]
    if final_step != expected:
        raise SystemExit(
            f"NOT RESOLVED: {name} ended off-budget - final_step={final_step} vs "
            f"total_steps={total_steps} (state budget says {expected}). The job says Completed, so "
            "this is NOT a crash: likely the epochs cap truncated it (cell 5 warned), or the "
            "config changed mid-run. Inspect runs/<job>/output/data/metrics.jsonl tail and the "
            "CONFIG banner in the CloudWatch log before re-checking the verifier.")
    print(f"[verify] {name}: final_step={final_step} == budget {expected}  best_val_acc="
          f"{summary.get('best_val_acc')}")
save_state()
