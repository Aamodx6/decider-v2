# Cell 18 - cleanup / kill switch. SageMaker has NO delete-training-job API: a hanging run can
# only be STOPPED, which is terminal. This cell stops (and then records) every job this notebook
# has in an InProgress or Stopping state, both arms plus ablations; anything Completed/Failed
# stays untouched so the receipts are auditable.
for name in sorted(state["jobs"]):
    live = sm.describe_training_job(TrainingJobName=name)["TrainingJobStatus"]
    meta = state["jobs"][name]
    if live in ("InProgress", "Stopping"):
        print(f"[stop] {name} ({meta['arm']}) in state {live}")
        sm.stop_training_job(TrainingJobName=name)
        meta["status"] = "Stopping"
        meta["stopped_by"] = "cell 18 cleanup"
    elif meta.get("status") != live:
        meta["status"] = live
        print(f"[sync] {name}: recorded {live}")
save_state()
print("cleanup pass complete - re-run cell 13 once each stopped job turns Stopped and the partial "
     "artifacts get downloaded (no budget verdict for partial runs)")
