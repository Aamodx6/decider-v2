# Cell 7 - upload the four shards, skipping any PUT whose destination object already carries
# the same ContentLength (idempotent re-runs do not resend the train shard).
uploaded, skipped = [], []
SHARD_KEYS = {k: f"{ARM_TAG}/data/{k}/questions.parquet" for k in ("train", "val", "calib", "test")}
for split, key in SHARD_KEYS.items():
    path = DATA_DIR / split / "questions.parquet"
    size = path.stat().st_size
    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
        if head["ContentLength"] == size:
            skipped.append(split)
            continue
        print(f"{split}: S3 has {head['ContentLength']}B, local has {size}B - re-uploading")
    except s3.exceptions.NoSuchKey:
        pass
    s3.upload_file(str(path), BUCKET, key)
    uploaded.append(split)

assert set(SHARD_KEYS) == set(uploaded) | set(skipped), (uploaded, skipped)
print(f"uploaded {uploaded}, skipped (same ContentLength) {skipped}")
print("data URI convention:", ", ".join(f"{S3_DATA}/{k}" for k in SHARD_KEYS))
