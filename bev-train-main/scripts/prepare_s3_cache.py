"""Mirror the Qwen backbone and the bev-decision dataset into S3, so SageMaker jobs
never touch the HuggingFace Hub at training time.

Why this exists
---------------
`dataset.py` and `tokenization.py` resolve their assets *by Hub name*
(`load_dataset("avbiswas/bev-decision", ...)` and
`AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")`). Restoring the same cache layout
under HF_HOME/hub makes those calls resolve offline with no code change.

Do not commit the dataset to git instead: it is `license: unknown` with no blanket
license asserted, and decider-v2 is a public repo. The upstream card says to consult
each source before redistribution.

Usage
-----
    python scripts/prepare_s3_cache.py upload  --bucket my-bucket --region us-east-1
    python scripts/prepare_s3_cache.py verify  --bucket my-bucket
    python scripts/prepare_s3_cache.py restore --bucket my-bucket --dest /cache/hf/hub

`restore` is what the training container runs. It uses boto3 throughout, so neither
command needs the AWS CLI installed.

Both --cache-dir and --dest are the `hub` directory, not HF_HOME itself:
`snapshot_download(cache_dir=...)` creates `<cache_dir>/models--<repo>`, and HF only
resolves a by-name lookup under `HF_HOME/hub/`. Pointing either at HF_HOME would
mirror the tree one level too high and the offline lookup would miss it.
"""
import argparse
import os
import sys
from pathlib import Path

MODEL_REPO = "Qwen/Qwen3-0.6B"
DATASET_REPO = "avbiswas/bev-decision"
PREFIX = "hf-cache"
MANIFEST = ".s3-cache-manifest.json"

# The dataset's configs (`default`, `hard_50k`, `numeric_temporal`, `skills`,
# `counterfactual_15k`, `all`) are defined in the YAML frontmatter of the repo's
# README.md, not by directory names. `all` is a virtual config listing all five
# parquet paths. If the card is missing, only `load_dataset(name)` with no config
# works and `all` raises. So the mirror must include the card.
REQUIRED_DATASET_FILES = ["README.md"]


def _client(region):
    try:
        import boto3
    except ImportError:
        sys.exit("boto3 is required: pip install boto3")
    return boto3.client("s3", region_name=region)


def default_hub_dir():
    """The `hub` directory inside HF_HOME. Both upload and restore must target this
    exact path: snapshot_download lays the cache out as <dir>/models--<repo>, and a
    by-name lookup only ever looks under $HF_HOME/hub."""
    hf_home = os.environ.get("HF_HOME")
    return os.path.join(hf_home, "hub") if hf_home else os.path.abspath("hf-cache/hub")


def _files_under(root):
    """Every regular file under root, following symlinks (HF cache snapshots are
    symlinks into blobs/, and we want the bytes, not the dangling link).

    Symlinks mean the same content is uploaded twice (once as a blob, once as the
    snapshot entry). That roughly doubles the upload, which is why this script
    prints the size before starting. It is cheaper than trying to reconstruct
    symlinks on restore: a plain-file cache reads fine, and that is also what the
    HF cache looks like on platforms without symlink support.
    """
    return sorted(p for p in root.rglob("*") if p.is_file())


def upload(args):
    from huggingface_hub import snapshot_download

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for repo, repo_type in ((MODEL_REPO, "model"), (DATASET_REPO, "dataset")):
        print(f"[upload] fetching {repo} ({repo_type})", flush=True)
        snapshot_download(repo, repo_type=repo_type, cache_dir=str(cache_dir))

    dataset_dir = next(cache_dir.glob("datasets--*"), None)
    missing = [f for f in REQUIRED_DATASET_FILES
               if dataset_dir and not any(dataset_dir.rglob(f))]
    if missing:
        sys.exit(f"[upload] dataset card missing from the snapshot: {missing}. "
                 f"Without README.md the 'all' config cannot be resolved.")

    files = _files_under(cache_dir)
    total = sum(p.stat().st_size for p in files)
    print(f"[upload] {len(files)} files, {total / 2**30:.2f} GiB (symlinks counted twice)", flush=True)
    if not args.yes and input("continue? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    s3 = _client(args.region)
    for i, path in enumerate(files, 1):
        key = f"{args.prefix or PREFIX}/{path.relative_to(cache_dir).as_posix()}"
        s3.upload_file(str(path), args.bucket, key)
        if i % 100 == 0 or i == len(files):
            print(f"[upload] {i}/{len(files)}", flush=True)

    import json
    manifest = {"files": {str(p.relative_to(cache_dir).as_posix()): p.stat().st_size
                          for p in files}}
    s3.put_object(Bucket=args.bucket, Key=f"{args.prefix or PREFIX}/{MANIFEST}",
                  Body=json.dumps(manifest).encode())
    print(f"[upload] done -> s3://{args.bucket}/{args.prefix or PREFIX}/", flush=True)


def restore(args):
    """Pull the cache down into HF_HOME/hub so by-name lookups resolve offline."""
    s3 = _client(args.region)
    prefix = args.prefix or PREFIX
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=args.bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(f"{prefix}/{MANIFEST}") or key.endswith("/"):
                continue
            rel = key[len(prefix) + 1:]
            if not rel:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(args.bucket, key, str(target))
            count += 1
            if count % 100 == 0:
                print(f"[restore] {count} files", flush=True)

    if not count:
        sys.exit(f"[restore] nothing under s3://{args.bucket}/{prefix}/. Run `upload` first.")

    card = next(dest.glob("datasets--*/snapshots/*/README.md"), None)
    print(f"[restore] {count} files -> {dest}", flush=True)
    print(f"[restore] dataset card: {card or 'NOT FOUND'}", flush=True)
    if card is None:
        sys.exit("[restore] no dataset card restored; the 'all' config will not resolve.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print("[restore] set HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1", flush=True)


def verify(args):
    import json

    s3 = _client(args.region)
    prefix = args.prefix or PREFIX
    try:
        body = s3.get_object(Bucket=args.bucket, Key=f"{prefix}/{MANIFEST}")["Body"].read()
    except s3.exceptions.NoSuchKey:
        sys.exit(f"[verify] no manifest at s3://{args.bucket}/{prefix}/{MANIFEST}. Run `upload`.")
    manifest = json.loads(body)

    actual = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix) + 1:]
            if rel and rel != MANIFEST:
                actual[rel] = obj["Size"]

    missing = sorted(set(manifest["files"]) - set(actual))
    extra = sorted(set(actual) - set(manifest["files"]))
    mismatch = [k for k in set(actual) & set(manifest["files"])
                if actual[k] != manifest["files"][k]]
    print(f"[verify] expected {len(manifest['files'])} objects, found {len(actual)}")
    for label, items in (("missing", missing), ("unexpected", extra), ("size mismatch", mismatch)):
        for item in items[:20]:
            print(f"  {label}: {item}")
    ok = not (missing or mismatch)
    print("[verify] OK" if ok else "[verify] FAILED")
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("upload", "restore", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--bucket", required=True)
        p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
        p.add_argument("--prefix", default=PREFIX)
        p.add_argument("--cache-dir", default=default_hub_dir(),
                       help="HF hub dir to download into (default: $HF_HOME/hub)")
        p.add_argument("--dest", default=default_hub_dir(),
                       help="HF hub dir to restore into (default: $HF_HOME/hub)")
        if name == "upload":
            p.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    {"upload": upload, "restore": restore, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()