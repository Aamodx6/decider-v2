# Cell 4 - session, role, clients, bucket. Region is pinned on the session so a stray default
# profile cannot drag API calls to the wrong account/region pair (the classic way to "lose" a
# training job: it runs fine, the notebook silently watches the wrong endpoint).
import datetime as dt

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session, default_bucket=BUCKET)
account = boto_session.client("sts").get_caller_identity()["Account"]
ROLE = ROLE_ARN or sagemaker.get_execution_role(sagemaker_session)

sm      = boto_session.client("sagemaker")
s3      = boto_session.client("s3")
logs    = boto_session.client("logs")                                       # filter_log_events (cell 10)
pricing = boto_session.client("pricing", region_name="us-east-1")           # the API only serves us-east-1

# bucket: create on need (us-east-1 takes no LocationConstraint), then verify access
try:
    s3.head_bucket(Bucket=BUCKET)
except s3.exceptions.NoSuchBucket:  # us-east-1 takes no LocationConstraint; others do
    kwargs = {} if REGION == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": REGION}}
    s3.create_bucket(Bucket=BUCKET, **kwargs)
    print(f"created bucket {BUCKET}")
s3.head_bucket(Bucket=BUCKET)

S3_DATA = f"s3://{BUCKET}/{ARM_TAG}/data"          # /train | /val | /calib | /test  -> questions.parquet
S3_RUNS = f"s3://{BUCKET}/{ARM_TAG}/runs"          #  output_path of the training jobs
S3_CKP  = f"s3://{BUCKET}/{ARM_TAG}/checkpoints"   #  checkpoint_s3_uri (spot restarts + the final copy)
S3_EVAL = f"s3://{BUCKET}/{ARM_TAG}/eval"          #  output_path of the evaluate.py jobs (cell 14)

RUNS_LOCAL.mkdir(exist_ok=True)

print(f"account {account}  role {ROLE}")
print(f"S3_DATA {S3_DATA}\nS3_RUNS {S3_RUNS}\nS3_CKP  {S3_CKP}\nS3_EVAL {S3_EVAL}")
