# Single bucket for everything the pipeline reads/writes: raw + processed
# training data, model artifacts, and the holdout impression stream the
# producer Lambda replays. Kept as one bucket with prefixes (raw/, processed/,
# models/) rather than several - nothing here needs separate access policies
# per data type at this scale.

resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project_name}-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Disabled" # portfolio demo - not worth the extra storage cost
  }
}
