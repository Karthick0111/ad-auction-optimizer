# Three Lambda functions, two dependency layers:
#   - shared_code layer: simulation/ (auction.py, bandit.py, schema.py) +
#     common.py - pure Python, no C extensions, used by all three functions.
#   - ml_deps layer: lightgbm + numpy + scipy + libgomp.so.1 - only
#     bid_consumer needs this (it's the only function that scores the CTR
#     model). producer deliberately reads the holdout set as JSONL instead
#     of parquet specifically so it doesn't need a pandas/pyarrow layer -
#     pyarrow alone unzips to 200MB+, which combined with pandas/numpy blew
#     past Lambda's 250MB unzipped-size limit when first tried.
#
# Runtime is python3.12: Lambda's Python 3.11 base image runs on Amazon
# Linux 2 (glibc 2.26), too old for current numpy/lightgbm manylinux
# wheels (they required a from-source build with no compiler available).
# Python 3.12's base image runs Amazon Linux 2023 (glibc 2.34), which
# actually has prebuilt wheels available.
#
# The ml_deps zip is built by infra/build_layers.sh (Docker, matching
# Lambda's actual Linux runtime) rather than by Terraform itself - see that
# script's header comment for why.

locals {
  build_dir = "${path.module}/build"
  runtime   = "python3.12"
}

data "archive_file" "shared_code_layer" {
  type        = "zip"
  output_path = "${local.build_dir}/shared_code_layer.zip"

  source {
    content  = file("${path.module}/../lambda_functions/common.py")
    filename = "python/common.py"
  }
  source {
    content  = file("${path.module}/../simulation/__init__.py")
    filename = "python/simulation/__init__.py"
  }
  source {
    content  = file("${path.module}/../simulation/auction.py")
    filename = "python/simulation/auction.py"
  }
  source {
    content  = file("${path.module}/../simulation/bandit.py")
    filename = "python/simulation/bandit.py"
  }
  source {
    content  = file("${path.module}/../simulation/schema.py")
    filename = "python/simulation/schema.py"
  }
}

resource "aws_lambda_layer_version" "shared_code" {
  layer_name          = "${var.project_name}-shared-code"
  filename            = data.archive_file.shared_code_layer.output_path
  source_code_hash    = data.archive_file.shared_code_layer.output_base64sha256
  compatible_runtimes = [local.runtime]
}

# ml_deps_layer.zip (~67MB) is over the ~70MB hard limit for direct
# PublishLayerVersion uploads, so it has to go through S3 instead - the
# standard route for any Lambda package/layer past that size, using the
# same artifacts bucket the rest of the project already uses.
#
# etag is deliberately NOT set to filemd5(...): S3 uses multipart upload
# for a file this size, which produces a hash-of-part-hashes ETag (e.g.
# "...-14"), never a plain file MD5 - so a filemd5() comparison would never
# match and would force a pointless re-upload + a cascading Lambda layer
# replacement on every single `terraform apply`, even with no real change.
# Ignoring etag means re-running ./infra/build_layers.sh with genuinely new
# dependencies won't auto-trigger a re-upload - `terraform taint
# aws_s3_object.ml_deps_layer` (or delete build/ml_deps_layer.zip and
# rebuild) forces one when that's actually needed.
resource "aws_s3_object" "ml_deps_layer" {
  bucket = aws_s3_bucket.artifacts.bucket
  key    = "lambda-layers/ml_deps_layer.zip"
  source = "${local.build_dir}/ml_deps_layer.zip"

  lifecycle {
    ignore_changes = [etag]
  }
}

resource "aws_lambda_layer_version" "ml_deps" {
  layer_name          = "${var.project_name}-ml-deps"
  s3_bucket           = aws_s3_object.ml_deps_layer.bucket
  s3_key              = aws_s3_object.ml_deps_layer.key
  s3_object_version   = aws_s3_object.ml_deps_layer.version_id
  compatible_runtimes = [local.runtime]
}

# --- bid_consumer ---

data "archive_file" "bid_consumer" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda_functions/bid_consumer"
  output_path = "${local.build_dir}/bid_consumer.zip"
}

resource "aws_lambda_function" "bid_consumer" {
  function_name = "${var.project_name}-bid-consumer"
  role          = aws_iam_role.bid_consumer.arn
  handler       = "handler.handler"
  runtime       = local.runtime
  timeout       = 30
  memory_size   = 512

  filename         = data.archive_file.bid_consumer.output_path
  source_code_hash = data.archive_file.bid_consumer.output_base64sha256

  layers = [aws_lambda_layer_version.shared_code.arn, aws_lambda_layer_version.ml_deps.arn]

  environment {
    variables = {
      DYNAMODB_RUNS_TABLE   = aws_dynamodb_table.simulation_runs.name
      DYNAMODB_EVENTS_TABLE = aws_dynamodb_table.simulation_events.name
      DYNAMODB_BANDIT_TABLE = aws_dynamodb_table.bandit_state.name
      S3_BUCKET             = aws_s3_bucket.artifacts.bucket
      MODEL_S3_KEY          = "models/latest/ctr_model.txt"
      LD_LIBRARY_PATH       = "/opt/python/lib"
    }
  }
}

resource "aws_lambda_event_source_mapping" "bid_consumer_kinesis" {
  event_source_arn  = aws_kinesis_stream.bid_requests.arn
  function_name     = aws_lambda_function.bid_consumer.arn
  starting_position = "LATEST"
  batch_size        = 50
}

# --- producer ---

data "archive_file" "producer" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda_functions/producer"
  output_path = "${local.build_dir}/producer.zip"
}

resource "aws_lambda_function" "producer" {
  function_name = "${var.project_name}-producer"
  role          = aws_iam_role.producer.arn
  handler       = "handler.handler"
  runtime       = local.runtime
  timeout       = 300 # up to 5000 impressions in batches of 500 - generous headroom
  memory_size   = 512

  filename         = data.archive_file.producer.output_path
  source_code_hash = data.archive_file.producer.output_base64sha256

  layers = [aws_lambda_layer_version.shared_code.arn]

  environment {
    variables = {
      S3_BUCKET           = aws_s3_bucket.artifacts.bucket
      HOLDOUT_S3_KEY      = "models/latest/holdout.jsonl"
      KINESIS_STREAM_NAME = aws_kinesis_stream.bid_requests.name
    }
  }
}

# --- run_trigger ---

data "archive_file" "run_trigger" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda_functions/run_trigger"
  output_path = "${local.build_dir}/run_trigger.zip"
}

resource "aws_lambda_function" "run_trigger" {
  function_name = "${var.project_name}-run-trigger"
  role          = aws_iam_role.run_trigger.arn
  handler       = "handler.handler"
  runtime       = local.runtime
  timeout       = 10
  memory_size   = 256

  filename         = data.archive_file.run_trigger.output_path
  source_code_hash = data.archive_file.run_trigger.output_base64sha256

  layers = [aws_lambda_layer_version.shared_code.arn]

  environment {
    variables = {
      DYNAMODB_RUNS_TABLE    = aws_dynamodb_table.simulation_runs.name
      DYNAMODB_BANDIT_TABLE  = aws_dynamodb_table.bandit_state.name
      PRODUCER_FUNCTION_NAME = aws_lambda_function.producer.function_name
    }
  }
}
