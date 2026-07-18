# All three tables use PAY_PER_REQUEST (on-demand) billing - no capacity to
# provision/tune, and effectively free at this project's scale (well within
# DynamoDB's free tier). TTL on both tables auto-expires old demo runs after
# 30 days instead of accumulating storage indefinitely.

resource "aws_dynamodb_table" "simulation_runs" {
  name         = "${var.project_name}-simulation-runs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"

  attribute {
    name = "run_id"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled         = true
  }
}

# One item per (run_id, sequence) - a full audit trail of every impression
# processed in a run, which is what the dashboard's live charts read from.
resource "aws_dynamodb_table" "simulation_events" {
  name         = "${var.project_name}-simulation-events"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  range_key    = "sequence"

  attribute {
    name = "run_id"
    type = "S"
  }

  attribute {
    name = "sequence"
    type = "N"
  }

  ttl {
    attribute_name = "expires_at"
    enabled         = true
  }
}

# Separate from simulation_runs so the hot read-modify-write path (every
# single impression) only touches this small item, not the larger run
# record - keeps contention/latency down under Kinesis-driven concurrency.
resource "aws_dynamodb_table" "bandit_state" {
  name         = "${var.project_name}-bandit-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"

  attribute {
    name = "run_id"
    type = "S"
  }
}
