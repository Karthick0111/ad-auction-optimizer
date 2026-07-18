# ON_DEMAND capacity mode deliberately, not provisioned/shard-hour billing:
# provisioned mode runs ~$10-11/month per shard even sitting idle between
# demo sessions, while on-demand bills per request - effectively $0 when
# nobody's running a simulation.
resource "aws_kinesis_stream" "bid_requests" {
  name = "${var.project_name}-bid-requests"

  stream_mode_details {
    stream_mode = "ON_DEMAND"
  }
}
