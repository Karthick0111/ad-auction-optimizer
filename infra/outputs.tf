output "s3_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "kinesis_stream_name" {
  value = aws_kinesis_stream.bid_requests.name
}

output "dynamodb_simulation_runs_table" {
  value = aws_dynamodb_table.simulation_runs.name
}

output "dynamodb_simulation_events_table" {
  value = aws_dynamodb_table.simulation_events.name
}

output "dynamodb_bandit_state_table" {
  value = aws_dynamodb_table.bandit_state.name
}

output "run_trigger_function_name" {
  value = aws_lambda_function.run_trigger.function_name
}

output "step_functions_state_machine_arn" {
  value = aws_sfn_state_machine.train_ctr_model.arn
}

output "streamlit_readonly_access_key_id" {
  value     = aws_iam_access_key.streamlit_readonly.id
  sensitive = true
}

output "streamlit_readonly_secret_access_key" {
  value     = aws_iam_access_key.streamlit_readonly.secret
  sensitive = true
}
