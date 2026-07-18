# Orchestrates the offline CTR training path: one state, a synchronous
# SageMaker CreateTrainingJob call (states waits for the job to actually
# finish, not just kick it off). No SageMaker endpoint is created - the
# trained model lands in S3 and bid_consumer loads it directly, which is
# what keeps this project's standing cost near zero between demo sessions.
#
# The training job itself uses SageMaker's built-in scikit-learn framework
# container in script mode (model/train_ctr_model.py + a requirements.txt
# that pip-installs lightgbm at container start) rather than a custom Docker
# image - avoids an ECR build/push step for what's fundamentally a small,
# fast-training job.

data "aws_sagemaker_prebuilt_ecr_image" "sklearn" {
  repository_name = "sagemaker-scikit-learn"
  image_tag        = "1.2-1-cpu-py3"
}

resource "aws_sfn_state_machine" "train_ctr_model" {
  name     = "${var.project_name}-train-ctr-model"
  role_arn = aws_iam_role.step_functions.arn

  definition = jsonencode({
    Comment = "Trains the CTR model via a SageMaker Training Job and waits for completion"
    StartAt = "TrainCTRModel"
    States = {
      TrainCTRModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync"
        Parameters = {
          "TrainingJobName.$" = "$$.Execution.Name"
          AlgorithmSpecification = {
            TrainingImage     = data.aws_sagemaker_prebuilt_ecr_image.sklearn.registry_path
            TrainingInputMode = "File"
          }
          RoleArn = aws_iam_role.sagemaker_execution.arn
          HyperParameters = {
            sagemaker_program            = "train_ctr_model.py"
            "sagemaker_submit_directory" = "\"s3://${aws_s3_bucket.artifacts.bucket}/code/sourcedir.tar.gz\""
          }
          InputDataConfig = [
            {
              ChannelName = "train"
              DataSource = {
                S3DataSource = {
                  S3DataType             = "S3Prefix"
                  S3Uri                   = "s3://${aws_s3_bucket.artifacts.bucket}/processed/"
                  S3DataDistributionType = "FullyReplicated"
                }
              }
            }
          ]
          OutputDataConfig = {
            S3OutputPath = "s3://${aws_s3_bucket.artifacts.bucket}/models/latest/"
          }
          ResourceConfig = {
            InstanceType   = var.sagemaker_training_instance_type
            InstanceCount  = 1
            VolumeSizeInGB = 10
          }
          StoppingCondition = {
            MaxRuntimeInSeconds = 1800 # 30 min ceiling - a 200k-row LightGBM job takes minutes, this is a safety cap on cost
          }
        }
        End = true
      }
    }
  })
}
