variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix used for naming all resources"
  type        = string
  default     = "ad-auction-optimizer"
}

variable "sagemaker_training_instance_type" {
  description = "Instance type for the CTR model SageMaker Training Job - kept small since the dataset is a 200k-row sample"
  type        = string
  default     = "ml.m5.large"
}
