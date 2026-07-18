#!/usr/bin/env bash
# Packages the SageMaker training entrypoint (script-mode convention: a flat
# source directory SageMaker's SKLearn framework container pip-installs
# requirements.txt into, then runs `python train_ctr_model.py`) and uploads
# it to S3. Run this once the S3 bucket exists (after the first
# `terraform apply`) and again any time model/train_ctr_model.py or
# simulation/ changes.
#
# Usage: ./infra/package_training_code.sh <bucket-name>
set -euo pipefail

BUCKET="${1:?Usage: package_training_code.sh <bucket-name>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp "$REPO_ROOT/model/train_ctr_model.py" "$STAGING/"
cp -r "$REPO_ROOT/simulation" "$STAGING/"

cat > "$STAGING/requirements.txt" <<EOF
lightgbm
joblib
pandas
pyarrow
scikit-learn
scipy
EOF

tar -czf "$STAGING/sourcedir.tar.gz" -C "$STAGING" train_ctr_model.py simulation requirements.txt
aws s3 cp "$STAGING/sourcedir.tar.gz" "s3://$BUCKET/code/sourcedir.tar.gz"

echo "==> Uploaded s3://$BUCKET/code/sourcedir.tar.gz"
