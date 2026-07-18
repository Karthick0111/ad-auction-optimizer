#!/usr/bin/env bash
# Builds the ml_deps Lambda layer (lightgbm + numpy + scipy + libgomp.so.1,
# used only by bid_consumer) inside AWS's official Lambda Python 3.12 base
# image (Amazon Linux 2023) - the only reliable way to guarantee the
# compiled wheels actually match Lambda's real runtime, not just "some
# Linux". producer deliberately needs no compiled-dependency layer at all
# (reads JSONL with the stdlib, not parquet) and run_trigger only needs the
# shared_code layer (pure Python, built inline by Terraform) - see
# lambda.tf's header comment for the full reasoning, including why
# Python 3.12/AL2023 was chosen over 3.11/AL2.
#
# Run this once before the first `terraform apply`, and again any time
# requirements change - Terraform picks up the new zip via its content hash.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

echo "==> Building ml_deps_layer.zip (lightgbm + libgomp)"
docker run --rm --platform linux/amd64 \
  --entrypoint /bin/bash \
  -v "$BUILD_DIR":/build \
  -v "$SCRIPT_DIR/build_scripts":/build_scripts \
  public.ecr.aws/lambda/python:3.12 \
  /build_scripts/build_ml_layer.sh

echo "==> Done:"
ls -lh "$BUILD_DIR"/*.zip
