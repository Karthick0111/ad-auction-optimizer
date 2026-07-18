#!/bin/bash
# Runs inside the Lambda Python 3.11 base image (Amazon Linux 2). numpy is
# pinned explicitly and installed in its own step before lightgbm - numpy's
# newest releases have dropped manylinux2014 wheel support (AL2's old
# glibc can't use them), so installing it first and separately forces pip
# to resolve a compatible 1.x wheel rather than letting a transitive
# resolution pull in a numpy version with no wheel for this environment.
set -euo pipefail
microdnf install -y libgomp
pip install -q --target /build/ml_deps/python "numpy==1.26.4"
pip install -q --target /build/ml_deps/python --no-deps lightgbm==4.5.0
pip install -q --target /build/ml_deps/python scipy
mkdir -p /build/ml_deps/python/lib
cp /usr/lib64/libgomp.so.1 /build/ml_deps/python/lib/
python3 /build_scripts/zip_helper.py /build/ml_deps/python /build/ml_deps_layer.zip
