#!/bin/bash
# Runs inside the Lambda Python base image (Amazon Linux 2023).
#
# numpy and scipy are pinned to versions known to work together (1.26.4 /
# 1.13.1 - the same pair used in requirements.txt/local dev) and installed
# in a SINGLE pip invocation. This matters more than it looks: `pip install
# --target DIR` does not treat DIR as an already-satisfied environment
# across separate invocations - each `pip install --target` call resolves
# dependencies from scratch, blind to whatever's already sitting in DIR
# from a previous call. Installing numpy in one call and scipy in a later,
# separate call let scipy's own unconstrained numpy dependency silently
# fetch a second, different numpy version into the same directory
# alongside the first - both installs "succeeded", pip never complained,
# and the breakage only surfaced as an AttributeError at Lambda runtime
# when whichever numpy's files happened to win the file-overwrite race
# didn't match what scipy was compiled against. Lambda deps are installed
# separately via --no-deps specifically because ITS dependency spec has no
# upper bound on numpy and would happily reintroduce the same problem.
set -euo pipefail
microdnf install -y libgomp
pip install -q --target /build/ml_deps/python "numpy==1.26.4" "scipy==1.13.1"
pip install -q --target /build/ml_deps/python --no-deps lightgbm==4.5.0
mkdir -p /build/ml_deps/python/lib
cp /usr/lib64/libgomp.so.1 /build/ml_deps/python/lib/
python3 /build_scripts/zip_helper.py /build/ml_deps/python /build/ml_deps_layer.zip
