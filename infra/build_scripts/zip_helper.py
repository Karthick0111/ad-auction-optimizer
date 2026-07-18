"""Zips a directory into a Lambda Layer archive using Python's zipfile module
(the `zip` CLI isn't present in the Lambda base image). Usage:
    python3 zip_helper.py <source_dir> <output_zip>
Preserves the parent directory name as a path prefix (e.g. zipping
/build/ml_deps/python produces entries under "python/...", which is the
prefix Lambda Layers require)."""
import os
import sys
import zipfile

src, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(src):
        for f in files:
            full = os.path.join(root, f)
            zf.write(full, os.path.relpath(full, os.path.dirname(src)))
print(f"Wrote {out}")
