#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./release_pypi.sh 0.1.2
#
# Optional env vars:
#   PYPI_TOKEN=...            # preferred
#   PACKAGE_NAME=inferscale   # default: inferscale
#   SKIP_UPLOAD=1             # build/check only
#
# Example:
#   chmod +x release_pypi.sh
#   PYPI_TOKEN=pypi-xxxx ./release_pypi.sh 0.1.2

PACKAGE_NAME="${PACKAGE_NAME:-inferscale}"
NEW_VERSION="${1:-}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Error: pyproject.toml not found. Run this from the project root."
  exit 1
fi

if [[ -z "$NEW_VERSION" ]]; then
  echo "Usage: $0 <version>"
  echo "Example: $0 0.1.2"
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python not found in PATH."
  exit 1
fi

echo "==> Releasing ${PACKAGE_NAME} version ${NEW_VERSION}"

echo "==> Ensuring required tools are installed"
python -m pip install --upgrade build twine >/dev/null

echo "==> Updating version in pyproject.toml"
python - <<PY
from pathlib import Path
import re
import sys

path = Path("pyproject.toml")
text = path.read_text(encoding="utf-8")

new_version = "${NEW_VERSION}"

pattern = r'(^version\s*=\s*")([^"]+)(")'
updated, n = re.subn(pattern, rf'\g<1>{new_version}\g<3>', text, count=1, flags=re.MULTILINE)

if n == 0:
    print("Error: Could not find version field in pyproject.toml")
    sys.exit(1)

path.write_text(updated, encoding="utf-8")
print(f"Updated version to {new_version}")
PY

echo "==> Cleaning old build artifacts"
rm -rf dist build ./*.egg-info

echo "==> Building package"
python -m build

echo "==> Checking distributions"
python -m twine check dist/*

echo "==> Built artifacts:"
ls -lh dist/

if [[ "$SKIP_UPLOAD" == "1" ]]; then
  echo "==> SKIP_UPLOAD=1, stopping before upload"
  exit 0
fi

if [[ -z "${PYPI_TOKEN:-}" ]]; then
  echo "Error: PYPI_TOKEN is not set."
  echo "Run like:"
  echo "  PYPI_TOKEN=pypi-xxxx $0 ${NEW_VERSION}"
  exit 1
fi

echo "==> Uploading to PyPI"
TWINE_USERNAME="__token__" \
TWINE_PASSWORD="${PYPI_TOKEN}" \
python -m twine upload dist/*

echo "==> Done"
echo "Verify with:"
echo "  pip install --upgrade ${PACKAGE_NAME}==${NEW_VERSION}"