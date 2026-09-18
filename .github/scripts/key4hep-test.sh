#!/bin/bash
# Install rootfig on a Key4hep stack, with every runtime dependency from the stack, and run
# the tests. Usage: key4hep-test.sh /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
set -eo pipefail

KEY4HEP_SETUP="$1"
shift  # a sourced script sees the caller's arguments
CVMFS_VENV_REF=42c0aeb90f79e86e7a11dea9ad04eff3cc272d6d
VENV="${VENV:-$(mktemp -d)/venv}"

echo '::group::Set up Key4hep and a venv on top'
source "$KEY4HEP_SETUP"
curl -fsSL -o "${VENV}.cvmfs-venv" \
    "https://raw.githubusercontent.com/jbeirer/cvmfs-venv/${CVMFS_VENV_REF}/cvmfs-venv.sh"
bash "${VENV}.cvmfs-venv" "$VENV"
source "$VENV/bin/activate"
echo '::endgroup::'

echo '::group::Install rootfig with its dependencies from the stack only'
python -m pip install --no-index --no-build-isolation .
python -m pip install pytest-mpl pytest-xdist
echo '::endgroup::'

# The stack registers pytest plugins of its own, e.g. Gaudi's CTest reporter.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg \
    python -m pytest -p xdist -p pytest_mpl -n auto -p no:cacheprovider
