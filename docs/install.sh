#!/usr/bin/env sh
# ragrep installer — https://github.com/fntune/ragrep
#
# Usage:
#   curl -fsSL https://ragrep.cc/install.sh | sh
#
# What it does:
#   1. Installs uv if it's not already on PATH
#   2. Installs ragrep with the [full] extras (FAISS, embeddings, retrieval)
#
# Read this script before running it: https://ragrep.cc/install.sh

set -eu

REPO_URL="https://github.com/fntune/ragrep"
PKG_SPEC="ragrep[full] @ git+${REPO_URL}"

info() { printf '\033[1;32m==>\033[0m %s\n' "$1"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$1" 1>&2; exit 1; }

# 1. Ensure uv is installed (ragrep installs into a uv-managed tool venv)
if ! command -v uv >/dev/null 2>&1; then
    info "uv not found — installing it from astral.sh first..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv visible to this script after its installer finishes
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    command -v uv >/dev/null 2>&1 || fail "uv installed but is not on PATH. Open a new shell and re-run."
fi

# 2. Install ragrep with the full local-pipeline extras
info "Installing ragrep from ${REPO_URL} ..."
uv tool install --force "${PKG_SPEC}"

info "Installed."
printf '\n'
info "Next steps"
printf '   1. Create %s with your API keys\n' "${HOME}/.config/ragrep/.env"
printf '      Template: %s/blob/main/.env.example\n' "${REPO_URL}"
printf '   2. Try:  ragrep --help\n'
printf '\n'
info "Docs: https://ragrep.cc"
