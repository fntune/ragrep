#!/usr/bin/env sh
# ragrep installer — https://github.com/fntune/ragrep
#
# Usage:
#   curl -fsSL https://ragrep.cc/install.sh | sh
#
# What it does:
#   1. Detects your OS + arch (uname).
#   2. Downloads the matching prebuilt binary from GitHub Releases.
#   3. Verifies the sha256 checksum.
#   4. Installs to ~/.local/bin/ragrep.
#
# Use --legacy to install the old Python distribution via uv:
#   curl -fsSL https://ragrep.cc/install.sh | sh -s -- --legacy
#
# Read this script before running it: https://ragrep.cc/install.sh

set -eu

REPO="fntune/ragrep"
RELEASES_URL="https://github.com/${REPO}/releases"
INSTALL_DIR="${HOME}/.local/bin"

info()  { printf '\033[1;32m==>\033[0m %s\n' "$1"; }
warn()  { printf '\033[1;33mwarn:\033[0m %s\n' "$1" 1>&2; }
fail()  { printf '\033[1;31merror:\033[0m %s\n' "$1" 1>&2; exit 1; }

# ---- legacy (Python via uv) ----------------------------------------------
legacy_install() {
    info "Installing the legacy Python distribution via uv."
    if ! command -v uv >/dev/null 2>&1; then
        info "uv not found — installing it from astral.sh first..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
        command -v uv >/dev/null 2>&1 || fail "uv installed but not on PATH. Open a new shell and re-run."
    fi
    uv tool install --force "ragrep[full] @ git+https://github.com/${REPO}"
    info "Installed (legacy/Python). Try: ragrep --help"
    exit 0
}

case "${1:-}" in
    --legacy) legacy_install ;;
    --help|-h)
        printf 'ragrep installer\n\n'
        printf 'Usage:  curl -fsSL https://ragrep.cc/install.sh | sh\n'
        printf '        curl -fsSL https://ragrep.cc/install.sh | sh -s -- --legacy\n'
        exit 0
        ;;
esac

# ---- detect platform -----------------------------------------------------
detect_target() {
    os="$(uname -s)"
    arch="$(uname -m)"
    case "${os}" in
        Linux)
            case "${arch}" in
                x86_64|amd64)        echo "x86_64-unknown-linux-gnu" ;;
                aarch64|arm64)       echo "aarch64-unknown-linux-gnu" ;;
                *) fail "unsupported Linux arch: ${arch}" ;;
            esac
            ;;
        Darwin)
            case "${arch}" in
                x86_64)              echo "x86_64-apple-darwin" ;;
                arm64)               echo "aarch64-apple-darwin" ;;
                *) fail "unsupported macOS arch: ${arch}" ;;
            esac
            ;;
        *) fail "unsupported OS: ${os} (Windows: download from ${RELEASES_URL})" ;;
    esac
}

TARGET="$(detect_target)"
info "Detected target: ${TARGET}"

# ---- resolve latest release ----------------------------------------------
# Use GitHub's redirect-to-latest pattern: /releases/latest/download/<asset>
# avoids needing the API or jq.
ASSET="ragrep-${TARGET}.tar.gz"
URL="${RELEASES_URL}/latest/download/${ASSET}"
SHA_URL="${URL}.sha256"

# ---- download + verify ---------------------------------------------------
TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT INT TERM

info "Downloading ${ASSET} ..."
curl -fsSL --output "${TMPDIR}/${ASSET}" "${URL}" \
    || fail "download failed: ${URL}"

info "Fetching checksum ..."
curl -fsSL --output "${TMPDIR}/${ASSET}.sha256" "${SHA_URL}" \
    || fail "checksum download failed: ${SHA_URL}"

info "Verifying checksum ..."
expected="$(awk '{print $1}' < "${TMPDIR}/${ASSET}.sha256")"
if command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "${TMPDIR}/${ASSET}" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${TMPDIR}/${ASSET}" | awk '{print $1}')"
else
    fail "no sha256 tool found (need shasum or sha256sum)"
fi
[ "${expected}" = "${actual}" ] \
    || fail "sha256 mismatch: expected ${expected}, got ${actual}"

# ---- install -------------------------------------------------------------
info "Extracting and installing to ${INSTALL_DIR}/ragrep ..."
mkdir -p "${INSTALL_DIR}"
tar -xzf "${TMPDIR}/${ASSET}" -C "${TMPDIR}"
mv "${TMPDIR}/ragrep" "${INSTALL_DIR}/ragrep"
chmod +x "${INSTALL_DIR}/ragrep"

# ---- PATH check + next steps ---------------------------------------------
case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *) warn "${INSTALL_DIR} is not on PATH. Add to your shell rc:
        export PATH=\"${INSTALL_DIR}:\$PATH\"" ;;
esac

info "Installed: $("${INSTALL_DIR}/ragrep" --version 2>/dev/null || echo "${INSTALL_DIR}/ragrep")"
printf '\n'
info "Next steps"
printf '   1. Create %s/.config/ragrep/.env with your API keys\n' "${HOME}"
printf '      Template: https://github.com/%s/blob/main/.env.example\n' "${REPO}"
printf '   2. Try:  ragrep "your question"\n'
printf '   3. Or run as a server: ragrep serve\n'
printf '\n'
info "Docs: https://ragrep.cc"
