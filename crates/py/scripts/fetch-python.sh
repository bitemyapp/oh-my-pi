#!/usr/bin/env bash
# Fetches the python-build-standalone "full" archive (static libpython +
# stdlib) and generates the build inputs omp-py derives from it:
#   <dest>/python/stdlib.bin       in-memory stdlib blob embedded by omp-py
#   <dest>/python/pyo3-config.txt  static-link config consumed via PYO3_CONFIG_FILE
#   crates/py/THIRD-PARTY-NOTICES.txt (checkout mode) from PYTHON.json,
#                                      its license corpus, and frozen wheels
#
# Usage: fetch-python.sh [dest-dir]
#   dest-dir  directory that receives the `python/` tree; defaults to the
#             repo checkout's `vendor/` when run from a checkout. Consumers
#             building the published crate pass an explicit directory and
#             point PYO3_CONFIG_FILE at <dest>/python/pyo3-config.txt.
#
# Idempotent: re-running regenerates derived artifacts for missing trees only.
#
# On macOS:
#   - <dest>/python         uses freethreaded+debug (machine-code .a, fast dev
#                           linking with default linker, zero LTO overhead)
#   - <dest>/python-release uses freethreaded+pgo+lto (LLVM LTO bitcode,
#                           linked with Homebrew LLD via scripts/ld64.lld for
#                           production release builds; marked with `needs-lld`)
# Linux uses freethreaded+debug for both.
set -euo pipefail

TAG=20260807
VER=3.14.7

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CRATE_DIR=$(dirname "$SCRIPT_DIR")

if [ $# -ge 1 ]; then
	DEST=$(mkdir -p "$1" && cd "$1" && pwd)
	REPO_MODE=
else
	DEST="$CRATE_DIR/../../vendor"
	mkdir -p "$DEST"
	DEST=$(cd "$DEST" && pwd)
	REPO_MODE=1
fi

REQ="$CRATE_DIR/requirements.txt"

prepare_tree() {
	local TRIPLE="$1"
	local BUILD="$2"
	local VENDOR_NAME="$3"
	local NEEDS_LLD="$4"

	local VENDOR="$DEST/$VENDOR_NAME"
	local NAME="cpython-${VER}+${TAG}-${TRIPLE}-${BUILD}-full"
	local URL="https://github.com/astral-sh/python-build-standalone/releases/download/${TAG}/${NAME}.tar.zst"

	local NEEDS_FETCH=1
	if [ -f "$VENDOR/.archive.stamp" ]; then
		local CURRENT_ARCHIVE
		CURRENT_ARCHIVE=$(cat "$VENDOR/.archive.stamp" 2>/dev/null || true)
		if [ "$CURRENT_ARCHIVE" = "$NAME" ]; then
			NEEDS_FETCH=0
		else
			echo "archive mismatch in ${VENDOR_NAME}: expected ${NAME}, found ${CURRENT_ARCHIVE}; refetching..." >&2
		fi
	fi

	if [ "$NEEDS_FETCH" = "1" ]; then
		echo "fetching ${NAME} into ${VENDOR_NAME}..." >&2
		local TMP_EXTRACT
		TMP_EXTRACT=$(mktemp -d "$DEST/.fetch-py.XXXXXX")
		curl -fsSL "$URL" | zstd -d | tar -x -C "$TMP_EXTRACT"
		rm -rf "$VENDOR"
		mv "$TMP_EXTRACT/python" "$VENDOR"
		echo "$NAME" > "$VENDOR/.archive.stamp"
		rm -rf "$TMP_EXTRACT"
	fi
	if [ "$NEEDS_LLD" = "1" ]; then
		touch "$VENDOR/needs-lld"
	else
		rm -f "$VENDOR/needs-lld"
	fi

	local STDLIB_DIR
	STDLIB_DIR=$(dirname "$(echo "$VENDOR"/install/lib/python3.14*/os.py)")
	local CONFIG_LIBS=("$STDLIB_DIR"/config-3.14*/libpython3.14*.a)
	if [ "${#CONFIG_LIBS[@]}" -ne 1 ] || [ ! -f "${CONFIG_LIBS[0]}" ]; then
		echo "error: expected exactly one static libpython under $STDLIB_DIR/config-3.14*" >&2
		exit 1
	fi
	local CONFIG_DIR
	CONFIG_DIR=$(dirname "${CONFIG_LIBS[0]}")
	local LIB_NAME
	LIB_NAME=$(basename "${CONFIG_LIBS[0]}" .a | sed 's/^lib//')
	local EXECUTABLE="$VENDOR/install/bin/python3.14td"
	[ -x "$EXECUTABLE" ] || EXECUTABLE="$VENDOR/install/bin/python3.14t"
	# The archive's interpreter asks for the dynamic loader at its standard
	# path, which NixOS doesn't have (it has a stub there that refuses to
	# run it). Under a Nix toolchain it gets that toolchain's loader.
	if ! "$EXECUTABLE" -c '' 2>/dev/null && [ -f "${NIX_CC:-}/nix-support/dynamic-linker" ]; then
		patchelf --set-interpreter "$(cat "$NIX_CC/nix-support/dynamic-linker")" "$(readlink -f "$EXECUTABLE")"
	fi

	echo "generating ${VENDOR_NAME}/stdlib.bin..." >&2
	"$EXECUTABLE" "$SCRIPT_DIR/pack-pymodules.py" "$STDLIB_DIR" "$VENDOR/stdlib.bin" \
		--prefix '<omp-stdlib>' \
		--exclude lib-dynload test idlelib tkinter turtledemo ensurepip \
		          site-packages __pycache__ 'config-*'

	local BUNDLED="$VENDOR/bundled"
	if grep -qvE '^[[:space:]]*(#|$)' "$REQ"; then
		if ! cmp -s "$REQ" "$BUNDLED/.requirements.stamp" 2>/dev/null; then
			echo "fetching bundled python packages for ${VENDOR_NAME}..." >&2
			local TMP
			TMP=$(mktemp -d "$VENDOR/bundled.XXXXXX")
			trap 'rm -rf "$TMP"' EXIT
			uv pip install --link-mode=copy --python "$EXECUTABLE" --target "$TMP" \
				--only-binary :all: --require-hashes -r "$REQ"
			local NATIVE
			NATIVE=$(find "$TMP" -name '*.so' -o -name '*.dylib' -o -name '*.pyd')
			if [ -n "$NATIVE" ]; then
				echo "error: $REQ pulled native extensions; only pure-Python packages can be" >&2
				echo "frozen — install native wheels into site-packages instead:" >&2
				echo "$NATIVE" >&2
				exit 1
			fi
			cp "$REQ" "$TMP/.requirements.stamp"
			rm -rf "$BUNDLED"
			mv "$TMP" "$BUNDLED"
			trap - EXIT
		fi
	else
		rm -rf "$BUNDLED"
	fi

	echo "generating ${VENDOR_NAME}/pyo3-config.txt..." >&2
	cat > "$VENDOR/pyo3-config.txt" <<EOF
implementation=CPython
version=3.14
shared=false
abi3=false
lib_name=${LIB_NAME}
lib_dir=${CONFIG_DIR}
executable=${EXECUTABLE}
pointer_width=64
build_flags=$(case "$LIB_NAME" in *td) echo "Py_DEBUG,Py_GIL_DISABLED";; *) echo "Py_GIL_DISABLED";; esac)
suppress_build_script_link_lines=false
EOF

	echo "done: ${VENDOR} (${LIB_NAME})" >&2
}

case "$(uname -s):$(uname -m)" in
	Darwin:arm64)
		prepare_tree "aarch64-apple-darwin" "freethreaded+debug" "python" "0"
		prepare_tree "aarch64-apple-darwin" "freethreaded+pgo+lto" "python-release" "1"
		;;
	Linux:x86_64)
		prepare_tree "x86_64-unknown-linux-gnu" "freethreaded+debug" "python" "0"
		;;
	*)
		echo "error: no embedded Python archive configured for $(uname -s) $(uname -m)" >&2
		exit 1
		;;
esac

if [ -n "$REPO_MODE" ]; then
	DEV_EXE="$DEST/python/install/bin/python3.14td"
	[ -x "$DEV_EXE" ] || DEV_EXE="$DEST/python/install/bin/python3.14t"
	"$DEV_EXE" "$SCRIPT_DIR/gen-py-notices.py" "$DEST/python" "$CRATE_DIR/THIRD-PARTY-NOTICES.txt"
fi
