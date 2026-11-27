#!/usr/bin/env bash
# Install matching libtriton-jit-nvidia runtime and development packages from
# a local staging directory when available, otherwise fall back to the
# distribution-matched FlagOS APT repository.
set -euo pipefail

DEPS_DIR="${1:-/tmp/local-deps}"

# Check whether a candidate runtime .deb matches the container architecture,
# distro suite suffix, and native libpython ABI.
deb_matches_host() {
    local deb="$1" host_arch="$2" host_suffix="$3" pyver="$4"
    [ -f "$deb" ] || return 1
    [ "$(dpkg-deb -f "$deb" Package 2>/dev/null || true)" = "libtriton-jit-nvidia" ] || return 1
    [ "$(dpkg-deb -f "$deb" Architecture 2>/dev/null || true)" = "$host_arch" ] || return 1

    local ver deps
    ver="$(dpkg-deb -f "$deb" Version 2>/dev/null || true)"
    deps="$(dpkg-deb -f "$deb" Depends 2>/dev/null || true)"
    case "$ver" in
        *"$host_suffix") ;;
        *+[a-z]*) return 1 ;;
    esac
    case "$deps" in
        *libpython3.*)
            case "$deps" in *"libpython${pyver}"*) ;; *) return 1 ;; esac ;;
    esac
    return 0
}

# Locate the paired -dev .deb sharing the exact architecture and version.
find_paired_dev() {
    local dir="$1" host_arch="$2" target_ver="$3" dv
    for dv in "$dir"/*.deb; do
        [ -f "$dv" ] || continue
        [ "$(dpkg-deb -f "$dv" Package 2>/dev/null || true)" = "libtriton-jit-nvidia-dev" ] || continue
        [ "$(dpkg-deb -f "$dv" Architecture 2>/dev/null || true)" = "$host_arch" ] || continue
        [ "$(dpkg-deb -f "$dv" Version 2>/dev/null || true)" = "$target_ver" ] || continue
        printf '%s\n' "$dv"
        return 0
    done
    return 1
}

main() {
    # shellcheck source=/dev/null
    . /etc/os-release
    local host_arch host_suffix pyver rt_deb="" dev_deb="" best_ver="" d ver paired_dev
    host_arch="$(dpkg --print-architecture)"
    host_suffix="+${ID}${VERSION_ID}"
    pyver="$(/usr/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    for d in "${DEPS_DIR}"/*.deb; do
        deb_matches_host "$d" "$host_arch" "$host_suffix" "$pyver" || continue
        ver="$(dpkg-deb -f "$d" Version)"
        paired_dev="$(find_paired_dev "${DEPS_DIR}" "$host_arch" "$ver")" || continue
        if [ -z "$best_ver" ] || dpkg --compare-versions "$ver" gt "$best_ver"; then
            rt_deb="$d"
            dev_deb="$paired_dev"
            best_ver="$ver"
        fi
    done

    if [ -n "$rt_deb" ] && [ -n "$dev_deb" ]; then
        apt-get update && apt-get install -y "$rt_deb" "$dev_deb"
    else
        local apt_entry="${FLAGOS_APT:-https://resource.flagos.net/repository/flagos-apt-${ID}${VERSION_ID} ${VERSION_CODENAME} main}"
        install -d -m 0755 /etc/apt/keyrings
        curl -fsSL "${FLAGOS_APT_KEY}" -o /etc/apt/keyrings/flagos.asc
        echo "deb [signed-by=/etc/apt/keyrings/flagos.asc] ${apt_entry}" \
            > /etc/apt/sources.list.d/flagos.list
        apt-get update
        apt-get install -y libtriton-jit-nvidia-dev
    fi
    rm -rf /var/lib/apt/lists/*
}

main "$@"
