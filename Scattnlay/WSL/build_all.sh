#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${IFDDA_SCATTNLAY_BUILD_DIR:-${source_dir}/../build/wsl}"

cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$build_dir" --target ifdda_scattnlay_shared ifdda_scattnlay_sections

output_dir="$(cd -- "$build_dir/scattnlay" && pwd)"
printf '%s\n' "$output_dir/libifdda_scattnlay.so" "$output_dir/ifdda_scattnlay_sections"
