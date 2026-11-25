#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${IFDDA_MISHCHENKO_BUILD_DIR:-${source_dir}/build}"

cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$build_dir" --target ifdda_mishchenko_shared ifdda_mishchenko_sections

output_dir="$(cd -- "$build_dir/mishchenko" && pwd)"
cmake -E copy_if_different "$output_dir/libifdda_mishchenko.so" "$output_dir/ifdda_mishchenko_sections" "$build_dir"
build_dir="$(cd -- "$build_dir" && pwd)"
printf '%s\n' "$build_dir/libifdda_mishchenko.so" "$build_dir/ifdda_mishchenko_sections"
