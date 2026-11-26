#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${IFDDA_MISHCHENKO_BUILD_DIR:-${source_dir}/build}"

cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release "$@" -DIFDDA_MISHCHENKO_BUILD_STANDALONE=ON -DIFDDA_MISHCHENKO_ENABLE_CUDA=ON
cmake --build "$build_dir" --parallel 4
output_dir="$(cd -- "$build_dir/mishchenko" && pwd)"
for name in libifdda_mishchenko.so libifdda_mishchenko_cpu.so libifdda_mishchenko_openmp.so ifdda_mishchenko_sections_cpu ifdda_mishchenko_sections_openmp ifdda_mishchenko_sections_cuda; do
    cmake -E copy_if_different "$output_dir/$name" "$build_dir"
done
python3 "$source_dir/../test_backends.py" "$build_dir"
python3 "$source_dir/../test_backends.py" "$build_dir" --qp
