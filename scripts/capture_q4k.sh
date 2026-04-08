#!/usr/bin/env bash
# Capture a GPU trace of the Q4_K GEMV bench for Xcode's shader profiler.
#
# Opens the resulting .gputrace in Xcode — drill into a matvec_q4k_simd
# dispatch → "Counters" / "Performance" tab for per-source-line ALU and
# memory cost.
#
# Usage:
#   ./scripts/capture_q4k.sh              # defaults to --iters 3
#   ./scripts/capture_q4k.sh --iters 1    # keep iters low so the trace stays small
set -euo pipefail

cd "$(dirname "$0")/.."

TRACE=/tmp/llm-rs-q4k.gputrace
rm -rf "$TRACE"

cargo build --release --example bench_q4k_gemv >&2

# METAL_CAPTURE_ENABLED=1 lets the Metal framework capture GPU work from a
# non-Xcode binary. LLM_CAPTURE tells MetalBackend to start/stop the capture
# around everything it dispatches.
ARGS=("$@")
[[ ${#ARGS[@]} -eq 0 ]] && ARGS=(--iters 3)

METAL_CAPTURE_ENABLED=1 \
LLM_CAPTURE="$TRACE" \
  ./target/release/examples/bench_q4k_gemv "${ARGS[@]}"

echo "trace: $TRACE"
open "$TRACE"
