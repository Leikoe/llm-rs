#!/usr/bin/env python3
"""Parse Metal System Trace from xctrace and report per-token timing breakdown.

Usage:
    # Capture a trace:
    xcrun xctrace record --template 'Metal System Trace' --output trace.trace \
        --launch -- ./target/release/llm-rs -m model.gguf complete -p "Hello" -n 20 --temperature 0

    # Analyze it:
    python3 scripts/profile.py trace.trace
    python3 scripts/profile.py trace.trace --verbose
"""

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class Frame:
    number: int
    start_ns: int        # CPU timestamp when encoding began
    duration_ns: int     # total command buffer duration (encoding start to commit)
    encoder_ns: int      # time spent encoding compute commands (CPU-side)
    gpu_ns: int = 0      # actual GPU execution time


@dataclass
class GpuState:
    start_ns: int
    duration_ns: int
    active: bool


def export_table(trace_path: str, schema: str) -> str:
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    result = subprocess.run(
        ["xctrace", "export", "--input", trace_path, "--xpath", xpath],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"xctrace export failed for {schema}: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


# ── Regex-based XML parsing (xctrace XML uses id/ref dedup) ──

def _make_resolver():
    """Create a stateful resolver for xctrace's id/ref dedup scheme."""
    cache = {}

    def resolve(tag_xml: str) -> str:
        ref = re.search(r'ref="([^"]+)"', tag_xml)
        if ref:
            return cache.get(ref.group(1), "")
        m_id = re.search(r'id="([^"]+)"', tag_xml)
        m_fmt = re.search(r'fmt="([^"]*)"', tag_xml)
        m_text = re.search(r'>([^<]+)<', tag_xml)
        text = m_fmt.group(1) if m_fmt else (m_text.group(1) if m_text else "")
        if m_id:
            cache[m_id.group(1)] = text
        return text

    return resolve


def parse_frames(xml_str: str) -> list[Frame]:
    """Parse command buffer submissions to get per-frame timing."""
    resolve = _make_resolver()
    frames = []

    for row in re.finditer(r'<row>(.*?)</row>', xml_str, re.DOTALL):
        body = row.group(1)

        proc = re.search(r'<process\s[^>]*(?:>.*?</process>|/>)', body, re.DOTALL)
        if not proc:
            continue
        proc_text = resolve(proc.group(0))
        if "llm-rs" not in proc_text:
            continue

        start_m = re.search(r'<start-time[^>]*>(\d+)</start-time>', body)
        if not start_m:
            continue
        start_ns = int(start_m.group(1))

        durs = re.findall(r'<duration[^>]*>(\d+)</duration>', body)
        if not durs:
            continue
        duration_ns = int(durs[0])
        encoder_ns = int(durs[1]) if len(durs) > 1 else duration_ns

        frame_nums = re.findall(r'<uint32[^>]*>(\d+)</uint32>', body)
        frame_num = int(frame_nums[1]) if len(frame_nums) > 1 else len(frames) + 1

        frames.append(Frame(frame_num, start_ns, duration_ns, encoder_ns))

    return frames


def parse_gpu_intervals(xml_str: str, frames: list[Frame]) -> None:
    """Parse GPU intervals to get actual GPU execution time per frame.

    Updates frames in-place with gpu_ns.
    """
    # Collect our GPU intervals (process may appear anywhere in the row)
    gpu_intervals = []  # (start_ns, duration_ns)

    for row in re.finditer(r'<row>(.*?)</row>', xml_str, re.DOTALL):
        body = row.group(1)
        if 'llm-rs' not in body:
            continue

        start_m = re.search(r'<start-time[^>]*>(\d+)</start-time>', body)
        dur_m = re.search(r'<duration[^>]*>(\d+)</duration>', body)
        if not (start_m and dur_m):
            continue

        gpu_intervals.append((int(start_m.group(1)), int(dur_m.group(1))))

    # Sort by start time and match to frames (1:1 ordering)
    gpu_intervals.sort()
    for gi, frame in zip(gpu_intervals, frames):
        frame.gpu_ns = gi[1]


def parse_gpu_states(xml_str: str) -> list[GpuState]:
    """Parse GPU active/idle state intervals."""
    resolve = _make_resolver()
    states = []

    for row in re.finditer(r'<row>(.*?)</row>', xml_str, re.DOTALL):
        body = row.group(1)

        start_m = re.search(r'<start-time[^>]*>(\d+)</start-time>', body)
        dur_m = re.search(r'<duration[^>]*>(\d+)</duration>', body)

        state_tag = re.search(r'<gpu-state[^>]*/>', body) or re.search(r'<gpu-state[^>]*>[^<]*</gpu-state>', body)
        if not (start_m and dur_m and state_tag):
            continue

        state_text = resolve(state_tag.group(0))
        states.append(GpuState(
            int(start_m.group(1)),
            int(dur_m.group(1)),
            state_text == "Active",
        ))

    return states


def fmt_time(ns: int) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    return f"{ns / 1_000:.0f}us"


def print_report(frames: list[Frame], gpu_states: list[GpuState], verbose: bool):
    if not frames:
        print("No llm-rs frames found in trace.")
        return

    has_gpu = any(f.gpu_ns > 0 for f in frames)

    print(f"{'=' * 76}")
    print(f"  Metal Profile: {len(frames)} command buffers")
    print(f"{'=' * 76}")
    print()

    prefill = frames[0]
    gen_frames = frames[1:]

    # ── Per-frame table ──
    if verbose:
        if has_gpu:
            print(f"{'Frame':>5} | {'Encode':>8} | {'GPU Exec':>9} | {'Wall':>9} | {'GPU/Wall':>8}")
            print(f"{'-' * 5}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}")
        else:
            print(f"{'Frame':>5} | {'Encode':>8} | {'Wall':>9}")
            print(f"{'-' * 5}-+-{'-' * 8}-+-{'-' * 9}")

        for i, f in enumerate(frames):
            if i < len(frames) - 1:
                wall = frames[i + 1].start_ns - f.start_ns
                wall_str = fmt_time(wall)
            else:
                wall = f.gpu_ns if f.gpu_ns > 0 else f.duration_ns
                wall_str = "(last)"

            if has_gpu:
                gpu_pct = f.gpu_ns / wall * 100 if wall > 0 and f.gpu_ns > 0 and i < len(frames) - 1 else 0
                gpu_str = fmt_time(f.gpu_ns) if f.gpu_ns > 0 else "n/a"
                pct_str = f"{gpu_pct:>6.1f}%" if i < len(frames) - 1 else ""
                print(f"{f.number:>5} | {fmt_time(f.encoder_ns):>8} | {gpu_str:>9} | {wall_str:>9} | {pct_str}")
            else:
                print(f"{f.number:>5} | {fmt_time(f.encoder_ns):>8} | {wall_str:>9}")
        print()

    # ── Prefill summary ──
    print(f"  PREFILL (Frame 1)")
    print(f"    CPU encoding:   {fmt_time(prefill.encoder_ns)}")
    if prefill.gpu_ns > 0:
        print(f"    GPU execution:  {fmt_time(prefill.gpu_ns)}")
    print(f"    Total duration: {fmt_time(prefill.duration_ns)}")
    print()

    if not gen_frames:
        print("  Only 1 frame -- no generation data.")
        return

    # ── Generation summary ──
    enc_times = [f.encoder_ns for f in gen_frames]
    gpu_times = [f.gpu_ns for f in gen_frames if f.gpu_ns > 0]
    wall_gaps = []
    for i in range(len(gen_frames) - 1):
        wall_gaps.append(gen_frames[i + 1].start_ns - gen_frames[i].start_ns)

    avg_enc = sum(enc_times) / len(enc_times)
    avg_wall = sum(wall_gaps) / len(wall_gaps) if wall_gaps else avg_enc
    tok_s = 1e9 / avg_wall if avg_wall > 0 else 0

    print(f"  GENERATION ({len(gen_frames)} tokens)")
    print(f"    Throughput:          {tok_s:>9.1f} tok/s")
    print(f"    Avg wall time/tok:  {fmt_time(int(avg_wall)):>9}")
    print(f"    Avg CPU encoding:   {fmt_time(int(avg_enc)):>9}")

    if gpu_times:
        avg_gpu = sum(gpu_times) / len(gpu_times)
        overhead = avg_wall - avg_gpu
        print(f"    Avg GPU execution:  {fmt_time(int(avg_gpu)):>9}")
        print(f"    Avg overhead:       {fmt_time(int(overhead)):>9}  (wall - GPU)")
        print(f"    GPU efficiency:     {avg_gpu / avg_wall * 100:>8.1f}%  (GPU time / wall time)")
    print()

    # ── Time distribution ──
    enc_sorted = sorted(enc_times)
    print(f"    CPU encoding:  min={fmt_time(enc_sorted[0])}  p50={fmt_time(enc_sorted[len(enc_sorted)//2])}  max={fmt_time(enc_sorted[-1])}")
    if gpu_times:
        gpu_sorted = sorted(gpu_times)
        print(f"    GPU execution: min={fmt_time(gpu_sorted[0])}  p50={fmt_time(gpu_sorted[len(gpu_sorted)//2])}  max={fmt_time(gpu_sorted[-1])}")
    if wall_gaps:
        wall_sorted = sorted(wall_gaps)
        print(f"    Wall time:     min={fmt_time(wall_sorted[0])}  p50={fmt_time(wall_sorted[len(wall_sorted)//2])}  max={fmt_time(wall_sorted[-1])}")
    print()

    # ── GPU utilization (system-wide, from GPU state intervals) ──
    if gpu_states:
        gen_start = gen_frames[0].start_ns
        gen_end = gen_frames[-1].start_ns + gen_frames[-1].duration_ns

        active_ns = 0
        idle_ns = 0
        for s in gpu_states:
            s_end = s.start_ns + s.duration_ns
            overlap_start = max(s.start_ns, gen_start)
            overlap_end = min(s_end, gen_end)
            if overlap_start >= overlap_end:
                continue
            overlap = overlap_end - overlap_start
            if s.active:
                active_ns += overlap
            else:
                idle_ns += overlap

        total = active_ns + idle_ns
        if total > 0:
            print(f"  GPU UTILIZATION (system-wide, during generation)")
            print(f"    Active: {fmt_time(active_ns):>9}  ({active_ns / total * 100:.1f}%)")
            print(f"    Idle:   {fmt_time(idle_ns):>9}  ({idle_ns / total * 100:.1f}%)")
            print()

    # ── Bottleneck analysis ──
    print(f"  BOTTLENECK ANALYSIS")
    if gpu_times:
        avg_gpu = sum(gpu_times) / len(gpu_times)
        gpu_eff = avg_gpu / avg_wall * 100
        overhead_ms = (avg_wall - avg_gpu) / 1e6
        if gpu_eff > 80:
            print(f"    GPU-bound ({gpu_eff:.0f}% efficiency). Focus on kernel optimization.")
            print(f"    ~{fmt_time(int(avg_gpu))} of GPU compute per token is the bottleneck.")

            # Memory bandwidth analysis for 1B model
            # BF16 weights: each layer = wq+wk+wv+wo+w1+w2+w3 matvec reads
            # dim=2048: wq=2048*2048, wk/wv=2048*512, wo=2048*2048, w1/w3=2048*5632, w2=5632*2048
            # Per layer: 2*2048*2048 + 2*2048*512 + 3*2048*5632 = ~42M elements * 2 bytes = ~84MB
            # 16 layers = ~1.34GB. Plus embed + output. BW = data/time
            print(f"    Faster kernels and/or reducing memory reads are the path forward.")
        elif gpu_eff > 50:
            print(f"    Mixed ({gpu_eff:.0f}% efficiency). ~{overhead_ms:.1f}ms overhead per token.")
            print(f"    Both kernel speed and CPU-GPU pipeline matter.")
        else:
            print(f"    CPU-bound ({gpu_eff:.0f}% efficiency). ~{overhead_ms:.1f}ms overhead per token.")
            print(f"    The GPU is idle between submissions. Overlap encoding with execution.")
    else:
        enc_pct = avg_enc / avg_wall * 100
        print(f"    No GPU execution data (enable Shader Timeline in Instruments for per-kernel data).")
        print(f"    CPU encoding is {enc_pct:.0f}% of wall time -- remaining {100-enc_pct:.0f}% is GPU + overhead.")
    print()

    # ── Outliers ──
    if verbose and wall_gaps:
        mean_wall = sum(wall_gaps) / len(wall_gaps)
        outliers = [(i, g) for i, g in enumerate(wall_gaps) if g > mean_wall * 1.5]
        if outliers:
            print(f"  OUTLIER GAPS (>1.5x mean)")
            for i, gap in outliers:
                f = gen_frames[i]
                print(f"    Frame {f.number}->{gen_frames[i+1].number}: {fmt_time(gap)} (gpu={fmt_time(f.gpu_ns)})")
            print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Metal System Trace for llm-rs")
    parser.add_argument("trace", help="Path to .trace file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-frame table and outliers")
    args = parser.parse_args()

    schemas = [
        "metal-application-command-buffer-submissions",
        "metal-gpu-state-intervals",
        "metal-gpu-intervals",
    ]

    print("Exporting trace data...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {s: pool.submit(export_table, args.trace, s) for s in schemas}
        xml = {s: f.result() for s, f in futures.items()}

    print("Parsing...", file=sys.stderr)
    frames = parse_frames(xml[schemas[0]])
    parse_gpu_intervals(xml[schemas[2]], frames)
    gpu_states = parse_gpu_states(xml[schemas[1]])
    print_report(frames, gpu_states, args.verbose)


if __name__ == "__main__":
    main()
