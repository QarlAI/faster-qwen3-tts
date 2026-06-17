#!/usr/bin/env python3
"""Compare latency benchmark artifacts across GPU instance types.

Reads the JSON files written by ``tests/test_tts_latency.py`` (one per run, tagged
with a GPU label) and prints a side-by-side table so you can decide which instance
type is best. If multiple artifacts share a GPU label, the most recent one wins.

Usage:
    python benchmarks/compare_latency.py [RESULTS_DIR] [--text medium]

    RESULTS_DIR   directory of latency_*.json files (default: benchmarks/latency_results)
    --text LABEL  which text length to compare (short|medium|long, default: medium)
"""
import glob
import json
import os
import sys


def load_latest_per_gpu(results_dir):
    """Return {label: artifact}, keyed by GPU+backend, keeping the newest timestamp.

    Keying on (gpu_label, backend) lets faster-qwen3 and vllm-omni runs on the
    same GPU coexist so they can be compared side-by-side.
    """
    latest = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "latency_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  (skipping unreadable {path}: {e})", file=sys.stderr)
            continue
        backend = data.get("backend", "faster-qwen3")
        label = f"{data.get('gpu_label', 'unknown')} [{backend}]"
        ts = data.get("timestamp", "")
        if label not in latest or ts >= latest[label].get("timestamp", ""):
            latest[label] = data
    return latest


def _g(d, *keys):
    """Safe nested get; returns None if any key is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    results_dir = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_results")
    text = "medium"
    if "--text" in sys.argv:
        text = sys.argv[sys.argv.index("--text") + 1]

    gpus = load_latest_per_gpu(results_dir)
    if not gpus:
        print(f"No latency_*.json artifacts found in {results_dir}")
        return 1

    print(f"\nComparison for text='{text}'  (lower latency / higher RTF is better)")
    print(f"Results dir: {results_dir}\n")

    header = (
        f"{'gpu [backend]':<24} {'ns req p50':>11} {'ns req p90':>11} {'ns rtf':>8} "
        f"{'str ttfa p50':>13} {'str ttfa p90':>13} {'str rtf':>8}  {'concurrency':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for label, data in gpus.items():
        ns = _g(data, "non_streaming", text) or {}
        stream = _g(data, "streaming", text, "concurrency") or {}
        conc_keys = sorted(stream.keys(), key=lambda c: int(c)) if stream else []
        base = stream.get(conc_keys[0]) if conc_keys else {}
        peak = stream.get(conc_keys[-1]) if conc_keys else {}

        # vLLM-Omni has no server-side RTF; fall back to client RTF so the column is populated.
        ns_rtf = _g(ns, "server_rtf", "mean")
        if ns_rtf is None:
            ns_rtf = _g(ns, "client_rtf", "mean")

        rows.append({
            "label": label,
            "ns_req_p50": _g(ns, "request_ms", "p50"),
            "ns_req_p90": _g(ns, "request_ms", "p90"),
            "ns_rtf": ns_rtf,
            "str_ttfa_p50": _g(base, "ttfa_ms", "p50"),
            # p90 at the highest concurrency level reveals how the GPU degrades under load
            "str_ttfa_p90_peak": _g(peak, "ttfa_ms", "p90"),
            "str_rtf": _g(base, "rtf", "mean"),
            "conc": ",".join(conc_keys) if conc_keys else "-",
        })

    # Sort by streaming TTFA p50 (single-stream responsiveness) when available.
    rows.sort(key=lambda r: (r["str_ttfa_p50"] is None, r["str_ttfa_p50"] or 0))

    def fmt(v, suffix="", width=0):
        s = "-" if v is None else f"{v:.0f}{suffix}" if suffix == "m" else f"{v:.2f}"
        return s.rjust(width) if width else s

    for r in rows:
        print(
            f"{r['label']:<24} {fmt(r['ns_req_p50'],'m',11)} {fmt(r['ns_req_p90'],'m',11)} "
            f"{fmt(r['ns_rtf'],'',8)} {fmt(r['str_ttfa_p50'],'m',13)} "
            f"{fmt(r['str_ttfa_p90_peak'],'m',13)} {fmt(r['str_rtf'],'',8)}  {r['conc']:>11}"
        )

    print("\nLegend: ns = non-streaming, str = streaming. Endpoints differ per backend")
    print("        (faster-qwen3: /tts + WS /tts/ws; vllm-omni: /v1/audio/speech).")
    print("        ns rtf falls back to client RTF when the server reports none (vllm-omni).")
    print("        'str ttfa p90' is measured at the HIGHEST concurrency level (load behaviour).")
    print("        RTF >= 1.0 means generation keeps up with real-time playback.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
