"""Latency / throughput benchmark against a *deployed* Faster Qwen3 TTS server.

Unlike ``benchmarks/throughput.py`` (which loads the model in-process and needs a
local GPU), this suite is a **client-side benchmark**: it hits a running server
over HTTP and WebSocket via ``TTS_URL``. Deploy the repo on a GPU instance, point
``TTS_URL`` at it, tag the run with ``GPU_LABEL``, and run this file. Each run
writes a JSON artifact tagged with the GPU label so results from different
instance types can be compared side-by-side (see ``benchmarks/compare_latency.py``).

It is EXCLUDED from CI (requires a live server + a real GPU model) via
``pyproject.toml``. Run it explicitly:

    GPU_LABEL=A10G TTS_URL=http://my-a10g-host:8000 \
        .venv/bin/pytest tests/test_tts_latency.py -v -s

Useful env vars (all optional):
    TTS_URL              base URL of the deployed server      (default http://localhost:8000)
    GPU_LABEL            label for this run, e.g. A10G/L4/A100 (default $INSTANCE_TYPE or "unknown")
    LATENCY_VOICE        voice name on the server              (default english-male)
    LATENCY_LANGUAGE     language                              (default English)
    LATENCY_WARMUP       discarded warmup iterations           (default 2)
    LATENCY_RUNS         measured iterations                   (default 8)
    LATENCY_CONCURRENCY  comma list of concurrent stream loads (default "1,4")
    LATENCY_RESULTS_DIR  where to write JSON artifacts         (default benchmarks/latency_results)
    LATENCY_ASSERT       set 0/false to disable threshold asserts (default on)
    LATENCY_MAX_TTFA_MS  max acceptable streaming TTFA (mean, c=1)   (default 2000)
    LATENCY_MIN_RTF      min acceptable real-time factor (mean, c=1) (default 0.9)
    LATENCY_STREAM_TIMEOUT  per-stream hard timeout, seconds         (default 600)

Default concurrency is 1 to match production (one pod per job). See
docs/event-loop-offload-plan.md for serving concurrent streams per pod.

── Backends ────────────────────────────────────────────────────────────────
TTS_BACKEND selects the wire protocol so the SAME harness can benchmark either
server and write comparable artifacts:
    faster-qwen3 (default) — POST /tts (float32 PCM + X-* timing headers) and
                             WS /tts/ws (JSON {type:audio,data:[...]} frames).
    vllm-omni              — POST /v1/audio/speech (OpenAI-compatible). Non-stream
                             returns the full PCM body; streaming (stream=true,
                             response_format=pcm) returns chunked PCM bytes, so
                             TTFA = time to first byte and audio duration is derived
                             from the byte count. vLLM-Omni exposes no server-side
                             RTF/generation-time, so those fields are None and the
                             comparison uses client-measured RTF.

vLLM-Omni env vars (used only when TTS_BACKEND=vllm-omni):
    VLLM_MODEL              model id (default Qwen/Qwen3-TTS-12Hz-1.7B-Base)
    LATENCY_VOICE          voice name — an ENGLISH preset from GET /v1/audio/voices
                           or your own cloned voice; match it to your test language
    VLLM_RESPONSE_FORMAT   audio format (default "pcm" → raw int16 LE @ 24kHz mono)
    VLLM_PCM_SAMPLE_RATE   PCM sample rate for duration math (default 24000;
                           set 48000 for VoxCPM2. WS path also reads it from audio.start)
    VLLM_PCM_BYTES_PER_SAMPLE  bytes/sample (default 2 = int16)
    VLLM_REF_AUDIO / VLLM_REF_TEXT  optional inline voice-cloning params
    VLLM_STREAM_TRANSPORT  streaming test transport: "http" (default) or "ws"
                           (text streamed in incrementally via /v1/audio/speech/stream)
    VLLM_TASK_TYPE         WS session task_type: CustomVoice|VoiceDesign|Base (default: model's)
    VLLM_WS_STT_DELAY      WS path: seconds between words (default 0 = burst; >0 simulates STT)

    # example:
    GPU_LABEL=L4 TTS_BACKEND=vllm-omni TTS_URL=http://<host>:8091 \
        LATENCY_VOICE=<english-voice> LATENCY_CONCURRENCY=1,2,4 \
        .venv/bin/pytest tests/test_tts_latency.py -v -s
"""
import asyncio
import json
import os
import time

import httpx
import numpy as np
import pytest
import requests
import websockets

# ── Config ───────────────────────────────────────────────────────────────
TTS_URL = os.getenv("TTS_URL", "http://localhost:8000").rstrip("/")
WS_URL = TTS_URL.replace("http://", "ws://").replace("https://", "wss://")
GPU_LABEL = os.getenv("GPU_LABEL") or os.getenv("INSTANCE_TYPE") or "unknown"
VOICE = os.getenv("LATENCY_VOICE", "english-male")
LANGUAGE = os.getenv("LATENCY_LANGUAGE", "English")
WARMUP = int(os.getenv("LATENCY_WARMUP", "2"))
RUNS = int(os.getenv("LATENCY_RUNS", "8"))
CONCURRENCY = [int(c) for c in os.getenv("LATENCY_CONCURRENCY", "1").split(",") if c.strip()]
RESULTS_DIR = os.getenv(
    "LATENCY_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "latency_results"),
)
ASSERT = os.getenv("LATENCY_ASSERT", "1").lower() not in ("0", "false", "no")
MAX_TTFA_MS = float(os.getenv("LATENCY_MAX_TTFA_MS", "2000"))
MIN_RTF = float(os.getenv("LATENCY_MIN_RTF", "0.9"))
STREAM_TIMEOUT = float(os.getenv("LATENCY_STREAM_TIMEOUT", "600"))

# Backend selection: "faster-qwen3" (default) or "vllm-omni"
BACKEND = os.getenv("TTS_BACKEND", "faster-qwen3").lower()
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VLLM_RESPONSE_FORMAT = os.getenv("VLLM_RESPONSE_FORMAT", "pcm")
PCM_SAMPLE_RATE = int(os.getenv("VLLM_PCM_SAMPLE_RATE", "24000"))
PCM_BYTES_PER_SAMPLE = int(os.getenv("VLLM_PCM_BYTES_PER_SAMPLE", "2"))  # 2 = int16 LE
VLLM_REF_AUDIO = os.getenv("VLLM_REF_AUDIO") or None
VLLM_REF_TEXT = os.getenv("VLLM_REF_TEXT") or None

# vLLM-Omni streaming transport: "http" (POST /v1/audio/speech, stream=true) or
# "ws" (WebSocket /v1/audio/speech/stream — text streamed IN incrementally,
# per-sentence PCM streamed OUT). Only affects the streaming test.
VLLM_STREAM_TRANSPORT = os.getenv("VLLM_STREAM_TRANSPORT", "http").lower()
VLLM_TASK_TYPE = os.getenv("VLLM_TASK_TYPE") or None  # CustomVoice|VoiceDesign|Base; omitted -> model default
# Seconds between words when feeding text on the WS path. 0 = burst all words
# immediately (measures TTS compute); >0 simulates an upstream STT/LLM token rate
# (TTFA then includes first-sentence accumulation).
VLLM_WS_STT_DELAY = float(os.getenv("VLLM_WS_STT_DELAY", "0"))

# Representative texts spanning a range of lengths — latency and RTF both scale
# with how much audio gets generated, so we want short/medium/long coverage.
TEXTS = [
    ("short", "Hello, how can I help you today?"),
    (
        "medium",
        "Thanks for reaching out. I have checked your account and everything looks good, "
        "so you should be all set to continue using the service without any issues.",
    ),
    (
        "long",
        "Artificial intelligence is rapidly transforming the way we live and work, reshaping "
        "industries from healthcare to transportation. As these systems become more capable, "
        "it is increasingly important to deploy them responsibly, with careful attention to "
        "performance, cost, and reliability.",
    ),
]


# ── Stats helper ───────────────────────────────────────────────────────────
def _stats(samples):
    """Summarize a list of numbers (Nones dropped). Returns None if empty."""
    arr = np.asarray([s for s in samples if s is not None], dtype=float)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
        "min": round(float(arr.min()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "max": round(float(arr.max()), 2),
    }


# ── Backend helpers ──────────────────────────────────────────────────────
def _pcm_seconds(n_bytes):
    """Seconds of audio in n_bytes of raw PCM (int16 mono @ PCM_SAMPLE_RATE by default)."""
    denom = PCM_BYTES_PER_SAMPLE * PCM_SAMPLE_RATE
    return n_bytes / denom if denom else 0.0


def _vllm_body(text, stream):
    """OpenAI-compatible /v1/audio/speech request body for vLLM-Omni."""
    body = {
        "model": VLLM_MODEL,
        "input": text,
        "voice": VOICE,
        "response_format": VLLM_RESPONSE_FORMAT,
        "stream": stream,
    }
    if LANGUAGE:
        body["language"] = LANGUAGE
    if VLLM_REF_AUDIO:
        body["ref_audio"] = VLLM_REF_AUDIO
    if VLLM_REF_TEXT:
        body["ref_text"] = VLLM_REF_TEXT
    return body


# ── Measurement primitives ─────────────────────────────────────────────────
def _measure_non_streaming(text):
    """One blocking non-streaming request. Dispatches on TTS_BACKEND."""
    if BACKEND == "vllm-omni":
        return _measure_non_streaming_vllm(text)
    return _measure_non_streaming_fq(text)


def _measure_non_streaming_fq(text):
    """faster-qwen3: POST /tts -> float32 PCM body + X-* timing headers."""
    t0 = time.perf_counter()
    r = requests.post(
        f"{TTS_URL}/tts",
        json={"text": text, "language": LANGUAGE, "voice": VOICE},
        timeout=300,
    )
    request_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"/tts failed ({r.status_code}): {r.text[:200]}"
    audio_dur = float(r.headers.get("X-Audio-Duration", 0.0))
    return {
        "request_ms": request_ms,
        "server_generation_ms": float(r.headers.get("X-Generation-Time", 0.0)) * 1000,
        "audio_duration_s": audio_dur,
        "server_rtf": float(r.headers.get("X-RTF", 0.0)),
        "client_rtf": (audio_dur / (request_ms / 1000)) if request_ms > 0 else 0.0,
    }


def _measure_non_streaming_vllm(text):
    """vLLM-Omni: POST /v1/audio/speech (stream=false) -> full PCM body.

    No server-side timing headers, so server_* are None; duration is derived
    from the PCM byte count.
    """
    t0 = time.perf_counter()
    r = requests.post(
        f"{TTS_URL}/v1/audio/speech",
        json=_vllm_body(text, stream=False),
        timeout=300,
    )
    request_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"/v1/audio/speech failed ({r.status_code}): {r.text[:200]}"
    audio_dur = _pcm_seconds(len(r.content))
    return {
        "request_ms": request_ms,
        "server_generation_ms": None,
        "audio_duration_s": audio_dur,
        "server_rtf": None,
        "client_rtf": (audio_dur / (request_ms / 1000)) if request_ms > 0 else 0.0,
    }


async def _measure_streaming_once(text):
    """One streaming request. Dispatches on TTS_BACKEND (and transport for vllm-omni)."""
    if BACKEND == "vllm-omni":
        if VLLM_STREAM_TRANSPORT == "ws":
            return await _measure_streaming_once_vllm_ws(text)
        return await _measure_streaming_once_vllm(text)
    return await _measure_streaming_once_fq(text)


async def _measure_streaming_once_vllm(text):
    """vLLM-Omni HTTP streaming: POST /v1/audio/speech (stream=true,
    response_format=pcm). TTFA = time to first PCM byte; duration from the total
    byte count. No server-side generation time (None). NB: TTFA includes the
    HTTP connect (negligible on localhost, one RTT over a network)."""

    async def _stream():
        async with httpx.AsyncClient(timeout=httpx.Timeout(STREAM_TIMEOUT)) as client:
            t0 = time.perf_counter()
            async with client.stream(
                "POST", f"{TTS_URL}/v1/audio/speech", json=_vllm_body(text, stream=True)
            ) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    raise RuntimeError(f"/v1/audio/speech failed ({r.status_code}): {body[:200]!r}")
                ttfa_ms = None
                total_bytes = 0
                chunk_count = 0
                async for chunk in r.aiter_bytes():
                    if not chunk:
                        continue
                    if ttfa_ms is None:
                        ttfa_ms = (time.perf_counter() - t0) * 1000
                    total_bytes += len(chunk)
                    chunk_count += 1
                total_ms = (time.perf_counter() - t0) * 1000
        audio_dur = _pcm_seconds(total_bytes)
        return {
            "ttfa_ms": ttfa_ms,
            "total_ms": total_ms,
            "server_generation_ms": None,
            "audio_duration_s": audio_dur,
            "rtf": (audio_dur / (total_ms / 1000)) if total_ms > 0 else 0.0,
            "chunk_count": chunk_count,
        }

    return await asyncio.wait_for(_stream(), timeout=STREAM_TIMEOUT)


def _vllm_ws_config():
    """session.config for /v1/audio/speech/stream. Forces pcm + stream_audio so
    we get raw bytes per chunk (true TTFA) and can derive duration from byte count."""
    cfg = {
        "voice": VOICE,
        "response_format": "pcm",
        "stream_audio": True,
    }
    if VLLM_MODEL:
        cfg["model"] = VLLM_MODEL
    if LANGUAGE:
        cfg["language"] = LANGUAGE
    if VLLM_TASK_TYPE:
        cfg["task_type"] = VLLM_TASK_TYPE
    if VLLM_REF_AUDIO:
        cfg["ref_audio"] = VLLM_REF_AUDIO
    if VLLM_REF_TEXT:
        cfg["ref_text"] = VLLM_REF_TEXT
    return cfg


async def _measure_streaming_once_vllm_ws(text):
    """vLLM-Omni streaming-text WebSocket: /v1/audio/speech/stream.

    Streams text IN word-by-word (the real LLM->TTS path) and per-sentence PCM
    OUT. TTFA = time from first text sent to first audio byte; duration from the
    PCM byte count (sample rate taken from the audio.start event). Sentence
    segmentation + pipelining happen server-side.
    """
    uri = f"{WS_URL}/v1/audio/speech/stream"

    async def _run():
        # Manual connect (not `async with`): the server drops the TCP socket
        # without a WS close handshake right after session.done, so the implicit
        # close raises ConnectionClosedError. We swallow that close-time error
        # once we've received session.done; a drop *before* session.done still
        # propagates as a real failure.
        ws = await websockets.connect(uri, max_size=None, ping_interval=None)
        await ws.send(json.dumps({"type": "session.config", **_vllm_ws_config()}))

        async def _send():
            words = text.split(" ")
            for i, w in enumerate(words):
                chunk = w + (" " if i < len(words) - 1 else "")
                await ws.send(json.dumps({"type": "input.text", "text": chunk}))
                if VLLM_WS_STT_DELAY > 0:
                    await asyncio.sleep(VLLM_WS_STT_DELAY)
            await ws.send(json.dumps({"type": "input.done"}))

        t0 = time.perf_counter()
        sender = asyncio.create_task(_send())
        ttfa_ms = None
        total_bytes = 0
        chunk_count = 0
        sentence_count = 0
        sample_rate = PCM_SAMPLE_RATE
        got_done = False
        try:
            while True:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    if msg:
                        if ttfa_ms is None:
                            ttfa_ms = (time.perf_counter() - t0) * 1000
                        total_bytes += len(msg)
                        chunk_count += 1
                    continue
                ev = json.loads(msg)
                et = ev.get("type")
                if et == "audio.start":
                    sentence_count += 1
                    sr = ev.get("sample_rate")
                    if sr:
                        sample_rate = int(sr)
                elif et == "session.done":
                    got_done = True
                    break
                elif et == "error":
                    raise RuntimeError(f"speech/stream error: {ev.get('message')}")
        except websockets.exceptions.ConnectionClosed:
            if not got_done:
                raise  # genuine mid-stream drop — real failure
        finally:
            sender.cancel()
            try:
                await sender
            except Exception:  # noqa: BLE001 - sender may have hit the closed socket
                pass
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - benign ungraceful close after session.done
                pass
        total_ms = (time.perf_counter() - t0) * 1000
        denom = PCM_BYTES_PER_SAMPLE * sample_rate
        audio_dur = total_bytes / denom if denom else 0.0
        return {
            "ttfa_ms": ttfa_ms,
            "total_ms": total_ms,
            "server_generation_ms": None,
            "audio_duration_s": audio_dur,
            "rtf": (audio_dur / (total_ms / 1000)) if total_ms > 0 else 0.0,
            "chunk_count": chunk_count,
            "sentence_count": sentence_count,
        }

    return await asyncio.wait_for(_run(), timeout=STREAM_TIMEOUT)


async def _measure_streaming_once_fq(text):
    """One WS /tts/ws stream. Returns client-observed TTFA / total / RTF.

    ping_interval=None mirrors the pipecat client: the server blocks the event
    loop while generating, so keepalive pings would time out under load.
    """
    uri = f"{WS_URL}/tts/ws"

    async def _stream():
        async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
            t0 = time.perf_counter()
            await ws.send(json.dumps({"text": text, "language": LANGUAGE, "voice": VOICE}))
            ttfa_ms = None
            total_samples = 0
            sr = 24000
            chunk_count = 0
            server_total_s = None
            while True:
                data = json.loads(await ws.recv())
                kind = data["type"]
                if kind == "audio":
                    if ttfa_ms is None:
                        ttfa_ms = (time.perf_counter() - t0) * 1000
                    total_samples += len(data["data"])
                    sr = data.get("sample_rate", sr)
                    chunk_count += 1
                elif kind == "end":
                    total_ms = (time.perf_counter() - t0) * 1000
                    server_total_s = data.get("total_time_seconds")
                    break
                elif kind == "error":
                    raise RuntimeError(f"TTS error: {data['message']}")
        audio_dur = total_samples / sr if sr else 0.0
        return {
            "ttfa_ms": ttfa_ms,
            "total_ms": total_ms,
            "server_generation_ms": (server_total_s * 1000) if server_total_s is not None else None,
            "audio_duration_s": audio_dur,
            "rtf": (audio_dur / (total_ms / 1000)) if total_ms > 0 else 0.0,
            "chunk_count": chunk_count,
        }

    return await asyncio.wait_for(_stream(), timeout=STREAM_TIMEOUT)


async def _run_streaming_concurrent(text, concurrency):
    """Fire `concurrency` streams at once. Failures come back as exceptions
    (return_exceptions) so one bad request doesn't drop the whole batch."""
    return await asyncio.gather(
        *[_measure_streaming_once(text) for _ in range(concurrency)],
        return_exceptions=True,
    )


def _get_health():
    health = {}
    try:
        r = requests.get(f"{TTS_URL}/health", timeout=30)
        try:
            health = r.json()
        except Exception:  # noqa: BLE001 - vLLM /health returns an empty 200 body
            health = {"status_code": r.status_code}
    except Exception as e:  # noqa: BLE001 - best-effort metadata
        health = {"error": str(e)}
    if BACKEND == "vllm-omni":
        try:
            v = requests.get(f"{TTS_URL}/v1/audio/voices", timeout=30)
            if v.status_code == 200:
                health["voices"] = v.json()
        except Exception:  # noqa: BLE001
            pass
    return health


# ── Collector fixture: writes artifact + prints summary at module teardown ──
@pytest.fixture(scope="module")
def collector():
    data = {
        "gpu_label": GPU_LABEL,
        "backend": BACKEND,
        "model": VLLM_MODEL if BACKEND == "vllm-omni" else None,
        "stream_transport": VLLM_STREAM_TRANSPORT if BACKEND == "vllm-omni" else "ws",
        "ws_stt_delay": VLLM_WS_STT_DELAY if (BACKEND == "vllm-omni" and VLLM_STREAM_TRANSPORT == "ws") else None,
        "tts_url": TTS_URL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "voice": VOICE,
        "language": LANGUAGE,
        "warmup": WARMUP,
        "runs": RUNS,
        "concurrency_levels": CONCURRENCY,
        "health": _get_health(),
        "non_streaming": {},
        "streaming": {},
    }
    yield data
    _write_and_summarize(data)


def _write_and_summarize(data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in GPU_LABEL)
    stamp = data["timestamp"].replace(":", "").replace("-", "")
    path = os.path.join(RESULTS_DIR, f"latency_{safe_label}_{BACKEND}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    ns_ep = "/v1/audio/speech" if BACKEND == "vllm-omni" else "/tts"
    if BACKEND == "vllm-omni":
        st_ep = "/v1/audio/speech/stream (ws)" if VLLM_STREAM_TRANSPORT == "ws" else "/v1/audio/speech (stream)"
    else:
        st_ep = "/tts/ws"
    line = "=" * 78
    print(f"\n{line}\nLATENCY BENCHMARK  backend={BACKEND}  gpu={GPU_LABEL}  url={TTS_URL}\n{line}")

    if data["non_streaming"]:
        print(f"\nNon-streaming  POST {ns_ep}")
        print(f"  {'text':<8} {'req p50':>9} {'req p90':>9} {'gen p50':>9} "
              f"{'audio s':>9} {'rtf':>8}")
        for label, m in data["non_streaming"].items():
            req, gen = m["request_ms"], m["server_generation_ms"]
            rtf_stat = m["server_rtf"] or m["client_rtf"]  # vLLM has no server RTF -> client
            print(f"  {label:<8} {req['p50']:>8.0f}m {req['p90']:>8.0f}m "
                  f"{(gen['p50'] if gen else 0):>8.0f}m {m['audio_duration_s']:>9.2f} "
                  f"{(rtf_stat['mean'] if rtf_stat else 0):>8.2f}")

    if data["streaming"]:
        print(f"\nStreaming  {st_ep}")
        print(f"  {'text':<8} {'conc':>4} {'ttfa p50':>9} {'ttfa p90':>9} "
              f"{'total p50':>10} {'rtf mean':>9}")
        for label, m in data["streaming"].items():
            for conc, s in m["concurrency"].items():
                ttfa, tot, rtf = s["ttfa_ms"], s["total_ms"], s["rtf"]
                fail = f"  ({s.get('failures', 0)} failed)" if s.get("failures") else ""
                if ttfa is None:
                    print(f"  {label:<8} {conc:>4}  all requests failed{fail}")
                    continue
                print(f"  {label:<8} {conc:>4} {ttfa['p50']:>8.0f}m {ttfa['p90']:>8.0f}m "
                      f"{tot['p50']:>9.0f}m {(rtf['mean'] if rtf else 0):>9.2f}{fail}")

    print(f"\nArtifact: {path}\n{line}")


# ── Tests ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("label,text", TEXTS)
def test_non_streaming_latency(label, text, collector):
    for _ in range(WARMUP):
        _measure_non_streaming(text)

    runs = [_measure_non_streaming(text) for _ in range(RUNS)]
    collector["non_streaming"][label] = {
        "text_words": len(text.split()),
        "audio_duration_s": round(float(np.mean([r["audio_duration_s"] for r in runs])), 3),
        "request_ms": _stats([r["request_ms"] for r in runs]),
        "server_generation_ms": _stats([r["server_generation_ms"] for r in runs]),
        "server_rtf": _stats([r["server_rtf"] for r in runs]),
        "client_rtf": _stats([r["client_rtf"] for r in runs]),
    }

    if ASSERT:
        ns = collector["non_streaming"][label]
        rtf_stat = ns["server_rtf"] or ns["client_rtf"]  # vLLM-Omni: no server RTF -> client
        rtf = rtf_stat["mean"] if rtf_stat else 0.0
        assert rtf >= MIN_RTF, f"[{label}] non-streaming RTF {rtf} < {MIN_RTF}"


@pytest.mark.parametrize("label,text", TEXTS)
def test_streaming_latency(label, text, collector):
    per_conc = {}
    for conc in CONCURRENCY:
        for _ in range(WARMUP):
            asyncio.run(_run_streaming_concurrent(text, conc))

        ttfa, total, rtf, gen = [], [], [], []
        failures = 0
        for _ in range(RUNS):
            for res in asyncio.run(_run_streaming_concurrent(text, conc)):
                if isinstance(res, Exception):
                    failures += 1
                    continue
                ttfa.append(res["ttfa_ms"])
                total.append(res["total_ms"])
                rtf.append(res["rtf"])
                gen.append(res["server_generation_ms"])

        per_conc[str(conc)] = {
            "requests": len(ttfa),
            "failures": failures,
            "ttfa_ms": _stats(ttfa),
            "total_ms": _stats(total),
            "server_generation_ms": _stats(gen),
            "rtf": _stats(rtf),
        }

    collector["streaming"][label] = {"text_words": len(text.split()), "concurrency": per_conc}

    if ASSERT:
        base = per_conc[str(CONCURRENCY[0])]
        assert base["ttfa_ms"] is not None, (
            f"[{label}] all streaming requests failed at c={CONCURRENCY[0]} ({base['failures']} failures)"
        )
        assert base["ttfa_ms"]["mean"] <= MAX_TTFA_MS, (
            f"[{label}] streaming TTFA {base['ttfa_ms']['mean']}ms > {MAX_TTFA_MS}ms (c={CONCURRENCY[0]})"
        )
        assert base["rtf"]["mean"] >= MIN_RTF, (
            f"[{label}] streaming RTF {base['rtf']['mean']} < {MIN_RTF} (c={CONCURRENCY[0]})"
        )
