#!/usr/bin/env python3
"""DeepInfer offline EP=DP=8 benchmark; run from a configured Ascend container.

200 global requests / 8 ranks = 25 per rank. One simultaneous burst (closed
batch), not an HTTP serving benchmark. Request timestamps are captured at the
engine API; output tokens, including accepted draft tokens, are counted once.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(os.environ.get("DEEPINFER_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / "examples"), str(ROOT)]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/home/weights/DeepSeek-V4.1-Flash-1017")
    p.add_argument("--input-len", type=int, default=4096)
    p.add_argument("--output-len", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=200, help="GLOBAL batch concurrency")
    p.add_argument("--dp", type=int, default=8)
    p.add_argument("--max-batched-tokens", type=int, default=4096, help="per DP rank")
    p.add_argument("--max-model-len", type=int, default=32768, help="engine capacity; independent of actual input/output lengths")
    p.add_argument("--memory-utilization", type=float, default=0.95)
    p.add_argument("--warmup-output-len", type=int, default=None, help="defaults to output-len, including batch-drain shapes")
    p.add_argument("--profile-steps", type=int, default=10, help="separate post-benchmark run; 0 disables")
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--dataset", default=None, help="GSM8K JSONL: repeat question tokens to input-len")
    a = p.parse_args()
    if a.warmup_output_len is None:
        a.warmup_output_len = a.output_len
    if a.concurrency <= 0 or a.concurrency % a.dp:
        p.error("concurrency must be a positive multiple of dp")
    if min(a.input_len, a.output_len, a.warmup_output_len) <= 0:
        p.error("token lengths must be positive")
    if a.max_model_len < a.input_len + max(a.output_len, a.warmup_output_len, 256) + 6:
        p.error("max-model-len must accommodate input, output, and speculative tokens")
    return a


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def load_questions(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [row["question"] for row in rows]
    if not all(isinstance(q, str) and q.strip() for q in questions):
        raise ValueError("Dataset contains empty/non-string questions")
    return questions


def worker(dp_size, rank):
    args = arguments()
    save(args.result_dir / f"worker-rank{rank}.json", dict(rank=rank, pid=os.getpid()))
    from _config import create_llm, setup_common_env
    os.environ["DEEPINFER_FORCE_LOAD_BALANCE"] = "0"
    os.environ["DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS"] = "0"
    setup_common_env(model_name=args.model, sampler_force_topk=4096,
                     set_override_experts=False, enable_eplb=False)
    local_batch = args.concurrency // dp_size
    llm = create_llm(args.model, rank, ep_size=dp_size, cp_size=1,
        env={"DEEPINFER_LOGGING_INTERVAL_STEP": "100"},
        data_parallel_size=dp_size, max_num_seqs=local_batch,
        max_num_batched_tokens=args.max_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.memory_utilization, seed=args.seed,
        lookahead_tokens=min(1, int(os.getenv("DEEPINFER_USE_NEXTN_DECODE", "5"))))
    # Import torch only AFTER create_llm assigns this rank's visible device.
    import numpy as np
    import torch
    import torch_npu
    from deepinfer import envs
    from deepinfer._C.engine import RequestOutput
    from deepinfer.model_executor.parallel_utils.parallel_state import PG
    from deepinfer.sampling_params import SamplingParams
    from deepinfer.timer import get_timers, cuda_timer
    group = PG.cpu_data_parallel_group()
    engine = llm.llm_engine
    outdir = args.result_dir

    def barrier():
        torch.distributed.barrier(group=group)

    corpus = (
        "A research team is designing a distributed computing system. "
        "The system processes documents, answers questions, and explains its reasoning. "
        "Engineers measure latency, memory capacity, communication costs, and throughput. "
        "They compare several algorithms using controlled experiments and repeatable inputs. "
        "The report describes numerical methods, matrix operations, networking, and scheduling. "
        "A separate chapter tells a story about a traveler exploring mountains and cities. "
        "Please continue this technical report with detailed explanations and examples. "
    )
    base = np.asarray(llm.get_tokenizer().encode(corpus, add_special_tokens=False), dtype=np.int32)
    prompt_hashes = []
    questions = load_questions(args.dataset) if args.dataset else None
    if questions is not None:
        selection = np.random.default_rng(args.seed).permutation(len(questions))

    def prompts(phase):
        if questions is not None:
            batch, manifest = [], []
            offset = phase * args.concurrency + rank * local_batch
            for row_id in selection[offset:offset + local_batch]:
                question = questions[int(row_id)]
                tokens = np.asarray(llm.get_tokenizer().encode(question + "\n\n", add_special_tokens=False), dtype=np.int32)
                assert len(tokens) > 0
                ids = np.resize(tokens, args.input_len)
                batch.append(ids)
                manifest.append(dict(row_id=int(row_id), question=question,
                    original_tokens=len(tokens), input_tokens=len(ids),
                    prompt_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
            assert len(batch) == local_batch
            save(outdir / f"prompts-phase{phase}-rank{rank}.json", manifest)
            return batch
        rng = np.random.default_rng(args.seed + rank * 10000 + phase * 1000000)
        batch = []
        for _ in range(local_batch):
            # Different first 64 tokens prevent prompt/prefix reuse across runs.
            prefix = rng.choice(base, size=min(64, args.input_len), replace=True)
            rest = np.resize(np.roll(base, int(rng.integers(len(base)))), args.input_len - len(prefix))
            ids = np.concatenate([prefix, rest]).astype(np.int32)
            assert len(ids) == args.input_len
            batch.append(ids)
        return batch

    def run_phase(name, phase, output_len, profile=False):
        batch = prompts(phase)
        params = SamplingParams(max_tokens=output_len, ignore_eos=True, temperature=0.0, top_p=1.0)
        torch.npu.synchronize()
        barrier()
        torch.npu.reset_peak_memory_stats()
        for _, timer in get_timers():
            timer.clear()
        records = {}
        arrivals = []
        start = time.time()
        save(outdir / f"{name}-start-rank{rank}.json", {"time": start})
        for ids in batch:
            submitted = time.time()
            seq = llm._add_request(None, params, ids)
            rid = str(seq.request_id)
            records[rid] = dict(rank=rank, request_id=rid, submitted=submitted,
                first_token=None, completed=None, input_tokens=len(ids), output_tokens=0,
                draft_tokens=0, accepted_draft_tokens=0, finish_reason=None)
            if name == "measure":
                prompt_hashes.append(hashlib.sha256(ids.tobytes()).hexdigest())
        profiler = None
        profile_started = False
        decode_steps = 0
        captured = 0
        step = 0
        last_report = time.monotonic()
        while True:
            unfinished = engine.has_unfinished_requests()
            if profile and unfinished and not profile_started:
                waiting = torch.tensor([len(engine.scheduler.waiting)], dtype=torch.int64)
                torch.distributed.all_reduce(waiting, group=group)
                if waiting.item() == 0:
                    decode_steps += 1
                if decode_steps == 8:
                    profile_started = True
                    if rank == 0:
                        profiler = torch_npu.profiler.profile(
                            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                        torch_npu.profiler.ProfilerActivity.NPU],
                            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=args.profile_steps, repeat=1),
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(outdir / "profile")),
                            record_shapes=False, profile_memory=False, with_stack=False)
                        profiler.start()
                        print("BENCH profiler started (separate diagnostic phase)", flush=True)
            outputs = engine.step() if unfinished else engine.flush_inflight_reqs()
            now = time.time()
            n_out = 0
            for raw_output in outputs:
                output = RequestOutput.clone(raw_output)
                r = records[str(output.request_id)]
                assert output.prompt_len == args.input_len
                if r["first_token"] is None:
                    r["first_token"] = now
                r["output_tokens"] += 1
                n_out += 1
                if output.draft_tok > 0:
                    r["draft_tokens"] += int(output.draft_tok) - 1
                    r["accepted_draft_tokens"] += int(output.accept_tok) - 1
                if output.finished:
                    r["completed"] = now
                    r["finish_reason"] = str(output.finish_reason)
            if name == "measure":
                arrivals.append([step, now, n_out, len(engine.scheduler.running), len(engine.scheduler.waiting)])
            if profiler is not None:
                profiler.step()
                captured += 1
                if captured >= args.profile_steps:
                    profiler.stop()
                    profiler = None
                    print(f"BENCH profiler finished: {captured} steps", flush=True)
            step += 1
            if rank == 0 and time.monotonic() - last_report >= 20:
                count = sum(r["output_tokens"] for r in records.values())
                complete = sum(r["completed"] is not None for r in records.values())
                print(f"BENCH {name}: step={step} rank0_tokens={count} done={complete}/{local_batch} elapsed={now-start:.1f}s", flush=True)
                last_report = time.monotonic()
            if not unfinished:
                break
        if profiler is not None:
            profiler.stop()
        torch.npu.synchronize()
        end = time.time()
        cuda_timer.flush_out()
        timers = {name: {key: {"seconds": value, "count": timer.called[key],
                              "max_seconds": timer.max_time[key]}
                          for key, value in timer.times.items()}
                  for name, timer in get_timers()}
        result = dict(rank=rank, phase=name, start=start, end=end, steps=step,
            peak_allocated_bytes=torch.npu.max_memory_allocated(),
            peak_reserved_bytes=torch.npu.max_memory_reserved(),
            comm_sms=envs.DEEPINFER_USERBUFFER_COMM_SMS,
            nextn_decode=envs.DEEPINFER_USE_NEXTN_DECODE,
            mtp_varlen=envs.DEEPINFER_ENABLE_MTP_VARLEN,
            request_records=list(records.values()), timers=timers,
            profile_steps=captured, prompt_sha256=prompt_hashes if name == "measure" else [])
        save(outdir / f"{name}-rank{rank}.json", result)
        if name == "measure":
            with (outdir / f"arrivals-rank{rank}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "timestamp", "output_tokens", "running", "waiting"])
                writer.writerows(arrivals)
        for r in records.values():
            assert r["completed"] is not None and r["output_tokens"] == output_len, r
        barrier()
        print(f"BENCH {name} rank={rank} complete elapsed={end-start:.3f}s", flush=True)

    try:
        run_phase("warmup", 0, args.warmup_output_len)
        run_phase("measure", 1, args.output_len)
        if args.profile_steps:
            run_phase("diagnostic", 2, 256, profile=True)
    finally:
        llm.close()


def memory_monitor(outdir, stop):
    with (outdir / "hbm_samples.csv").open("w", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "device", "used_mib", "total_mib", "util_percent"])
        while not stop.is_set():
            begin = time.monotonic()
            try:
                raw = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
                with (outdir / "npu_smi_snapshots.jsonl").open("a") as snapshots:
                    snapshots.write(json.dumps(dict(timestamp=time.time(), output=raw)) + "\n")
                current = None
                for line in raw.splitlines():
                    match = re.search(r"\|\s*(\d+)\s*\|\s*Ascend", line)
                    if match:
                        current = int(match.group(1))
                    if current is not None and " / " in line:
                        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
                        if pairs and int(pairs[-1][1]) > 1000:
                            columns = line.split("|")
                            util = re.search(r"\b(\d+)\b", columns[-2])
                            writer.writerow([time.time(), current, *map(int, pairs[-1]), int(util[1]) if util else -1])
                            current = None
            except Exception as exc:
                with (outdir / "monitor_errors.log").open("a") as f:
                    f.write(f"{time.time()}: {exc}\n")
            stop.wait(max(0.05, 1.0 - (time.monotonic() - begin)))


def stats(values):
    values = sorted(values)
    def percentile(q):
        i = (len(values) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return values[lo] + (values[hi] - values[lo]) * (i - lo)
    return dict(mean=sum(values)/len(values), p50=percentile(.5), p95=percentile(.95),
                p99=percentile(.99), min=values[0], max=values[-1])


def aggregate(args):
    data = [json.loads((args.result_dir / f"measure-rank{r}.json").read_text()) for r in range(args.dp)]
    requests = [r for rank in data for r in rank["request_records"]]
    assert len(requests) == args.concurrency
    assert all(r["completed"] and r["output_tokens"] == args.output_len for r in requests)
    start = min(r["submitted"] for r in requests)
    end = max(r["completed"] for r in requests)
    tokens = sum(r["output_tokens"] for r in requests)
    drafts = sum(r["draft_tokens"] for r in requests)
    accepted = sum(r["accepted_draft_tokens"] for r in requests)
    summary = dict(status="success", workload="one simultaneous offline burst; warmup and diagnostic excluded",
        concurrency_global=args.concurrency, concurrency_per_rank=args.concurrency//args.dp,
        requests=len(requests), input_tokens_per_request=args.input_len,
        output_tokens_per_request=args.output_len, total_output_tokens=tokens,
        duration_seconds=end-start, output_tokens_per_second=tokens/(end-start),
        ttft_seconds=stats([r["first_token"]-r["submitted"] for r in requests]),
        latency_seconds=stats([r["completed"]-r["submitted"] for r in requests]),
        request_decode_tokens_per_second=stats([(r["output_tokens"]-1)/(r["completed"]-r["first_token"]) for r in requests]),
        accepted_draft_tokens=accepted, proposed_draft_tokens=drafts,
        draft_acceptance=accepted/drafts if drafts else None,
        peak_torch_allocated_gib=[d["peak_allocated_bytes"]/2**30 for d in data],
        peak_torch_reserved_gib=[d["peak_reserved_bytes"]/2**30 for d in data],
        comm_sms=[d["comm_sms"] for d in data], nextn_decode=data[0]["nextn_decode"],
        mtp_varlen=data[0]["mtp_varlen"])
    samples = list(csv.DictReader((args.result_dir / "hbm_samples.csv").open()))
    for name, rows in [("entire_run", samples), ("measurement", [s for s in samples if start <= float(s["timestamp"]) <= end])]:
        summary[f"sampled_hbm_peak_gib_{name}"] = {
            str(rank): max([int(s["used_mib"]) / 1024 for s in rows if int(s["device"]) == rank], default=None)
            for rank in range(args.dp)}
    save(args.result_dir / "summary.json", summary)
    # Detect DeepJIT compilation inside the measurement window (e.g. tail batches).
    jit_cache = Path("/root/.dj/cache")
    if jit_cache.is_dir():
        compiled = [dict(path=str(p), modified=p.stat().st_mtime)
                    for p in jit_cache.rglob("kernel.o") if start <= p.stat().st_mtime <= end]
        save(args.result_dir / "deepjit_measurement_activity.json",
             dict(status="checked", cache=str(jit_cache), kernels=compiled))
    print("BENCH SUMMARY\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    args = arguments()
    args.result_dir = args.result_dir.resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if list(args.result_dir.glob("measure-rank*.json")):
        raise SystemExit("Use a fresh result directory to avoid mixing runs")
    if args.dataset:
        args.dataset = str(Path(args.dataset).resolve())
        questions = load_questions(args.dataset)
        assert len(questions) >= args.concurrency * 3, "Need disjoint questions for all three phases"
        save(args.result_dir / "dataset.json", dict(path=args.dataset, rows=len(questions),
            sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
            construction="question only plus two newlines; repeat token IDs and truncate to input-len; disjoint rows across ranks and phases"))
    save(args.result_dir / "config.json", {**vars(args), "result_dir": str(args.result_dir),
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("DEEPINFER_", "HCCL_", "ASCEND_", "OMP_"))}})
    stop = threading.Event()
    monitor = threading.Thread(target=memory_monitor, args=(args.result_dir, stop), daemon=True)
    monitor.start()
    try:
        from _launch import launch_dp
        launch_dp(worker, args.dp)
    finally:
        stop.set()
        monitor.join(timeout=20)
    aggregate(args)
