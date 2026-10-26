#!/usr/bin/env python3
"""Closed-loop DeepInfer offline benchmark. Requires benchmark_4k2k.py helpers."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

from benchmark_4k2k import ROOT, load_questions, memory_monitor, save
sys.path[:0] = [str(ROOT / 'examples'), str(ROOT)]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='/home/weights/DeepSeek-V4.1-Flash-1017')
    p.add_argument('--dataset', default=str(Path(__file__).with_name('GSM8K.jsonl')))
    p.add_argument('--input-len', type=int, default=4096)
    p.add_argument('--output-len', type=int, default=2048)
    p.add_argument('--num-requests', type=int, default=400)
    p.add_argument('--max-concurrency', type=int, default=200)
    p.add_argument('--dp', type=int, default=8)
    p.add_argument('--max-batched-tokens', type=int, default=4096)
    p.add_argument('--max-model-len', type=int, default=None)
    p.add_argument('--memory-utilization', type=float, default=.95)
    p.add_argument('--warmup-requests', type=int, default=None)
    p.add_argument('--seed', type=int, default=1024)
    p.add_argument('--tpot-slo-ms', type=float, default=None)
    p.add_argument('--result-dir', type=Path, required=True)
    a = p.parse_args()
    if min(a.input_len, a.output_len, a.num_requests, a.max_concurrency, a.dp, a.max_batched_tokens) <= 0:
        p.error('Lengths, request count, concurrency, dp and token budget must be positive')
    if a.output_len < 2:
        p.error('output-len must be >= 2 to measure TPOT')
    if a.max_concurrency < a.dp or a.num_requests < a.dp:
        p.error('This EP/DP runner needs at least one request and concurrency slot per rank')
    if not 0 < a.memory_utilization < 1:
        p.error('memory-utilization must be between 0 and 1')
    if a.tpot_slo_ms is not None and a.tpot_slo_ms <= 0:
        p.error('tpot-slo-ms must be positive')
    if a.warmup_requests is None:
        a.warmup_requests = min(a.max_concurrency, a.num_requests)
    if a.warmup_requests < 0 or (0 < a.warmup_requests < a.dp):
        p.error('warmup-requests must be 0 or >= dp')
    required = a.input_len + a.output_len + max(6, int(os.getenv('DEEPINFER_USE_NEXTN_DECODE', '5')) + 1)
    if a.max_model_len is None:
        a.max_model_len = max(32768, math.ceil(required / 32768) * 32768)
    if a.max_model_len < required:
        p.error(f'max-model-len must be >= {required}; inputs are never silently truncated')
    return a


def allocation(total, concurrency, dp, rank):
    """Assign global lanes round-robin; distribute requests proportional to lanes."""
    slots = min(total, concurrency)
    indexes = [i for i in range(total) if (i % slots) % dp == rank]
    capacity = sum(lane % dp == rank for lane in range(slots))
    return indexes, capacity


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def metrics(records, cards, slo=None, max_concurrency=None):
    if not records or any(r['completed'] is None or r['first_token'] is None for r in records):
        raise ValueError('Incomplete request records')
    start = min(r['submitted'] for r in records)
    end = max(r['completed'] for r in records)
    duration = end - start
    if duration <= 0:
        raise ValueError('Non-positive measurement duration')
    ttft = [(r['first_token'] - r['submitted']) * 1000 for r in records]
    latencies = [r['completed'] - r['submitted'] for r in records]
    tpots = [(r['completed'] - r['first_token']) * 1000 / (r['output_tokens'] - 1) for r in records]
    if min(ttft + latencies + tpots) < 0:
        raise ValueError('Non-monotonic timestamps')
    # Finish events precede submit events at identical timestamps.
    events = sorted([(r['submitted'], 1) for r in records] + [(r['completed'], -1) for r in records])
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    outputs = sum(r['output_tokens'] for r in records)
    inputs = sum(r['input_tokens'] for r in records)
    submission_span = max(r['submitted'] for r in records) - start
    e2e_tps = (inputs + outputs) / duration
    mean_latency = sum(latencies) / len(records)
    result = {
        '实测总请求数': len(records),
        '实测最大并发数': max_concurrency if max_concurrency is not None else peak,
        '观测峰值在途并发数': peak,
        '实测系统并发数': sum(latencies) / duration,
        '实测请求频率(req/s)': len(records) / duration,
        '提交阶段请求频率(req/s)': len(records) / submission_span if submission_span > 0 else None,
        '实测TTFT平均(ms)': sum(ttft) / len(ttft),
        '实测TTFT P90(ms)': percentile(ttft, .9),
        '实测TPOT平均(ms)': sum(tpots) / len(tpots),
        '实测TPOT SLO_P90(ms)': percentile(tpots, .9),
        '实测输入吞吐(tps)': inputs / duration,
        '实测输出吞吐(tps)': outputs / duration,
        '实测单卡输出吞吐(tps)': outputs / duration / cards,
        '实测E2E吞吐(tps)/端到端时延(s)': f'{e2e_tps:.6f}/{mean_latency:.6f}',
        '实测单卡E2E吞吐(tps)': e2e_tps / cards,
        '实测QPS': len(records) / duration,
        '实测QPM': len(records) / duration * 60,
        '实测E2E吞吐(tps)': e2e_tps,
        'Total Token Throughput(token/s)': e2e_tps,
        '实测请求端到端时延平均(s)': mean_latency,
        '实测请求端到端时延P90(s)': percentile(latencies, .9),
        '实际输入token总数': inputs,
        '实际输出token总数': outputs,
        'TPOT_SLO阈值(ms)': slo,
        'TPOT_SLO达标率': sum(t <= slo for t in tpots) / len(tpots) if slo is not None else None,
    }
    return result


def worker(dp_size, rank):
    args = arguments()
    save(args.result_dir / f'worker-rank{rank}.json', dict(rank=rank, pid=os.getpid()))
    from _config import create_llm, setup_common_env
    os.environ['DEEPINFER_FORCE_LOAD_BALANCE'] = '0'
    os.environ['DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS'] = '0'
    setup_common_env(model_name=args.model, sampler_force_topk=4096,
                     set_override_experts=False, enable_eplb=False)
    llm = create_llm(args.model, rank, ep_size=dp_size, cp_size=1,
        env={'DEEPINFER_LOGGING_INTERVAL_STEP': '100'},
        data_parallel_size=dp_size, max_num_seqs=math.ceil(args.max_concurrency / dp_size),
        max_num_batched_tokens=args.max_batched_tokens, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.memory_utilization, seed=args.seed,
        lookahead_tokens=min(1, int(os.getenv('DEEPINFER_USE_NEXTN_DECODE', '5'))))
    import numpy as np
    import torch
    import torch_npu
    from deepinfer._C.engine import RequestOutput
    from deepinfer.model_executor.parallel_utils.parallel_state import PG
    from deepinfer.sampling_params import SamplingParams
    group = PG.cpu_data_parallel_group()
    engine = llm.llm_engine
    questions = load_questions(args.dataset)
    selection = np.random.default_rng(args.seed).permutation(len(questions))
    params = SamplingParams(max_tokens=args.output_len, ignore_eos=True, temperature=0., top_p=1.)

    def run_phase(name, total, offset):
        indexes, capacity = allocation(total, args.max_concurrency, dp_size, rank)
        prompts, manifest = [], []
        for index in indexes:
            # For large suites, reuse rows after a full pass; record reuse explicitly.
            row_id = int(selection[(offset + index) % len(selection)])
            base = np.asarray(llm.get_tokenizer().encode(questions[row_id] + '\n\n', add_special_tokens=False), dtype=np.int32)
            if not len(base):
                raise ValueError('Empty encoded question')
            ids = np.resize(base, args.input_len)
            prompts.append(ids)
            manifest.append(dict(index=index, row_id=row_id, input_tokens=len(ids),
                original_tokens=len(base), sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
        save(args.result_dir / f'{name}-prompts-rank{rank}.json', manifest)
        torch.npu.synchronize()
        torch.distributed.barrier(group=group)
        torch.npu.reset_peak_memory_stats()
        records, active = {}, set()
        cursor = step = 0
        last_report = time.monotonic()
        while True:
            while cursor < len(prompts) and len(active) < capacity:
                submitted = time.time()
                seq = llm._add_request(None, params, prompts[cursor])
                rid = str(seq.request_id)
                records[rid] = dict(rank=rank, request_id=rid, index=indexes[cursor],
                    submitted=submitted, first_token=None, completed=None,
                    input_tokens=len(prompts[cursor]), output_tokens=0)
                active.add(rid)
                cursor += 1
            # All EP ranks must continue collective stepping even if locally idle.
            unfinished = engine.has_unfinished_requests()
            outputs = engine.step() if unfinished else engine.flush_inflight_reqs()
            now = time.time()
            for raw in outputs:
                output = RequestOutput.clone(raw)
                rid = str(output.request_id)
                record = records[rid]
                assert output.prompt_len == args.input_len
                if record['first_token'] is None:
                    record['first_token'] = now
                record['output_tokens'] += 1
                if output.finished:
                    record['completed'] = now
                    active.remove(rid)
                    # Preserve completed measurements even if a later request fails.
                    with (args.result_dir / f'{name}-requests-rank{rank}.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(record) + '\n')
            step += 1
            if rank == 0 and time.monotonic() - last_report >= 20:
                print(f'BENCH {name}: rank0 submitted={cursor}/{len(prompts)} active={len(active)} step={step}', flush=True)
                last_report = time.monotonic()
            if not unfinished:
                pending = torch.tensor([len(prompts) - cursor + len(active)], dtype=torch.int64)
                torch.distributed.all_reduce(pending, group=group)
                if pending.item() == 0:
                    break
        torch.npu.synchronize()
        assert len(records) == len(indexes)
        assert all(r['completed'] is not None and r['output_tokens'] == args.output_len for r in records.values())
        save(args.result_dir / f'{name}-rank{rank}.json', dict(rank=rank, capacity=capacity,
            request_records=list(records.values()), peak_allocated_bytes=torch.npu.max_memory_allocated(),
            peak_reserved_bytes=torch.npu.max_memory_reserved()))
        torch.distributed.barrier(group=group)

    try:
        if args.warmup_requests:
            run_phase('warmup', args.warmup_requests, 0)
        run_phase('measure', args.num_requests, args.warmup_requests)
    finally:
        llm.close()


def aggregate(args):
    ranks = [json.loads((args.result_dir / f'measure-rank{r}.json').read_text()) for r in range(args.dp)]
    records = [record for rank in ranks for record in rank['request_records']]
    assert len(records) == args.num_requests
    assert all(r['input_tokens'] == args.input_len and r['output_tokens'] == args.output_len for r in records)
    result = metrics(records, args.dp, args.tpot_slo_ms, args.max_concurrency)
    assert result['观测峰值在途并发数'] <= args.max_concurrency
    result = {'状态': 'success', '序列长度：输入+输出': f'{args.input_len}+{args.output_len}',
              '配置总请求数': args.num_requests, '配置最大并发数': args.max_concurrency, **result}
    samples = list(csv.DictReader((args.result_dir / 'hbm_samples.csv').open()))
    start, end = min(r['submitted'] for r in records), max(r['completed'] for r in records)
    peaks = {str(rank): max([int(s['used_mib']) / 1024 for s in samples
        if int(s['device']) == rank and start <= float(s['timestamp']) <= end], default=None) for rank in range(args.dp)}
    save(args.result_dir / 'memory.json', dict(sampled_measurement_hbm_gib=peaks,
        torch_allocated_gib=[r['peak_allocated_bytes']/2**30 for r in ranks],
        torch_reserved_gib=[r['peak_reserved_bytes']/2**30 for r in ranks]))
    cache = Path('/root/.dj/cache')
    compiled = [str(p) for p in cache.rglob('kernel.o') if start <= p.stat().st_mtime <= end] if cache.is_dir() else None
    save(args.result_dir / 'jit_audit.json', dict(compiled_during_measurement=compiled))
    result['测量期间新增JIT编译数'] = len(compiled) if compiled is not None else None
    save(args.result_dir / 'summary.json', result)
    with (args.result_dir / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    args = arguments()
    args.result_dir = args.result_dir.resolve()
    args.dataset = str(Path(args.dataset).resolve())
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if (args.result_dir / 'config.json').exists():
        raise SystemExit('Result directory already used; choose a fresh directory')
    questions = load_questions(args.dataset)
    if not questions:
        raise ValueError('Empty dataset')
    save(args.result_dir / 'config.json', {**vars(args), 'result_dir': str(args.result_dir),
        'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        'dataset_rows': len(questions), 'dataset_reused': args.num_requests + args.warmup_requests > len(questions),
        'environment': {k:v for k,v in os.environ.items() if k.startswith(('DEEPINFER_', 'HCCL_', 'ASCEND_', 'OMP_'))}})
    stop = threading.Event()
    monitor = threading.Thread(target=memory_monitor, args=(args.result_dir, stop), daemon=True)
    monitor.start()
    try:
        from _launch import launch_dp
        launch_dp(worker, args.dp)
    except BaseException as exc:
        save(args.result_dir / 'failure.json', dict(status='failed', error=repr(exc)))
        raise
    finally:
        stop.set()
        monitor.join(timeout=20)
    aggregate(args)


if __name__ == '__main__':
    main()
