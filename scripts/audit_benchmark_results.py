#!/usr/bin/env python3
"""Audit a completed configurable case without importing torch or loading a model."""
import argparse
import json
from pathlib import Path
import re


def audit(directory):
    root = Path(directory)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    ranks = [json.loads((root / f'measure-rank{r}.json').read_text(encoding='utf-8')) for r in range(config['dp'])]
    records = [row for rank in ranks for row in rank['request_records']]
    assert len(records) == config['num_requests']
    assert len({r['index'] for r in records}) == len(records)
    assert all(r['input_tokens'] == config['input_len'] and r['output_tokens'] == config['output_len'] for r in records)
    assert all(r['submitted'] <= r['first_token'] <= r['completed'] for r in records)
    start = min(r['submitted'] for r in records)
    end = max(r['completed'] for r in records)
    workers = {json.loads((root / f'worker-rank{r}.json').read_text())['pid'] for r in range(config['dp'])}
    snapshots, unexpected, observed = 0, [], set()
    with (root / 'npu_smi_snapshots.jsonl').open() as stream:
        for line in stream:
            item = json.loads(line)
            if start <= item['timestamp'] <= end:
                snapshots += 1
                pids = {int(m) for m in re.findall(r'^\|\s*\d+\s*\|\s*(\d+)\s*\|\s*\S+\s*\|', item['output'], re.M)}
                observed.update(pids)
                if pids - workers:
                    unexpected.append(dict(timestamp=item['timestamp'], pids=sorted(pids - workers)))
    result = dict(request_count=len(records), input_tokens_verified=config['input_len'],
        output_tokens_verified=config['output_len'], unique_request_indexes=True,
        measurement_snapshots=snapshots, expected_worker_pids=sorted(workers),
        observed_pids=sorted(observed), unexpected_process_samples=unexpected,
        all_workers_observed=workers <= observed,
        jit=json.loads((root / 'jit_audit.json').read_text()),
        note='Process occupancy and HBM are sampled, not a continuous exclusion guarantee.')
    (root / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()
    for directory in args.directories:
        print(directory, json.dumps(audit(directory), ensure_ascii=False), flush=True)
