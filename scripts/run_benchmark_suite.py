#!/usr/bin/env python3
"""Run each case in a fresh engine and collect a UTF-8 BOM CSV, including failures."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

CASES = [('4K+2K', 4096, 2048, 400, 200),
         ('64K+1K', 65536, 1024, 48, 12),
         ('128K+1K', 131072, 1024, 32, 8)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=Path('logs') / datetime.now().strftime('suite-%Y%m%d-%H%M%S'))
    parser.add_argument('--case', choices=['all'] + [c[0] for c in CASES], default='all')
    parser.add_argument('--timeout-seconds', type=int, default=14400, help='per case')
    args, extra = parser.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    reserved = ('--input-len', '--output-len', '--num-requests', '--max-concurrency', '--result-dir')
    if any(arg.split('=')[0] in reserved for arg in extra):
        parser.error('Case lengths/counts are fixed; use run_configurable_36.sh for custom cases')
    if args.timeout_seconds <= 0:
        parser.error('timeout-seconds must be positive')
    root = args.result_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if list(root.iterdir()):
        parser.error('result-root must be empty to avoid mixing runs')
    rows = []
    for label, input_len, output_len, count, concurrency in CASES:
        if args.case not in ('all', label):
            continue
        out = root / label.replace('+', '_')
        out.mkdir()
        command = ['timeout', '--signal=TERM', '--kill-after=30s', str(args.timeout_seconds),
            'bash', str(Path(__file__).with_name('run_configurable_36.sh')),
            '--input-len', str(input_len), '--output-len', str(output_len),
            '--num-requests', str(count), '--max-concurrency', str(concurrency), *extra]
        env = {**os.environ, 'BENCH_RESULT_DIR': str(out)}
        print(f'Running {label}: requests={count}, concurrency={concurrency}', flush=True)
        with (out / 'launcher.log').open('w') as log:
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode == 0 and (out / 'summary.json').exists():
            row = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        else:
            row = {'状态': 'failed', '序列长度：输入+输出': label,
                   '配置总请求数': count, '配置最大并发数': concurrency,
                   '错误': f'exit={proc.returncode}; inspect {out / "launcher.log"}'}
        rows.append(row)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (root / 'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (root / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{label}: {row["状态"]}; results: {root / "results.csv"}', flush=True)
        if row['状态'] != 'success':
            # Do not start another EP job after timeout/failure with uncertain cleanup.
            raise SystemExit('Stopped after failed case. Inspect processes/memory before continuing with --case.')


if __name__ == '__main__':
    main()
