"""Recompute successful offline records using AISBench Default/total formulas."""
import argparse
import csv
import json
from pathlib import Path
from benchmark_configurable import metrics, percentile

REFERENCE = '7bcb6440da11bef020943b2af9ed7c1ac8362306'


def export(suite):
    output = suite / 'aisbench_aligned'
    output.mkdir(exist_ok=True)
    rows, displays = [], []
    for case in ['4K_2K', '64K_1K', '128K_1K']:
        root = suite / case
        config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        records = [record for rank in range(config['dp']) for record in
            json.loads((root / f'measure-rank{rank}.json').read_text())['request_records']]
        assert len(records) == config['num_requests']
        assert all(r['output_tokens'] == config['output_len'] and r['completed'] for r in records)
        row = {'用例': case.replace('_', '+'), **metrics(records, config['dp'], max_concurrency=config['max_concurrency'])}
        rows.append(row)
        n = len(records)
        duration = max(r['completed'] for r in records) - min(r['submitted'] for r in records)
        latency = [r['completed']-r['submitted'] for r in records]
        ttft = [r['first_token']-r['submitted'] for r in records]
        tpot = [(r['completed']-r['first_token'])/(r['output_tokens']-1) for r in records]
        display = {'case': row['用例'], 'stage': 'total', 'timing_scope': 'offline engine, not HTTP client',
            'reference_commit': REFERENCE, 'Benchmark Duration(ms)': round(duration*1000,4),
            'Total Requests': n, 'Success Requests': n, 'Failed Requests': 0,
            'Concurrency': round(sum(latency)/duration,4), 'Max Concurrency': config['max_concurrency'],
            'Request Throughput(req/s)': round(n/duration,4),
            'Input Token Throughput(token/s)': round(row['实测输入吞吐(tps)'],4),
            'Output Token Throughput(token/s)': round(row['实测输出吞吐(tps)'],4),
            'Total Token Throughput(token/s)': round(row['实测E2E吞吐(tps)'],4)}
        for key, values in [('TTFT',ttft),('TPOT',tpot),('E2EL',latency)]:
            # Official calculator rounds seconds to 4 decimals, then converts to ms.
            display[key+' Average(ms)'] = round(round(sum(values)/n,4)*1000,4)
            display[key+' P90(ms)'] = round(round(percentile(values,.9),4)*1000,4)
        display['OutputTokenThroughput Average(token/s)'] = round(sum(r['output_tokens']/l for r,l in zip(records,latency))/n,4)
        displays.append(display)
    for name, data in [('results',rows),('aisbench_display',displays)]:
        (output/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(output)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite',type=Path)
    export(p.parse_args().suite)
