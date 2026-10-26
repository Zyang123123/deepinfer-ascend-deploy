import unittest
from unittest.mock import patch
from benchmark_configurable import allocation, arguments, metrics


class BenchmarkTests(unittest.TestCase):
    def test_long_context_capacity_and_invalid_limit(self):
        for input_len, output_len, expected in [(4096,2048,32768),(65536,1024,98304),(131072,1024,163840)]:
            with patch('sys.argv', ['bench','--result-dir','unused','--input-len',str(input_len),'--output-len',str(output_len)]):
                self.assertEqual(arguments().max_model_len, expected)
        with patch('sys.argv', ['bench','--result-dir','unused','--input-len','131072','--output-len','1024','--max-model-len','131072']):
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                arguments()

    def test_requested_allocations(self):
        for total, concurrency in [(400, 200), (48, 12), (32, 8), (37, 12)]:
            partitions = [allocation(total, concurrency, 8, rank) for rank in range(8)]
            self.assertEqual(sorted(i for ids, _ in partitions for i in ids), list(range(total)))
            self.assertEqual(sum(cap for _, cap in partitions), concurrency)
            self.assertTrue(all(len(ids) >= cap > 0 for ids, cap in partitions))
        self.assertEqual([allocation(48, 12, 8, rank)[1] for rank in range(8)], [2,2,2,2,1,1,1,1])

    def test_closed_loop_limits_and_completion(self):
        # Uneven service times and rank capacities: refill only a finished slot.
        for total, concurrency in [(400,200), (48,12), (32,8)]:
            ranks = [allocation(total, concurrency, 8, r) for r in range(8)]
            queues = [list(ids) for ids, _ in ranks]
            active = [[] for _ in ranks]
            completed, peak, tick = [], 0, 0
            while len(completed) < total:
                for rank, (_, cap) in enumerate(ranks):
                    completed.extend(i for i, end in active[rank] if end <= tick)
                    active[rank] = [(i,end) for i,end in active[rank] if end > tick]
                    while queues[rank] and len(active[rank]) < cap:
                        i = queues[rank].pop(0)
                        active[rank].append((i,tick+1+i%7))
                peak = max(peak, sum(map(len, active)))
                tick += 1
            self.assertEqual(sorted(completed), list(range(total)))
            self.assertEqual(peak, concurrency)

    def test_metrics_with_overlap_and_tied_events(self):
        records = [dict(submitted=0., first_token=1., completed=3., input_tokens=10, output_tokens=3),
                   dict(submitted=1., first_token=2., completed=5., input_tokens=10, output_tokens=3),
                   dict(submitted=3., first_token=4., completed=7., input_tokens=10, output_tokens=3)]
        result = metrics(records, 2, 1200, max_concurrency=4)
        self.assertEqual(result['实测总请求数'], 3)
        self.assertEqual(result['实测最大并发数'], 4)
        self.assertEqual(result['观测峰值在途并发数'], 2)
        self.assertAlmostEqual(result['实测系统并发数'], 11/7)
        self.assertAlmostEqual(result['实测请求频率(req/s)'], 3/7)
        self.assertEqual(result['提交阶段请求频率(req/s)'], 1)
        self.assertEqual(result['实测TTFT P90(ms)'], 1000)
        self.assertAlmostEqual(result['实测TPOT平均(ms)'], 4000/3)
        self.assertEqual(result['实测TPOT SLO_P90(ms)'], 1500)
        self.assertAlmostEqual(result['实测QPS'], 3/7)
        self.assertAlmostEqual(result['实测输出吞吐(tps)'], 9/7)
        self.assertAlmostEqual(result['实测输入吞吐(tps)'], 30/7)
        self.assertAlmostEqual(result['实测输入吞吐(tps)'] + result['实测输出吞吐(tps)'], result['实测E2E吞吐(tps)'])
        self.assertAlmostEqual(result['实测E2E吞吐(tps)'], 39/7)
        self.assertAlmostEqual(result['Total Token Throughput(token/s)'], 39/7)
        self.assertNotIn('实测E2E时间(s)', result)
        self.assertNotIn('实测E2EL时间(s)', result)
        self.assertAlmostEqual(result['实测单卡E2E吞吐(tps)'], 39/14)
        self.assertAlmostEqual(result['TPOT_SLO达标率'], 1/3)

    def test_incomplete_is_not_success(self):
        with self.assertRaises(ValueError):
            metrics([dict(submitted=0,first_token=None,completed=None)], 8)


if __name__ == '__main__':
    unittest.main()
