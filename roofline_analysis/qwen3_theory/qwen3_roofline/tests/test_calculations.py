"""Independent numerical references and invariants; no accelerator required."""
import math
from pathlib import Path
import sys
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import roofline_qwen3 as r


class CalculationTests(unittest.TestCase):
    def close(self, actual, expected, rel=1e-4):
        self.assertTrue(math.isclose(actual, expected, rel_tol=rel), (actual, expected))

    def test_ridge(self):
        self.close(r.ridge_point(r.HARDWARE), 184.32, 1e-12)

    def test_kv_token(self):
        self.assertEqual(r.kv_bytes_per_token(), 144 * 1024)

    def test_linear_reference(self):
        self.close(2 * r.parameter_statistics()['linear'] / 1e9, 13.892)

    def test_head_reference(self):
        self.close(2 * r.parameter_statistics()['lm_head'] / 1e9, 1.245, 5e-4)

    def test_prefill_reference(self):
        self.close(sum(r.prefill_flops(1024).values()) / 1e12, 14.536)

    def test_decode_reference(self):
        self.close(sum(r.decode_flops(1, 1024).values()) / 1e9, 15.741)

    def test_decode_ai(self):
        self.close(r.evaluate('decode', 1, 1024)['arithmetic_intensity'], 1.03, 2e-3)

    def test_roof_and_time_duality(self):
        for phase in ('prefill', 'decode'):
            for batch in (1, 8, 256):
                for length in (1, 1024, 32768):
                    x = r.evaluate(phase, batch, length)
                    self.assertLessEqual(x['roof_tflops'], 294.912 + 1e-10)
                    self.assertLessEqual(x['roof_tflops'], 1.6 * x['arithmetic_intensity'] + 1e-10)
                    self.close(x['flops'] / x['time_lower_s'] / 1e12, x['roof_tflops'], 1e-12)

    def test_embedding_lookup_not_dense_flops(self):
        p = r.parameter_statistics()
        self.assertEqual(p['input_embedding'], 4096 * 151936)
        self.assertEqual(p['input_embedding'], p['lm_head'])
        # 6,945,767,424 linear + 2*622,329,856 embedding/head
        # + 73*4096 RMS scales + 72*128 Q/K norm scales.
        self.assertEqual(p['total'], 8190735360)
        self.assertNotIn('input_embedding', r.decode_flops(1, 0))

    def test_gqa_queries_not_kv_heads(self):
        f = r.decode_flops(3, 15)
        self.assertEqual(f['attention_qk'], 3 * 36 * 32 * 16 * 128 * 2)
        self.assertEqual(f['attention_qk'], f['attention_av'])

    def test_causal_single_token_equivalence(self):
        self.assertEqual(r.prefill_flops(1, batch=2), r.decode_flops(2, 0))

    def test_head_all_positions(self):
        last = r.prefill_flops(17, batch=3)
        all_pos = r.prefill_flops(17, batch=3, lm_head_mode='all')
        self.assertEqual(all_pos['lm_head'], 17 * last['lm_head'])
        self.assertEqual(all_pos['q_proj'], last['q_proj'])

    def test_batch_scales_flops_and_amortizes_weights(self):
        one = r.evaluate('decode', 1, 4096)
        many = r.evaluate('decode', 8, 4096)
        self.assertEqual(many['flops'], 8 * one['flops'])
        self.assertEqual(many['weight_bytes'], one['weight_bytes'])
        self.assertEqual(many['kv_read_bytes'], 8 * one['kv_read_bytes'])
        self.close(many['bytes_per_output_token'], one['weight_bytes'] / 8 + 4097 * 144 * 1024)

    def test_activation_sensitivity(self):
        base = r.evaluate('prefill', 1, 1024)
        extra = r.evaluate('prefill', 1, 1024, extra_bytes_per_token=2 * 2**20)
        self.assertEqual(extra['traffic_bytes'] - base['traffic_bytes'], 1024 * 2 * 2**20)
        self.assertLess(extra['arithmetic_intensity'], base['arithmetic_intensity'])

    def test_capacity_is_separate_from_traffic(self):
        small = r.evaluate('decode', 1, 32768)
        large = r.evaluate('decode', 256, 32768)
        self.assertTrue(small['fits_capacity_lower_bound'])
        self.assertFalse(large['fits_capacity_lower_bound'])
        self.assertGreater(large['resident_lower_bound_bytes'], 64 * 2**30)

    def test_hypothetical_and_custom_hardware(self):
        self.assertEqual(r.ridge_point(r.HYPOTHETICAL_HARDWARE), 50)
        x = r.evaluate('prefill', 1, 8192, hardware={'name': 'Test', 'peak_bf16_tflops': 10, 'hbm_bandwidth_tb_s': 0.1})
        self.assertLessEqual(x['roof_tflops'], 10)

    def test_invalid_inputs(self):
        for phase, batch, length in [('prefill', 1, 0), ('decode', 0, 10), ('decode', 1, -1), ('x', 1, 1), ('decode', 1.5, 1)]:
            with self.assertRaises(ValueError):
                r.evaluate(phase, batch, length)
        with self.assertRaises(ValueError):
            r.evaluate('prefill', 1, 1, extra_bytes_per_token=-1)
        with self.assertRaises(ValueError):
            r.ridge_point({'peak_bf16_tflops': 1, 'hbm_bandwidth_tb_s': 0})

    def test_local_module_operand_model(self):
        # One weight read + input/output materialization; independent from base HBM model.
        rows = r.module_roofline_rows('prefill', 1, 1024)
        q = next(row for row in rows if row['module'] == 'q_proj')
        self.assertEqual(q['estimated_operand_bytes'], 1811939328)
        self.close(q['arithmetic_intensity'], 682.6666666667)
        self.close(q['roof_tflops'], 294.912)
        embedding = rows[0]
        self.assertEqual(embedding['flops'], 0)
        self.assertEqual(embedding['estimated_operand_bytes'], 16777216)
        self.assertEqual(embedding['roof_tflops'], '')
        qk = next(row for row in r.module_roofline_rows('decode', 1, 1024) if row['module'] == 'attention_qk')
        self.assertEqual(qk['estimated_operand_bytes'], 78227712)

    def test_extreme_hardware_roofline_stays_inside_canvas(self):
        # Previously ridge=1e6 was outside xlim=1000; annotation caused huge blank output.
        from PIL import Image
        plt = r._plot_setup()
        hardware = {'name': 'High ridge test', 'peak_bf16_tflops': 10000, 'hbm_bandwidth_tb_s': 0.01}
        prefill = [r.evaluate('prefill', 2, 7, hardware=hardware, lm_head_mode='all')]
        decode = [r.evaluate('decode', 3, 0, hardware=hardware)]
        with tempfile.TemporaryDirectory() as directory:
            r.plot_roofline(plt, Path(directory), prefill, decode, hardware)
            with Image.open(Path(directory)/'01_roofline.png') as picture:
                self.assertLess(picture.width/picture.height, 2.3)


if __name__ == '__main__':
    unittest.main()
