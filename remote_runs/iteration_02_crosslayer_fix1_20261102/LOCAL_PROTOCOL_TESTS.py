"""Static/standard-library tests only; actual models are remote CUDA preflight work."""
import ast
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'code/dual_stream_20261026/iteration02_crosslayer_fix1'
sys.path.insert(0, str(SOURCE))
import crosslayer_plan as p


class CrossLayerTests(unittest.TestCase):
    def test_routes_frozen(self):
        self.assertEqual(p.routes('LastLayerResidual'), (2, 2, 2))
        self.assertEqual(p.routes('CrossLayerResidual'), (0, 1, 2))
        self.assertEqual(p.routes('CrossLayerNoImage'), (0, 1, 2))
        with self.assertRaises(ValueError):
            p.routes('other')

    def test_exact_ten_epoch_budget(self):
        config = p.configuration()
        self.assertEqual(config['total_new_optimizer_updates'], 21570)
        self.assertEqual(config['per_arm_updates'], 7190)
        self.assertEqual(config['primary_endpoint'], dict(epoch=10, g=1))
        self.assertEqual(config['layers'], [8, 16, 24])
        self.assertFalse(config['shared_training_prefix'])
        self.assertFalse(config['original_independent_first_epoch_replay_passed'])

    def test_no_training_selection_leakage(self):
        config = p.configuration()
        for key in ('source_test_access', 'target_labels_for_training', 'layer_or_target_g_or_checkpoint_scan',
                    'promotion', 'invalid_or_hard_sample_filtering', 'source_gain_execution_gate'):
            self.assertIs(config[key], False)
        self.assertTrue(config['adaptive_exploration'])
        self.assertEqual(config['seed_training_sd'], 'NA')

    def test_no_image_not_lower_parameter_capacity(self):
        self.assertEqual(p.configuration()['no_image_parameter_delta_vs_full'], 192 * 32 - (32 * 96 + 32 * 25))
        self.assertGreater(p.configuration()['no_image_parameter_delta_vs_full'], 0)
        text = (SOURCE / 'crosslayer_cuda.py').read_text(encoding='utf-8')
        self.assertIn('self.base.front = nn.Identity()', text)
        self.assertIn('self.base.encoder.body[3] = nn.Identity()', text)
        self.assertIn('self.base.encoder.body[5] = nn.Identity()', text)
        self.assertIn("'No-image control leaks RGB gradient'", text)
        self.assertIn("'No-image control leaks RGB values'", text)

    def test_cache_budget_and_only_new_early_layers(self):
        estimate = p.cache_estimate()
        self.assertEqual(estimate['added_val_bytes'], 4341104640)
        self.assertLess(estimate['two_train_views_bytes'], 10 * 1024**3)
        self.assertEqual(estimate['old_val_reused_bytes'], 5494298880)
        self.assertEqual(estimate['memory_reserve_bytes'], 6 * 1024**3)
        self.assertEqual(estimate['disk_reserve_bytes'], 1024**3)
        for args in ((0, 1), (True, 1), (1, -2)):
            with self.assertRaises(ValueError):
                p.cache_estimate(*args)

    def test_layer_cache_is_real_and_semantic_unchanged(self):
        model = (SOURCE / 'crosslayer_cuda.py').read_text(encoding='utf-8')
        cache = (SOURCE / 'crosslayer_cache.py').read_text(encoding='utf-8')
        for marker in ('output_hidden_states=True', 'out.hidden_states[24], out.last_hidden_state', 'encoder.anchor(raw)',
                       'layer_raw = out.hidden_states[layer].half().float()', 'layer_raw[:, 1:].float()'):
            self.assertIn(marker, model)
        for marker in ('torch.equal(final, old)', 'isolated_rng'):
            self.assertIn(marker, cache)
        self.assertIn('original_cache_duplicated=False', cache)

    def test_syntax_and_cuda_checkpoint_loads(self):
        for file in SOURCE.glob('*.py'):
            tree = ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'load' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch':
                    self.assertEqual(next(k.value.value for k in node.keywords if k.arg == 'map_location'), 'cuda')

    def test_source_summary_fails_on_missing_arm_or_denominator(self):
        with self.assertRaises(ValueError):
            p.source_summary({})
        with self.assertRaises(ValueError):
            p.source_summary({name: [] for name in p.ARMS})

    def test_no_local_model_modules(self):
        self.assertFalse(any(name in sys.modules for name in ('torch', 'numpy', 'PIL')))

    def test_remote_runner_preserves_frozen_semantic_and_update_budget(self):
        text = (SOURCE / 'run_crosslayer_cuda.py').read_text(encoding='utf-8')
        for marker in ('Saved CUDA checkpoint identity mismatch', 'Saved AdamW steps mismatch',
                       'Complete initial semantic reproduction failed', 'Three-arm slot sampling mismatch',
                       'Preflight mutated weights', 'all(v.grad is None for v in model.base.semantic.parameters())',
                       'outputs = losses = result = loss = None'):
            self.assertIn(marker, text)

    def test_archive_preserves_external_cache_and_history(self):
        root = SOURCE.parents[2]
        manager = (root / 'tools/manage_iteration02_crosslayer_fix1.py').read_text(encoding='utf-8')
        for marker in ('excluded_reconstructible_cache', 'remote_preserved=True',
                       '--exclude={NAME}/VAL_EARLY/patches.bin', "'Existing archive preserved; inspect/resume it, do not repeat'",
                       'External GPU user; manual handling, no queue'):
            self.assertIn(marker, manager)
        self.assertNotIn('os.remove(', manager)
        self.assertNotIn('shutil.rmtree(', manager)

    def test_all33_epochs_and_all_three_arms_required_before_source_pass(self):
        verifier = (SOURCE.parents[2] / 'tools/verify_iteration02_source_fix1.py').read_text(encoding='utf-8')
        for marker in ("len(audit['records']) == 33", 'for epoch in range(11)',
                       'full_seven_domains_still_required=True', 'source_frame_prediction_replay=False'):
            self.assertIn(marker, verifier)

    def test_builder_requires_previous_round_verified_and_old_input_hash(self):
        text = (SOURCE.parents[2] / 'tools/manage_iteration02_crosslayer_fix1.py').read_text(encoding='utf-8')
        for marker in ("control['completed_rounds'] == 1", "control['current_round']['iteration'] == 2",
                       'Frozen inherited source inputs changed', 'Frozen file changed'):
            self.assertIn(marker, text)

    def test_repair_preserves_layout_and_ownership(self):
        config = p.configuration()
        self.assertTrue(config['implementation_repair']['scientific_protocol_unchanged'])
        self.assertFalse(config['implementation_repair']['counts_as_new_round'])
        self.assertFalse(config['implementation_repair']['failure_overwritten'])
        runtime = (SOURCE / 'run_crosslayer_cuda.py').read_text(encoding='utf-8')
        self.assertIn('owned_nvml_pid=guard.owned_nvml_pid', runtime)
        self.assertIn('owned_guard_bound_and_checked=True', runtime)
        self.assertIn("anchor = compact.CompactSemantic(config['semantic_checkpoint']).cuda().eval()\n        guard.check()", runtime)
        model = (SOURCE / 'crosslayer_cuda.py').read_text(encoding='utf-8')
        self.assertNotIn('out.hidden_states[layer][:, 1:].half().float()', model)


if __name__ == '__main__':
    unittest.main()
