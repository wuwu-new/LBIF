"""Stdlib-only concurrency, transform, identity and finite-budget tests."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import test_target_prefetch_20261105 as original_tests
from build_latest_ablation_prefetch_20261105 import runner,verifier,remaining_source_runner
SOURCE=ROOT/'experiments/guarded_transfer_20261026/ffpp_latest_dual_ablation_seed2027_20261105'
MOD=ROOT/'code/dual_stream_20261026/latest_ablation_prefetch_20261105'
spec=importlib.util.spec_from_file_location('test_latest_prefetch_plan',MOD/'plan.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
TEXT=(SOURCE/'bundle/implementation/run_cuda.py').read_text(encoding='utf-8')


class Checks(original_tests.PrefetchChecks):
    def test_protocol(self):
        cfg=p.configuration()
        self.assertEqual((p.ENDPOINTS,p.EXTRA_BENCHMARK_FORWARDS),(5,5120))
        self.assertEqual(cfg['original_target_probe_forwards'],1440)
        self.assertEqual(cfg['target_preflight_forwards_per_invocation'],6560)
        self.assertEqual(cfg['target_preflight_total_forwards']+cfg['full_target_residual_forwards'],1123680)
        self.assertFalse(cfg['source_training_authorized'])
        self.assertEqual(cfg['retained_updates'],0)
        self.assertEqual(list(p.SEEDS),[2027])
        self.assertEqual(cfg['adopted_source_budget']['retained_updates'],35950)
        self.assertEqual(cfg['selected_reload_forwards'],320)

    def test_original_bundle_unchanged(self):
        lock=json.loads((SOURCE/'REGISTRATION.json').read_text(encoding='utf-8'))
        self.assertEqual(lock['registration_sha256'],p.SOURCE_REG)
        for name,digest in lock['files'].items():
            self.assertEqual(hashlib.sha256((SOURCE/'bundle'/name).read_bytes()).hexdigest(),digest,name)

    def test_runner_AST_and_target_only(self):
        text=runner(TEXT);ast.parse(text)
        self.assertNotIn('AdamW(',text)
        self.assertNotIn('loss.backward()',text)
        self.assertIn("choices=('target-preflight','target')",text)
        self.assertIn("done=exited(run,'SOURCE',p.SOURCE_REG)",text)
        self.assertIn("719*endpoint['epoch'],p.SOURCE_REG]",text)
        self.assertIn('torch.equal(xx,parallel_xx)',text)
        self.assertIn("choice['enable_prefetch']",text)
        self.assertIn("('logit','delta','weight_mass','pooled_norm')",text)
        self.assertNotIn('7008',text)
        self.assertNotIn('6144',text)

    def test_benchmark_once(self):
        text=runner(TEXT)
        self.assertEqual(text.count('for mode in (False,True,True,False):'),1)
        self.assertIn('benchmark_rows=val[:256]',text)
        self.assertIn('extra_frame_forwards=p.EXTRA_BENCHMARK_FORWARDS',text)
        self.assertIn("assert maximum<=2e-7",text)
        self.assertIn('ratio>=1.05',text)
        self.assertIn("assert sum(meter.values())==p.configuration()['original_target_probe_forwards']",text)
        self.assertIn("preflight_forwards=p.configuration()['original_target_probe_forwards'],new_optimizer_updates=0)",text)

    def test_verifier_AST_and_original_identity(self):
        text=verifier((SOURCE/'bundle/LOCAL_VERIFIER.py').read_text(encoding='utf-8'))
        ast.parse(text)
        self.assertIn("sourceproof['registration_sha256']==m.p.SOURCE_REG",text)
        self.assertIn('acceleration_benchmark=m.p.EXTRA_BENCHMARK_FORWARDS',text)
        self.assertIn("done['network_frame_forwards']==dict(**expected,full_target=cfg['full_target_residual_forwards'])",text)
        self.assertIn('raise ValueError(\'Use original source verifier\')',text)

    def test_all_new_files_AST(self):
        for file in [*MOD.glob('*.py'),ROOT/'tools/build_latest_ablation_prefetch_20261105.py',ROOT/'tools/manage_latest_ablation_prefetch_20261105.py']:
            ast.parse(file.read_text(encoding='utf-8'),filename=str(file))

    def test_CPU_benchmark_no_cuda(self):
        text=(ROOT/'code/dual_stream_20261026/target_prefetch_20261105/cpu_benchmark.py').read_text(encoding='utf-8')
        self.assertNotIn('import torch',text.replace('no torch import',''))
        self.assertIn('total_image_reads=1024',text)
        self.assertIn('os.nice(19)',text)

    def test_model_and_forward_unchanged(self):
        old=ast.parse(TEXT);new=ast.parse(runner(TEXT))
        def fn(tree,name):
            return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name))
        self.assertEqual(fn(old,'forward'),fn(new,'forward'))
        self.assertEqual(fn(old,'make'),fn(new,'make'))
        self.assertIn('torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False',runner(TEXT))

    def test_future_source_draft_does_not_touch_training(self):
        result=remaining_source_runner(TEXT);ast.parse(result)
        begin=TEXT.index('                        for step,ids in enumerate(')
        end=TEXT.index('                        m.eval();logits=[]',begin)
        old=TEXT[begin:end]
        self.assertIn(old,result)
        self.assertEqual(result.count('for start,chunk,arrays in batches:'),2)
        self.assertEqual(result.count('rows=chunk;xx=torch.from_numpy(arrays).cuda().float()/255'),2)
        self.assertNotIn('for start in range(0,22080,16):',result)
        self.assertEqual(result.count('optimizer(m)'),TEXT.count('optimizer(m)'))

    def test_source_declared_size_and_hash_retained(self):
        text=(SOURCE/'bundle/implementation/checked_pixels.py').read_text(encoding='utf-8')
        self.assertIn("assert list(image.size) == row['source_size_wh']",text)
        self.assertIn('hashlib.sha256(raw).hexdigest() == expected',text)

    def test_delivery_gate_not_bypassed(self):
        text=(ROOT/'tools/manage_latest_ablation_prefetch_20261105.py').read_text(encoding='utf-8')
        self.assertIn("assert not (PARENT/'FINAL_DELIVERY.json').exists()",text)
        self.assertIn("assert proof['verified_epochs']==p.ENDPOINTS*10",text)
        self.assertIn("assert not any((src/(prefix+s)).exists()",text)


if __name__=='__main__':unittest.main(verbosity=2)
