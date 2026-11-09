"""Local standard-library checks only; never import torch/PIL/numpy locally."""
import ast
from contextlib import closing
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
MOD=ROOT/'code/dual_stream_20261026/target_prefetch_20261105'
SOURCE=ROOT/'experiments/guarded_transfer_20261026/ffpp_patch_region_full_20261105'
sys.path.insert(0,str(ROOT/'tools'))
from build_target_prefetch_20261105 import runner,verifier


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


prefetch=load('stdlib_prefetch',MOD/'prefetch.py')
plan=load('stdlib_prefetch_plan',MOD/'plan.py')


class PrefetchChecks(unittest.TestCase):
    def run_batches(self,rows,parallel=True,batch_size=16):
        with closing(prefetch.ordered_batches(rows,lambda rr:[dict(r) for r in rr],parallel=parallel,batch_size=batch_size)) as it:
            return list(it)

    def test_order(self):
        rows=[dict(i=i) for i in range(99)]
        values=self.run_batches(rows)
        self.assertEqual([s for s,_,_ in values],list(range(0,99,16)))
        self.assertEqual([r for _,_,batch in values for r in batch],rows)

    def test_serial_equal(self):
        rows=[dict(i=i) for i in range(65)]
        self.assertEqual(self.run_batches(rows),self.run_batches(rows,False))

    def test_tail(self):
        self.assertEqual([len(batch) for _,_,batch in self.run_batches([{}]*33)],[16,16,1])

    def test_empty(self):self.assertEqual(self.run_batches([]),[])

    def test_small(self):self.assertEqual(len(self.run_batches([{}])[0][2]),1)

    def test_no_mutation(self):
        rows=[dict(i=i) for i in range(40)];saved=repr(rows)
        self.run_batches(rows);self.assertEqual(repr(rows),saved)

    def test_invalid_batch(self):
        with self.assertRaises(ValueError):self.run_batches([{}],batch_size=0)

    def test_worker_exception(self):
        def decode(rr):
            if rr[0]==16:raise RuntimeError('expected fixture failure')
            return rr
        with self.assertRaisesRegex(RuntimeError,'expected fixture failure'):
            with closing(prefetch.ordered_batches(list(range(64)),decode)) as it:list(it)

    def test_bound_and_close(self):
        started=[];lock=threading.Lock()
        def decode(rr):
            with lock:started.append(rr[0])
            return rr
        with closing(prefetch.ordered_batches(list(range(1000)),decode)) as it:
            self.assertEqual(next(it)[0],0)
            time.sleep(.03)
            # At most yielded current + two pending, regardless of dataset size.
            self.assertLessEqual(len(started),3)
        self.assertFalse(any(t.name.startswith('checked-pixels') for t in threading.enumerate()))

    def test_completion_order_not_output_order(self):
        def decode(rr):
            if rr[0]==0:time.sleep(.03)
            return rr
        with closing(prefetch.ordered_batches(list(range(50)),decode)) as it:
            self.assertEqual([start for start,_,_ in it],[0,16,32,48])

    def test_stdlib_module(self):
        tree=ast.parse((MOD/'prefetch.py').read_text(encoding='utf-8'))
        top_imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) or 'numpy' in ast.unparse(n) or 'PIL' in ast.unparse(n) for n in top_imports))

    def test_protocol(self):
        c=plan.configuration()
        self.assertEqual(c['workers'],2);self.assertEqual(c['pending_batches'],2)
        self.assertEqual(c['batch'],16);self.assertEqual(c['retained_updates'],0)
        self.assertEqual(c['target_preflight_total_forwards']+c['full_target_residual_forwards'],677280)
        self.assertEqual(c['target_preflight_forwards_per_invocation'],7008)
        self.assertFalse(c['source_training_authorized'])
        self.assertEqual(plan.NEW_ARMS,('PatchClassificationContrast',))

    def test_original_bundle_unchanged(self):
        import json
        lock=json.loads((SOURCE/'REGISTRATION.json').read_text(encoding='utf-8'))
        self.assertEqual(lock['registration_sha256'],plan.SOURCE_REG)
        for name,digest in lock['files'].items():
            self.assertEqual(hashlib.sha256((SOURCE/'bundle'/name).read_bytes()).hexdigest(),digest,name)

    def test_runner_AST_and_target_only(self):
        text=runner((SOURCE/'bundle/implementation/run_cuda.py').read_text(encoding='utf-8'))
        ast.parse(text)
        self.assertNotIn('loss.backward()',text)
        self.assertNotIn('AdamW(',text)
        self.assertNotIn('retained_training\');loss',text)
        self.assertIn("choices=('target-preflight','target')",text)
        self.assertIn("done=exited(run,'SOURCE',p.SOURCE_REG)",text)
        self.assertIn("719*endpoint['epoch'],p.SOURCE_REG]",text)
        self.assertIn('torch.equal(xx,parallel_xx)',text)
        self.assertIn("choice['enable_prefetch']",text)
        self.assertIn("meter['acceleration_benchmark']==6144",text)

    def test_benchmark_once(self):
        text=runner((SOURCE/'bundle/implementation/run_cuda.py').read_text(encoding='utf-8'))
        self.assertEqual(text.count('for mode in (False,True,True,False):'),1)
        self.assertIn("dump(run/'ACCELERATION_DECISION.json',choice)",text)
        self.assertIn("ratio>=1.05",text)
        self.assertIn("assert maximum<=2e-7",text)

    def test_verifier_AST_and_original_identity(self):
        text=verifier((SOURCE/'bundle/LOCAL_VERIFIER.py').read_text(encoding='utf-8'))
        ast.parse(text)
        self.assertIn("sourceproof['registration_sha256']==m.p.SOURCE_REG",text)
        self.assertIn("pre['preflight_forwards']==7008",text)
        self.assertIn("dict(source_reload_probe=192,target_repeat_probe=672,full_target=669408)",text)
        self.assertNotIn("sourceproof['registration_sha256']==reg",text)

    def test_all_new_files_AST(self):
        for file in [*MOD.glob('*.py'),ROOT/'tools/manage_target_prefetch_20261105.py']:
            ast.parse(file.read_text(encoding='utf-8'),filename=str(file))

    def test_CPU_benchmark_no_cuda(self):
        text=(MOD/'cpu_benchmark.py').read_text(encoding='utf-8')
        self.assertNotIn('import torch',text.replace('no torch import',''))
        self.assertIn('os.nice(19)',text)
        self.assertIn('total_image_reads=1024',text)


if __name__=='__main__':unittest.main(verbosity=2)
