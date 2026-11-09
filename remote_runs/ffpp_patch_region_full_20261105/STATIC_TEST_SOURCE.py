"""Stdlib-only frozen full-variant routing, budgets and integration checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from build_patch_region_full_20261105 import runner,verifier

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('patch_full_test_plan',ROOT/'code/dual_stream_20261026/patch_region_full_20261105/plan.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
PARENT=ROOT/'experiments/guarded_transfer_20261026'/p.PARENT
IMPL=ROOT/'experiments/guarded_transfer_20261026'/p.IMPLEMENTATION
original=(PARENT/'bundle/implementation/run_cuda.py').read_text(encoding='utf-8')
new=runner(original)
audit=verifier((ROOT/'tools/verify_paper_components_20261104.py').read_text(encoding='utf-8'))


class FullTests(unittest.TestCase):
    def test_01_only_full(self):self.assertEqual(p.NEW_ARMS,('PatchClassificationContrast',))
    def test_02_seeds(self):self.assertEqual(p.SEEDS,(2027,3407,4099))
    def test_03_budget_updates(self):self.assertEqual(p.configuration()['retained_updates'],3*10*719)
    def test_04_training(self):self.assertEqual(p.configuration()['retained_training_frame_forwards'],21570*16)
    def test_05_VAL(self):self.assertEqual(p.configuration()['source_VAL_frame_forwards'],3*10*22080)
    def test_06_selected(self):self.assertEqual(p.configuration()['selected_localization_forwards'],3*22080)
    def test_07_reload(self):self.assertEqual(p.configuration()['selected_reload_forwards'],3*64)
    def test_08_preflight(self):
        self.assertEqual(p.configuration()['source_preflight_total_forwards'],3*64+3*8)
        self.assertEqual(p.configuration()['preflight_disposable_updates'],3*2)
    def test_09_target(self):
        self.assertEqual(p.configuration()['full_target_residual_forwards'],3*223136)
        self.assertEqual(p.configuration()['target_preflight_forwards_per_invocation'],3*(64+7*32))
    def test_10_sampling(self):
        rows=[dict(method=m) for m in p.METHODS for _ in range(7)]
        for s in p.SEEDS:
            for e in range(1,11):self.assertEqual(p.sampling(rows,e,s),p.old.sampling(rows,e,s))
    def test_11_training_objective_connected(self):
        self.assertIn("terms=objective(m,out,yy,mm,valid,split='TRAIN')",new)
        self.assertIn("'return_patch':tag in ('retained_training'",new)
        self.assertIn('patch_classification',' '.join(new.split()))
        self.assertNotIn('loss,bc,aux,penalty=objective(',new)
    def test_12_shared_init_separate(self):
        self.assertIn("if not k.startswith('patch_classifier.')",new)
        self.assertIn('shared_initialization_sha256=shared_initial',new)
        self.assertIn("initialization_sha256'][str(seed)][arm]",new)
    def test_13_real_TRAIN_gradient(self):
        for text in ('real_TRAIN_gradient','fake_indices','[:64]','2<=positives<=254','real_TRAIN_patch_encoder_gradient','real_TRAIN_contrast_encoder_gradient'):
            self.assertIn(text,new)
    def test_14_old_guards_preserved(self):
        for text in ("exited(run,'SOURCE_PREFLIGHT',registration)","exited(run,'TARGET_PREFLIGHT',registration)","SOURCE_LOCAL_VERIFICATION.json",'OwnedCudaGuard','LOCK_EX|fcntl.LOCK_NB','assert maximum<=2e-7'):
            self.assertIn(text,new)
    def test_15_CUDA_only_loading(self):
        for n in ast.walk(ast.parse(new)):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='load':
                self.assertTrue(any(k.arg=='map_location' and isinstance(k.value,ast.Constant) and k.value.value=='cuda' for k in n.keywords))
    def test_16_no_new_model_edits(self):
        lock=json.loads((IMPL/'REGISTRATION.json').read_text(encoding='utf-8'))
        for name in ('model_cuda.py','patch_config.py'):
            file=IMPL/'bundle/implementation'/name
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),lock['files']['implementation/'+name])
    def test_17_verifier_budgets(self):
        for text in ('all30_epochs_verified','all_three_endpoints_seven_domains_verified','full_target_frames=669408',
                     'retained_training=345120,full_source_VAL=662400,selected_reload=192,selected_localization=66240'):
            self.assertIn(text,audit)
    def test_18_loss_logging(self):
        self.assertIn("loss_names=['image','boundary','regularizer','patch_classification','patch_contrast']",new)
        self.assertIn('patch_coverage_totals',new)
        self.assertIn('patch_TP',new)
    def test_19_no_local_models(self):
        for name in ('manage_patch_region_full_20261105.py','build_patch_region_full_20261105.py','test_patch_region_full_20261105.py'):
            tree=ast.parse((ROOT/'tools'/name).read_text(encoding='utf-8'))
            modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            modules +=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
            self.assertFalse(any((m or '').split('.')[0] in ('torch','numpy','PIL','transformers') for m in modules))
    def test_20_compile(self):
        compile(new,'full_runner','exec');compile(audit,'full_verifier','exec')
        for file in (ROOT/'code/dual_stream_20261026/patch_region_full_20261105/plan.py',ROOT/'tools/manage_patch_region_full_20261105.py'):
            compile(file.read_text(encoding='utf-8'),str(file),'exec')
    def test_21_implementation_tests(self):
        q=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'tools/test_patch_region_learning_20261105.py')],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(q.returncode,0,q.stdout+q.stderr);self.assertIn('Ran 36 tests',q.stderr)
    def test_22_no_target_training(self):
        self.assertFalse(p.configuration()['target_labels_for_training'])
        self.assertFalse(p.configuration()['source_TEST'])
        self.assertFalse(p.configuration()['new_CLIP_training'])
        self.assertTrue(p.configuration()['classification_only_deferred'])
    def test_23_transform_fails_closed(self):
        with self.assertRaises(AssertionError):runner(original.replace('from model_cuda import ComponentDelta,objective','changed'))
    def test_24_preflight_retained_zero(self):
        self.assertIn('disposable_optimizer_updates=6,new_retained_updates=0',new)
        self.assertEqual(p.configuration()['prior_implementation_retained_updates'],0)


if __name__=='__main__':unittest.main(verbosity=2)
