"""Stdlib-only contract, transformation and source-preservation tests."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from build_latest_dual_ablation_20261105 import model,patch_config,runner

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('latest_ablation_test_plan',ROOT/'code/dual_stream_20261026/latest_dual_ablation_20261105/plan.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
PARENT=ROOT/'experiments/guarded_transfer_20261026'/p.PARENT
original=PARENT/'bundle/implementation'
old_runner=(original/'run_cuda.py').read_text(encoding='utf-8')
new_runner=runner(old_runner)
old_model=(original/'model_cuda.py').read_text(encoding='utf-8')
new_model=model(old_model)


class AblationTests(unittest.TestCase):
    def setUp(self):p.configure('first')
    def test_01_five_only(self):self.assertEqual(len(p.NEW_ARMS),5);self.assertNotIn('PatchClassificationContrast',p.NEW_ARMS)
    def test_02_first_seed(self):self.assertEqual(p.SEEDS,(2027,))
    def test_03_remaining_seeds(self):p.configure('remaining');self.assertEqual(p.SEEDS,(3407,4099))
    def test_04_total_budget(self):
        first=p.configuration()['retained_updates'];p.configure('remaining')
        self.assertEqual(first+p.configuration()['retained_updates'],107850)
    def test_05_first_budget(self):
        c=p.configuration();self.assertEqual(c['retained_updates'],35950);self.assertEqual(c['retained_training_frame_forwards'],575200)
    def test_06_first_VAL(self):self.assertEqual(p.configuration()['source_VAL_frame_forwards'],1104000)
    def test_07_first_target(self):self.assertEqual(p.configuration()['full_target_residual_forwards'],1115680)
    def test_08_first_preflight(self):
        c=p.configuration();self.assertEqual(c['source_preflight_total_forwards'],168);self.assertEqual(c['preflight_disposable_updates'],10)
    def test_09_target_probes(self):self.assertEqual(p.configuration()['target_preflight_forwards_per_invocation'],1440)
    def test_10_RGB_only_input(self):
        a=p.variant('RGBFull');b=p.variant('PatchClassificationContrast');a['input']=b['input'];self.assertEqual(a,b)
    def test_11_uniform_only_pool(self):
        a=p.variant('UniformPoolFull');b=p.variant('PatchClassificationContrast');a['pooling']=b['pooling'];self.assertEqual(a,b)
    def test_12_no_classification(self):
        a=p.variant('NoPatchClassification');self.assertEqual(a['patch_classification'],0);self.assertEqual(a['patch_contrast'],.1)
    def test_13_no_contrast(self):
        a=p.variant('NoRegionContrast');self.assertEqual(a['patch_classification'],.2);self.assertEqual(a['patch_contrast'],0)
    def test_14_neither(self):
        a=p.variant('NoPatchLearningMatched');self.assertEqual((a['patch_classification'],a['patch_contrast']),(0,0))
    def test_15_boundary_all_retained(self):self.assertTrue(all(p.variant(a)['boundary_coefficient']==.2 for a in p.NEW_ARMS))
    def test_16_semantic_not_modified(self):
        self.assertIn("p.f32(float(r['s'+str(s)])/2)",new_runner)
        self.assertIn("bb=torch.from_numpy(bases[ids]).cuda()",new_runner)
        self.assertFalse(p.configuration()['new_CLIP_training'])
    def test_17_sampling_identical(self):
        rows=[dict(method=m) for m in p.METHODS for _ in range(7)]
        for e in range(1,11):self.assertEqual(p.sampling(rows,e,2027),p.old.sampling(rows,e,2027))
    def test_18_constructor_unchanged(self):
        def constructor(t):return ast.dump(next(n for n in ast.walk(ast.parse(t)) if isinstance(n,ast.FunctionDef) and n.name=='__init__'))
        self.assertEqual(constructor(old_model),constructor(new_model))
    def test_19_objectives_unchanged(self):
        for name in ('objective','patch_targets','patch_classification_loss','patch_contrastive_loss'):
            def fn(t):return ast.dump(next(n for n in ast.walk(ast.parse(t)) if isinstance(n,ast.FunctionDef) and n.name==name))
            self.assertEqual(fn(old_model),fn(new_model))
    def test_20_fusion_unchanged(self):self.assertIn('out = dict(logit=baseline.detach()+delta',new_model);self.assertIn('.5*self.delta_head(pooled)',new_model)
    def test_21_explicit_input(self):self.assertIn("front = rgb if cfg.variant(self.patch_variant)['input'] == 'RGB' else self.error_map(rgb)",new_model)
    def test_22_uniform_detached(self):self.assertIn("torch.ones_like(f2[:, :1])",new_model);self.assertIn('boundary.detach().sigmoid()',new_model)
    def test_23_latest_full_replay(self):self.assertIn('current_full_errors',new_runner);self.assertIn("assert saved['registration_sha256']==p.PARENT_REG",new_runner)
    def test_24_initial_state_exact(self):self.assertIn("assert digest==latest_initial[str(seed)]['PatchClassificationContrast']",new_runner)
    def test_25_disabled_gradient_checked(self):self.assertIn("assert bool(total_patch.abs().sum()>0)==bool(weights['patch_classification'])",new_runner)
    def test_26_no_target_gradient(self):
        self.assertFalse(p.configuration()['target_labels_for_training']);self.assertFalse(p.configuration()['source_TEST'])
    def test_27_guard_preserved(self):
        for s in ('OwnedCudaGuard','LOCK_EX|fcntl.LOCK_NB',"exited(run,'SOURCE_PREFLIGHT',registration)","SOURCE_LOCAL_VERIFICATION.json",'assert maximum<=2e-7'):
            self.assertIn(s,new_runner)
    def test_28_CUDA_loads(self):
        for n in ast.walk(ast.parse(new_runner)):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='load':
                self.assertTrue(any(k.arg=='map_location' and isinstance(k.value,ast.Constant) and k.value.value=='cuda' for k in n.keywords))
    def test_29_frozen_parent(self):
        lock=json.loads((PARENT/'REGISTRATION.json').read_text(encoding='utf-8'));self.assertEqual(lock['registration_sha256'],p.PARENT_REG)
        for name,digest in lock['files'].items():self.assertEqual(hashlib.sha256((PARENT/'bundle'/name).read_bytes()).hexdigest(),digest,name)
    def test_30_transform_fails_closed(self):
        with self.assertRaises(AssertionError):model(old_model.replace('f0 = self.encoder[0](self.error_map(rgb))','changed'))
        with self.assertRaises(AssertionError):runner(old_runner.replace("contract['disposable_optimizer_updates']==6",'changed'))
    def test_31_compile(self):
        for text in (new_model,new_runner,patch_config((original/'patch_config.py').read_text(encoding='utf-8'))):compile(text,'generated','exec')
        for name in ('manage_latest_dual_ablation_20261105.py','verify_latest_dual_ablation_20261105.py'):
            compile((ROOT/'tools'/name).read_text(encoding='utf-8'),name,'exec')
    def test_32_local_stdlib_only(self):
        for name in ('manage_latest_dual_ablation_20261105.py','verify_latest_dual_ablation_20261105.py','build_latest_dual_ablation_20261105.py'):
            tree=ast.parse((ROOT/'tools'/name).read_text(encoding='utf-8'))
            modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            modules +=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
            self.assertFalse(any((s or '').split('.')[0] in ('torch','numpy','PIL','transformers') for s in modules))
    def test_33_protocol_targets(self):self.assertEqual(sum(32*(a+b) for a,b in p.COUNTS),223136)
    def test_34_no_optional_expansion(self):self.assertFalse(p.configuration()['optional_boundary_loss_ablation_authorized']);self.assertEqual(p.configuration()['automatic_retries'],0)
    def test_35_selection(self):
        rows=[dict(epoch=i,metrics=dict(auc_fraction=[1,2])) for i in range(1,11)]
        self.assertEqual(p.select(rows)['epoch'],1)
    def test_36_disabled_head_not_claimed_capacity(self):self.assertIn('zero coefficient',p.configuration()['disabled_auxiliary_head'])


if __name__=='__main__':unittest.main(verbosity=2)
