"""Standard-library synthetic algebra/AST checks, NOT neural or CUDA tests."""
import ast
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import random
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];MOD=ROOT/'code/dual_stream_20261026/paper_components_20261104'
spec=importlib.util.spec_from_file_location('components_test_plan',MOD/'plan.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class ComponentsTests(unittest.TestCase):
    def test_01_seeds(self):self.assertEqual(p.SEEDS,(2027,3407,4099))
    def test_02_arms(self):self.assertEqual(p.NEW_ARMS,('UniformNoAux','UniformBoundary','RGBBoundary'))
    def test_03_updates(self):self.assertEqual(p.configuration()['retained_updates'],3*3*10*719)
    def test_04_training_frames(self):self.assertEqual(p.configuration()['retained_training_frame_forwards'],64710*16)
    def test_05_VAL_budget(self):self.assertEqual(p.configuration()['source_VAL_frame_forwards'],9*10*22080)
    def test_06_target_denominators(self):
        self.assertEqual(sum(sum(x) for x in p.COUNTS),6973)
        self.assertEqual(32*6973,223136)
    def test_07_target_budget(self):self.assertEqual(p.configuration()['full_target_residual_forwards'],9*223136)
    def test_08_localization_budget(self):self.assertEqual(p.configuration()['selected_localization_forwards'],6*22080)
    def test_09_probe_budgets(self):
        self.assertEqual(len(p.PROBES),64)
        self.assertEqual(p.configuration()['target_preflight_forwards_per_invocation'],9*64+7*9*32)
    def test_10_scope(self):
        c=p.configuration()
        for k in ('source_TEST','target_masks','new_CLIP_training','target_selection','seed_search','coefficient_search','deployment','literature_CLIP_in_experiment'):self.assertFalse(c[k],k)
        self.assertTrue(c['automation_paused'] and c['old_tasks_paused'] and c['no_new_monitor'])
    def test_11_one_variable_supervision(self):
        a,b=p.variant('UniformNoAux'),p.variant('UniformBoundary')
        self.assertEqual([k for k in a if a[k]!=b[k]],['boundary_coefficient'])
    def test_12_RGB_only_input_to_full(self):
        self.assertEqual(p.variant('RGBBoundary'),dict(input='RGB',pooling='predicted_boundary_cells_quarter_floor',boundary_coefficient=.2))
    def test_13_sampling_balance(self):
        rows=[dict(method=m) for m in p.METHODS for _ in range(4)]
        schedule=p.sampling(rows,1,2027);self.assertEqual(len(schedule),719)
        for batch in schedule:
            self.assertEqual(len(batch),16)
            self.assertEqual([sum(rows[i]['method']==m for i in batch) for m in p.METHODS],[8,2,2,2,2])
    def test_14_sampling_deterministic(self):
        rows=[dict(method=m) for m in p.METHODS for _ in range(4)]
        a=p.fingerprint(p.sampling(rows,1,2027));self.assertEqual(a,p.fingerprint(p.sampling(rows,1,2027)))
        self.assertNotEqual(a,p.fingerprint(p.sampling(rows,2,2027)))
        self.assertNotEqual(a,p.fingerprint(p.sampling(rows,1,3407)))
    def test_15_sampling_matches_frozen_original(self):
        original=ROOT/'code/dual_stream_20261026/legacy_delta_full_fix2_20261104/plan.py'
        spec=importlib.util.spec_from_file_location('old_components_test_plan',original);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        rows=[dict(method=m) for m in p.METHODS for _ in range(7)]
        for s in p.SEEDS:self.assertEqual(p.sampling(rows,4,s),old.sampling(rows,4,s))
    def test_16_auc_direction(self):
        self.assertEqual(p.auc([0,1],[0.,1.]),Fraction(1))
        self.assertEqual(p.auc([0,1],[1.,0.]),Fraction(0))
    def test_17_auc_ties(self):self.assertEqual(p.auc([0,1],[2.,2.]),Fraction(1,2))
    def test_18_auc_independent_pair_reference(self):
        rng=random.Random(13)
        for _ in range(30):
            y=[0]*5+[1]*7;z=[rng.randrange(4) for _ in y]
            pair=Fraction(sum(2*(a>b)+(a==b) for a in z[5:] for b in z[:5]),70)
            self.assertEqual(p.auc(y,z),pair)
    def test_19_nonfinite_rejected(self):
        for x in (math.inf,math.nan):
            with self.assertRaises(AssertionError):p.auc([0,1],[0.,x])
    def test_20_earliest_tie(self):
        records=[dict(epoch=e,metrics=dict(auc_fraction=[3,4])) for e in range(1,11)]
        self.assertEqual(p.select(records)['epoch'],1)
        records[4]['metrics']['auc_fraction']=[4,5];self.assertEqual(p.select(records)['epoch'],5)
    def test_21_epoch_coverage(self):
        with self.assertRaises(AssertionError):p.select([dict(epoch=1,metrics=dict(auc_fraction=[1,2]))])
    def test_22_exact_aggregation(self):
        rows=[dict(video_id='v',family_id='f',label=0,method='real') for _ in range(32)];z=[-3.,1.]*16
        expected=0.
        for x in z:expected+=p.sigmoid(x)
        score=p.videos(rows,z)[0]['score'];self.assertEqual(score,expected/32)
        self.assertNotEqual(score,p.sigmoid(sum(z)/32))
    def test_23_incomplete_video_rejected(self):
        with self.assertRaises(AssertionError):p.videos([dict(video_id='v',family_id='f',label=0)],[0.])
    def test_24_f32_fusion(self):
        self.assertEqual(p.f32(p.f32(.1/2)+.01),p.f32(p.f32(.1/2)+p.f32(.01)))
        self.assertTrue(math.isfinite(p.f32(1.)))
    def test_25_compile_all(self):
        for file in [*MOD.glob('*.py'),ROOT/'tools/manage_paper_components_20261104.py',ROOT/'tools/verify_paper_components_20261104.py']:
            ast.parse(file.read_text(encoding='utf-8'),filename=str(file));compile(file.read_text(encoding='utf-8'),str(file),'exec')
    def test_26_no_GT_forward(self):
        tree=ast.parse((MOD/'model_cuda.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward')
        self.assertEqual([a.arg for a in node.args.args],['self','rgb','baseline','enabled'])
    def test_27_no_untrained_attention_control(self):
        code=(MOD/'model_cuda.py').read_text();self.assertIn('pooled=contrast.mean((2,3))',code)
        self.assertIn("if self.variant=='RGBBoundary':return super().forward",code)
    def test_28_valid_masks_only(self):
        code=(MOD/'model_cuda.py').read_text();self.assertIn("out['boundary_logit'][valid]",code);self.assertIn('masks[valid]',code)
        self.assertIn("aux=out['boundary_logit'].sum()*0",code)
    def test_29_frozen_baseline(self):self.assertIn('baseline.detach()+delta',(MOD/'model_cuda.py').read_text())
    def test_30_all_loads_CUDA(self):
        tree=ast.parse((MOD/'run_cuda.py').read_text())
        loads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch']
        self.assertTrue(loads)
        for n in loads:self.assertTrue(any(k.arg=='map_location' and isinstance(k.value,ast.Constant) and k.value.value=='cuda' for k in n.keywords))
    def test_31_stage_gates(self):
        code=(MOD/'run_cuda.py').read_text();self.assertIn("exited(run,'SOURCE_PREFLIGHT',registration)",code)
        self.assertIn("proof=read(run/'SOURCE_LOCAL_VERIFICATION.json')",code)
        self.assertIn("exited(run,'TARGET_PREFLIGHT',registration)",code)
    def test_32_guard_and_no_retries(self):
        code=(MOD/'run_cuda.py').read_text();self.assertIn('LOCK_EX|fcntl.LOCK_NB',code);self.assertIn('OwnedCudaGuard',code)
        self.assertEqual(p.configuration()['automatic_retries'],0)
    def test_33_independent_video_replay(self):
        from verify_paper_components_20261104 import replay_videos
        actual=[dict(video_id='a',family_id='fa',label=0,method='real',frames=32,score=.2),dict(video_id='b',family_id='fb',label=1,method='fake',frames=32,score=.8)]
        saved=[dict(r) for r in actual];saved[0]['score']=math.nextafter(.2,1.)
        self.assertLessEqual(replay_videos(actual,saved)['max_ULPs'],1.)
        saved[0]['score']=.2+8*math.ulp(.2)
        with self.assertRaises(AssertionError):replay_videos(actual,saved)
    def test_34_replay_preserves_ties(self):
        from verify_paper_components_20261104 import replay_videos
        actual=[dict(video_id='a',family_id='fa',label=0,frames=32,score=.5),dict(video_id='b',family_id='fb',label=1,frames=32,score=.5)]
        saved=[dict(r) for r in actual];saved[0]['score']=math.nextafter(.5,0.)
        with self.assertRaises(AssertionError):replay_videos(actual,saved)
    def test_35_no_local_neural_import(self):
        for file in ('manage_paper_components_20261104.py','verify_paper_components_20261104.py'):
            tree=ast.parse((ROOT/'tools'/file).read_text())
            modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            modules += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
            self.assertFalse(any((m or '').split('.')[0] in ('torch','numpy','PIL','transformers') for m in modules))
    def test_36_complete_preflight_budget(self):
        c=p.configuration()
        self.assertEqual(c['source_preflight_synthetic_forwards'],9*(2+2+4))
        self.assertEqual(c['source_preflight_total_forwards'],192+72)
        self.assertIn('preflight_frame_forwards=264',(MOD/'run_cuda.py').read_text())
    def test_37_input_bytes_checked(self):
        spec=importlib.util.spec_from_file_location('checked_pixels_test',MOD/'checked_pixels.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        import hashlib
        with tempfile.TemporaryDirectory(prefix='paper_components_integrity_') as directory:
            file=Path(directory)/'fixture.bin';file.write_bytes(b'isolated-synthetic-input')
            self.assertEqual(module.checked_bytes(file,hashlib.sha256(file.read_bytes()).hexdigest()),b'isolated-synthetic-input')
            with self.assertRaises(AssertionError):module.checked_bytes(file,'0'*64)
    def test_38_each_source_read_checked(self):
        code=(MOD/'checked_pixels.py').read_text();tree=ast.parse(code)
        for name in ('pixels','mask_pixels'):
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checked_bytes' for n in ast.walk(node)))
        self.assertIn('from checked_pixels import pixels,mask_pixels',(MOD/'run_cuda.py').read_text())
    def test_39_reuse_SHA_gate(self):
        code=(ROOT/'tools/manage_paper_components_20261104.py').read_text()
        self.assertIn("sha(src)==entry['sha256']",code)
        self.assertIn("sha(TARGET/'SAVED_RESULTS.tar.gz')==archive_proof['sha256']",code)
    def test_40_reporting_covers_all_six(self):
        from verify_paper_components_20261104 import GROUPS,CONTRASTS
        self.assertEqual(len(GROUPS),6);self.assertEqual(len(CONTRASTS),6)
        self.assertIn(('SemanticHalf','SemanticRaw'),CONTRASTS)
        code=(ROOT/'tools/verify_paper_components_20261104.py').read_text()
        for key in ('source_six_groups','target85_met','source95_met','paired_seed_macro_delta_pp','eligible_mask_coverage','costs'):
            self.assertIn(key,code)
    def test_41_reused_source_fusion_identity(self):
        from verify_paper_components_20261104 import table,correction,inspect
        import manage_paper_components_20261104 as manager
        val=manager.read(manager.SOURCE/'run/inputs/source_val.json')
        sem=table(manager.SOURCE/'run/semantic_source/VAL.csv',22080)
        selected=manager.read(manager.SELECT/'run/SELECTED_ENDPOINT_LOCK.json')
        for seed in p.SEEDS:
            item=selected[str(seed)];base=[p.f32(float(r['s'+str(seed)])/2) for r in sem]
            file=manager.SELECT/f'run/evaluation/{seed}/epoch_{item["epoch"]:02}/FRAMES.csv'
            logits=correction(table(file,22080),base);metric,_=inspect(val,logits)
            self.assertEqual(Fraction(*metric['auc_fraction']),Fraction(item['auc_numerator'],item['auc_denominator']))
            file=manager.SOURCE/f'run/seeds/{seed}/FRAMES.csv'
            self.assertEqual(len(correction(table(file,22080),base)),22080)

if __name__=='__main__':unittest.main(verbosity=2)
