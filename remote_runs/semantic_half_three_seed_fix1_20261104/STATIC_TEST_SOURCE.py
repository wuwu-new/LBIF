"""Static contracts and isolated synthetic score tests. Not a CUDA/effectiveness test."""
import ast
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];MOD=ROOT/'code/dual_stream_20261026/semantic_half_three_seed_fix1_20261104'
sys.path.insert(0,str(MOD))
import confirmation_plan as q
import analyze_semantic_half_three_seed_fix1_20261104 as producer
import verify_semantic_half_three_seed_fix1_20261104 as independent


def fixture():
    rng=random.Random(17);rows=[];z=[]
    for video in range(12):
        for frame in range(32):
            rows.append(dict(video_id=f'v{video}',family_id=f'f{video//2}',label=video%2,frame_index=frame))
            z.append(q.f32(rng.uniform(-4,4)+.2*(video%2)))
    return rows,z


class Contracts(unittest.TestCase):
    def test_01_fixed_seeds(self):self.assertEqual(q.SEEDS,(2027,3407,4099));self.assertEqual(q.NEW_SEEDS,(3407,4099))
    def test_02_exact_budget(self):
        c=q.configuration();self.assertEqual(c['new_retained_updates'],2*719*10);self.assertEqual(c['source_total_disposable_updates'],8)
    def test_03_source_forwards(self):
        c=q.configuration();self.assertEqual(c['retained_training_frame_forwards'],2*10*2*11492);self.assertEqual(c['source_VAL_frame_forwards'],2*11*22080);self.assertEqual(c['checkpoint_replay_frame_forwards'],2*11*16)
    def test_04_preflight_budget(self):
        c=q.configuration();self.assertEqual(c['source_preflight_predict_forwards_per_invocation'],64+2*4*16);self.assertEqual(c['target_preflight_predict_forwards_per_invocation'],7*3*16)
    def test_05_target_denominators(self):self.assertEqual(sum(a+b for a,b in q.COUNTS),6973);self.assertEqual(sum(32*(a+b) for a,b in q.COUNTS),223136)
    def test_06_encoder_budget(self):
        c=q.configuration();self.assertEqual(c['source_fresh_CLIP_frames'],2*11492+2*32);self.assertEqual(c['target_fresh_CLIP_frames'],223136+2*112)
    def test_07_no_search(self):
        c=q.configuration();self.assertFalse(any(c[k] for k in ('source_TEST','target_masks','target_labels_for_training','seed_search','epoch_search','coefficient_search','automatic_extra_seeds')))
    def test_08_all_python_parses(self):
        paths=[*MOD.glob('*.py'),ROOT/'tools/manage_semantic_half_three_seed_fix1_20261104.py',Path(producer.__file__),Path(independent.__file__)]
        for p in paths:ast.parse(p.read_text(encoding='utf-8'),str(p))
    def test_09_no_local_neural_import(self):
        for p in (Path(producer.__file__),Path(independent.__file__),ROOT/'tools/manage_semantic_half_three_seed_fix1_20261104.py',MOD/'confirmation_plan.py'):
            for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
                if isinstance(n,ast.Import):self.assertFalse(any(a.name.split('.')[0] in ('torch','numpy','PIL') for a in n.names))
                if isinstance(n,ast.ImportFrom):self.assertNotIn((n.module or '').split('.')[0],('torch','numpy','PIL'))
    def test_10_verifier_not_producer_import(self):
        text=Path(independent.__file__).read_text(encoding='utf-8');self.assertNotIn('import analyze_',text);self.assertNotIn('import confirmation_plan',text)
    def test_11_readonly_and_gradient_guards(self):
        text=(MOD/'run_source_cuda.py').read_text(encoding='utf-8')
        for s in ('cache_checks[str(seed)]==cache_hashes','frozen_anchor_sha','all(t.grad is None','error_if_nonfinite=True','with isolated_rng():replay=make(seed)'):self.assertIn(s,text)
    def test_12_full_training_not_half(self):
        text=(MOD/'run_source_cuda.py').read_text(encoding='utf-8')
        for s in ("F.binary_cross_entropy_with_logits(result['logit'],labels[ids])",".05*F.mse_loss(outputs[0]['delta'],outputs[1]['delta'])",'seed+10000*epoch','for epoch in range(11)'):self.assertIn(s,text)
        self.assertNotIn("result['logit']/2",text)
    def test_13_cuda_load_only(self):
        for p in (MOD/'common_cuda.py',MOD/'run_source_cuda.py'):
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='load':
                    self.assertTrue(any(k.arg=='map_location' and isinstance(k.value,ast.Constant) and k.value.value=='cuda' for k in n.keywords))
    def test_14_target_no_updates(self):
        text=(MOD/'run_target_cuda.py').read_text(encoding='utf-8');self.assertNotIn('optimizer.step',text);self.assertNotIn('.backward(',text)
        for s in ('models[2027]','del models[2027]','pixel_SHA_checked','SOURCE_LOCAL_VERIFICATION.json','TARGET_PREFLIGHT.json'):self.assertIn(s,text)
    def test_15_exclusive_no_overwrite(self):
        for p in (MOD/'run_source_cuda.py',MOD/'run_target_cuda.py'):
            text=p.read_text(encoding='utf-8');self.assertIn('LOCK_EX|fcntl.LOCK_NB',text);self.assertIn('OwnedCudaGuard',text)
        text=(ROOT/'tools/manage_semantic_half_three_seed_fix1_20261104.py').read_text(encoding='utf-8');self.assertIn('Existing GPU user, no queue',text);self.assertIn('remote_checkpoints_only',text)
    def test_16_fp32_half(self):
        for x in (0.,-0.,1/3,1e-8,100.,-100.):self.assertEqual(q.f32(q.f32(x)/2),independent.fp(independent.fp(x)/2))
    def test_17_auc_known_and_ties(self):
        for y,z,want in (([0,1],[0.,1.],1.),([0,1],[1.,0.],0.),([0,1],[1.,1.],.5),([0,0,1,1],[0.,1.,1.,2.],.875)):
            self.assertEqual(producer.auc(y,z),want);self.assertEqual(independent.auc(y,z),want)
    def test_18_auc_one_class(self):self.assertIsNone(producer.auc([1,1],[0,1]));self.assertIsNone(independent.auc([1,1],[0,1]))
    def test_19_full_synthetic_replay(self):
        rows,z=fixture();a,b=producer.aggregate(rows,z);c,d=independent.replay(rows,z);independent.close(a,c,1e-12);independent.close(b,d['frame_metrics'])
        self.assertEqual(len(a),12);self.assertEqual(sum(r['label'] for r in a),6)
    def test_20_duplicate_rejected(self):
        rows,z=fixture();rows[1]['frame_index']=0
        with self.assertRaises(AssertionError):producer.aggregate(rows,z)
        with self.assertRaises(AssertionError):independent.replay(rows,z)
    def test_21_missing_frame_rejected(self):
        rows,z=fixture()
        with self.assertRaises(AssertionError):producer.aggregate(rows[:-1],z[:-1])
        with self.assertRaises(AssertionError):independent.replay(rows[:-1],z[:-1])
    def test_22_family_materialization_equivalence(self):
        rows,z=fixture();v,_=producer.aggregate(rows,z);a=producer.draw_values(v,0);b=independent.bootstrap(v,0);independent.close(a,b)
    def test_23_rank_classification_repair(self):
        rows,z=fixture();v,frame=producer.aggregate(rows,z);a={k:producer.auc([r['label'] for r in v],[r['scores'][k] for r in v]) for k in producer.KEYS};draws=producer.draw_values(v,1)
        p=producer.paired(v,a,draws);s=independent.paired(v,a,draws);independent.close(p,s);self.assertAlmostEqual(p['rank_repair_pp']-p['rank_damage_pp'],p['delta_pp'])
    def test_24_sample_sd(self):
        for values in ([.8,.9],[.8,.9,1.]):independent.close(producer.mean_SD(values),independent.mean_sd(values));self.assertEqual(producer.mean_SD(values)['sample_SD'],statistics.stdev(values))
    def test_25_csv_fp32_and_indices(self):
        with tempfile.TemporaryDirectory(prefix='semantic_scores_synthetic_') as folder:
            p=Path(folder)/'scores.csv'
            with p.open('w',newline='') as f:csv.writer(f).writerows([['index','logit'],[0,q.f32(1/3)],[1,2.]])
            self.assertEqual(producer.logits(p,['index','logit']),independent.frames(p,['index','logit']))
            with p.open('w',newline='') as f:csv.writer(f).writerows([['index','logit'],[1,2.]])
            with self.assertRaises(AssertionError):producer.logits(p,['index','logit'])
            with self.assertRaises(AssertionError):independent.frames(p,['index','logit'])
    def test_26_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(prefix='semantic_scores_synthetic_') as folder:
            p=Path(folder)/'scores.csv'
            with p.open('w',newline='') as f:csv.writer(f).writerows([['index','logit'],[0,'nan']])
            with self.assertRaises(AssertionError):producer.logits(p,['index','logit'])
            with self.assertRaises(AssertionError):independent.frames(p,['index','logit'])
    def test_27_historical_comparisons(self):
        metrics={d:dict(seeds={str(s):dict(video_AUC={'Semantic':.8,'SemanticHalf':.9}) for s in q.SEEDS},historical_video_AUC={k:.85 for k in producer.HISTORICAL}) for d in q.DOMAINS}
        r=producer.comparison_summary(metrics,3407,('SameSeedOriginal',*producer.HISTORICAL));self.assertEqual(r['V18']['improved_domains'],7);self.assertAlmostEqual(r['V18']['macro_delta_pp'],5.)
    def test_28_real_inputs_and_old_identity(self):
        old=ROOT/'experiments/guarded_transfer_20261026/iteration_02_crosslayer_fix1_20261102/remote_bundle'
        train=json.loads((old/'inputs/fit.json').read_text(encoding='utf-8'));val=json.loads((old/'inputs/source_val.json').read_text(encoding='utf-8'))
        self.assertEqual((len(train),len(val)),(11492,22080));self.assertFalse({r['family_id'] for r in train}&{r['family_id'] for r in val})
        self.assertEqual(len({r['video_id'] for r in train}),2873)
    def test_29_no_neural_modules_loaded(self):self.assertFalse(any(m in sys.modules for m in ('torch','numpy','PIL')))
    def test_30_real_old_source_scores(self):
        base=ROOT/'experiments/guarded_transfer_20261026';old=base/'iteration_02_crosslayer_fix1_20261102/remote_bundle'
        rows=json.loads((old/'inputs/source_val.json').read_text(encoding='utf-8'))
        path=base/'ffpp_boundary_readout_paired_20261103/final_20261103T084508316349Z/run/SEMANTIC_VAL.csv'
        from manage_semantic_half_three_seed_fix1_20261104 import mapped_old_source
        z,mapping,prior=mapped_old_source();self.assertEqual(len(mapping),22080);v,m=independent.replay(rows,z)
        self.assertEqual(m['video_AUC']['Semantic'],.9805975635370721);self.assertEqual(m['video_AUC']['SemanticHalf'],.9810964083175804)
        p,frame=producer.aggregate(rows,z);independent.close(v,p,1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
