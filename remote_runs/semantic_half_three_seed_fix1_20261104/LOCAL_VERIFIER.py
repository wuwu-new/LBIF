"""Independent stdlib replay; never imports the producer, runtime, or neural libraries."""
import argparse
import bisect
import collections
import csv
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/guarded_transfer_20261026'
LOCAL=BASE/'best_four_three_seed_confirmation_fix1_20261104'
HISTORY=BASE/'iteration_03_sequence_target_20261102/final_20261102T113604Z/run'
SEEDS=(2027,3407,4099);NEW=(3407,4099);KEYS=('Semantic','SemanticHalf')
DOMAINS=('cdf1','cdf2','dfd','dfdc','dfdcp','wild','uadfv')
COUNTS=((32,57),(156,326),(138,1692),(1538,1800),(107,232),(396,410),(44,45))
HISTORICAL=('LegacySemantic','OldV2','V18','V22FixedFour','OrderedCLIP')
PARENT_REG='fd3b984647210e456cb0000291cd76d592a7e04e7c7c84421f91e5c862571b5f'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def fp(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def probability(x):return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
def auc(y,s):
    negative=sorted(z for label,z in zip(y,s) if label==0);positive=[z for label,z in zip(y,s) if label==1]
    if not negative or not positive:return None
    return math.fsum((bisect.bisect_left(negative,z)+bisect.bisect_right(negative,z))/2 for z in positive)/(len(positive)*len(negative))
def close(a,b,tolerance=1e-10):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a:close(a[k],b[k],tolerance)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y,tolerance)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert math.isfinite(b) and abs(a-b)<=tolerance,(a,b)
    else:assert a==b,(a,b)
def mean_sd(values):return dict(mean=statistics.mean(values),sample_SD=statistics.stdev(values),n=len(values))
def frames(path,columns):
    with Path(path).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==columns
        out=[]
        for i,row in enumerate(reader):
            assert int(row['index'])==i
            values=[float(row[k]) for k in columns[1:]]
            assert all(math.isfinite(v) and fp(v)==v for v in values)
            out.append(values)
    return out
def replay(rows,z):
    assert len(rows)==len(z);groups=collections.OrderedDict();seen=set();frame_y=[]
    for row,value in zip(rows,z):
        assert row['label'] in (0,1)
        ident=(row['video_id'],row['frame_index']);assert ident not in seen;seen.add(ident)
        g=groups.setdefault(row['video_id'],dict(video_id=row['video_id'],family_id=row['family_id'],label=row['label'],frames=0,scores={k:[] for k in KEYS},indices=[]))
        assert (g['family_id'],g['label'])==(row['family_id'],row['label']);g['frames']+=1;g['indices'].append(row['frame_index']);frame_y.append(row['label'])
        g['scores']['Semantic'].append(probability(value));g['scores']['SemanticHalf'].append(probability(fp(value/2)))
    videos=[]
    for g in groups.values():
        assert g['frames']==32 and sorted(g.pop('indices'))==list(range(32))
        g['scores']={k:math.fsum(v)/32 for k,v in g['scores'].items()};videos.append(g)
    frame={}
    for k,values in zip(KEYS,(z,[fp(x/2) for x in z])):
        frame[k]=dict(AUC=auc(frame_y,values),FP=sum(y==0 and a>=0 for y,a in zip(frame_y,values)),FN=sum(y==1 and a<0 for y,a in zip(frame_y,values)))
    y=[r['label'] for r in videos]
    metric=dict(real_videos=len(y)-sum(y),fake_videos=sum(y),real_frames=len(frame_y)-sum(frame_y),fake_frames=sum(frame_y),families=len({r['family_id'] for r in videos}),
        video_AUC={k:auc(y,[r['scores'][k] for r in videos]) for k in KEYS},frame_metrics=frame,
        confusion={k:dict(FP=sum(r['label']==0 and r['scores'][k]>=.5 for r in videos),FN=sum(r['label']==1 and r['scores'][k]<.5 for r in videos)) for k in KEYS})
    return videos,metric
def bootstrap(videos,index):
    families=sorted({r['family_id'] for r in videos});members=collections.defaultdict(list)
    for i,r in enumerate(videos):members[r['family_id']].append(i)
    rng=random.Random(20271104+index);samples=[]
    for _ in range(1000):
        ids=[i for _ in families for i in members[rng.choice(families)]];labels=[videos[i]['label'] for i in ids]
        a=auc(labels,[videos[i]['scores']['Semantic'] for i in ids]);b=auc(labels,[videos[i]['scores']['SemanticHalf'] for i in ids])
        samples.append(None if a is None else 100*(b-a))
    return samples
def paired(videos,aa,draws):
    good=bad=0.
    for p in videos:
        if not p['label']:continue
        for n in videos:
            if n['label']:continue
            h=(p['scores']['SemanticHalf']>n['scores']['SemanticHalf'])+.5*(p['scores']['SemanticHalf']==n['scores']['SemanticHalf'])
            o=(p['scores']['Semantic']>n['scores']['Semantic'])+.5*(p['scores']['Semantic']==n['scores']['Semantic'])
            if h>o:good+=h-o
            else:bad+=o-h
    positive=sum(r['label'] for r in videos);scale=100/(positive*(len(videos)-positive));sample=sorted(x for x in draws if x is not None)
    changes=[((r['scores']['Semantic']>=.5)==bool(r['label']),(r['scores']['SemanticHalf']>=.5)==bool(r['label'])) for r in videos]
    return dict(delta_pp=100*(aa['SemanticHalf']-aa['Semantic']),rank_repair_pp=good*scale,rank_damage_pp=bad*scale,
        repaired_videos=sum(not a and b for a,b in changes),damaged_videos=sum(a and not b for a,b in changes),
        family_descriptive_95pct=[sample[int(.025*(len(sample)-1))],sample[int(.975*(len(sample)-1))]],valid_bootstrap=len(sample))
def inspect(stage):
    final=ROOT/read(LOCAL/(stage+'_ARCHIVE_RECEIPT.json'))['path'];run=final/'run';out=final/'analysis';inventory=read(final/'REMOTE_INVENTORY.json');lock=read(run/'REGISTRATION.json')
    assert inventory['process_exit_verified'] and sha(final/'run.tar.gz')==inventory['sha256']
    assert fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})==lock['registration_sha256']
    for name,item in inventory['files'].items():assert sha(run/name)==item['sha256'] and (run/name).stat().st_size==item['bytes']
    for name,value in lock['files'].items():assert sha(run/name)==value
    for name,value in lock['local_evidence_sha256'].items():assert sha(ROOT/name)==value
    for name,value in read(out/'COMPLETE.json')['files'].items():assert sha(out/name)==value
    assert sha(BASE/'iterative_crossdomain_10rounds_20261102/ROUNDS.jsonl')==lock['ledger_sha256']
    c=lock['configuration']
    assert c['winner']=='SemanticHalf' and c['seeds']==list(SEEDS) and c['new_training_seeds']==list(NEW)
    assert (c['epochs'],c['updates_per_epoch'],c['new_retained_updates'],c['source_total_disposable_updates'])==(10,719,14380,8)
    assert (c['coefficient'],c['threshold'],c['bootstrap'])==(.5,.5,1000)
    assert not any(c[k] for k in ('source_TEST','target_masks','target_labels_for_training','coefficient_search','seed_search','epoch_search','automatic_extra_seeds'))
    assert not list(run.glob('*_FAILURE.json'))
    assert read(run/(stage+'_COMPLETE.json'))['report_sha256']==sha(run/(stage+'_REPORT.json'))
    return final,run,out,lock
def source_runtime(run,reg,inventory):
    initial=read(run/'TRAIN_CACHE_IDENTITY.json')['sha256'];initializations=[];sampling={}
    for name in ('SOURCE_PREFLIGHT.json','SOURCE_RUNTIME_PREFLIGHT.json'):
        p=read(run/name);assert p['passed'] and p['models_cuda_only'] and p['owned_nvml_pid']>0 and p['registration_sha256']==reg
        assert p['source_logit_max_error']<=2e-7 and p['fresh_cache_max_error']<=2e-7
        assert (p['source_predict_frame_forwards'],p['source_encoding_probe_frames'],p['disposable_updates'])==(192,32,4)
        assert p['fixture_steps']=={'3407':2,'4099':2} and len(set(p['initialization_sha256'].values()))==2 and len(set(p['fixed_token_input_sha256'].values()))==1
        assert not p['source_TEST'] and not p['target_access'];initializations.append(p['initialization_sha256'])
    close(initializations[0],initializations[1])
    report=read(run/'SOURCE_REPORT.json');assert report['new_retained_updates']==14380 and report['source_training_seeds']==list(NEW)
    assert not report['source_TEST'] and not report['target_access'] and report['disposable_updates_this_process']==4
    for seed in NEW:
        folder=run/'seeds'/str(seed);cache=read(folder/'READONLY_CACHE_CHECK.json');assert cache['passed'] and cache['sha256']==initial==report['readonly_cache_checks'][str(seed)]
        schedules=[]
        for epoch in range(11):
            r=read(folder/f'epoch_{epoch:02}.json');assert r['seed']==seed and r['epoch']==epoch and r['updates']==(719 if epoch else 0) and r['total_updates']==719*epoch
            assert r['schedule_size']==(11492 if epoch else 0);assert r['identity']==['CrossLayerNoImage',epoch,seed,reg]
            assert r['checkpoint_logit_max_error']<=2e-7 and r['initialization_sha256']==initializations[0][str(seed)]
            assert sha(folder/r['frame_scores'])==r['frames_sha256']
            item=inventory['remote_checkpoints_only'][f'seeds/{seed}/'+r['checkpoint']];assert item['sha256']==r['checkpoint_sha256'] and item['bytes']>0
            audits=r['gradient_audits'];assert [a['step'] for a in audits]==([0,350,718] if epoch else [])
            for a in audits:
                assert a['gradients']['adapter']>0 and a['gradients']['correction']>0
                if epoch>1 or a['step']>0:assert a['gradients']['readout']>0
            schedules.append(r['slot_sampling_sha256'])
        assert r==report['seed_receipts'][str(seed)];sampling[seed]=schedules
    assert sampling[3407][0]==sampling[4099][0] and all(a!=b for a,b in zip(sampling[3407][1:],sampling[4099][1:]))
def source():
    started=time.monotonic();final,run,out,lock=inspect('SOURCE');reg=lock['registration_sha256'];source_runtime(run,reg,read(final/'REMOTE_INVENTORY.json'))
    val=read(run/'inputs/source_val.json');train=read(run/'inputs/fit.json');assert len(val)==22080 and len(train)==11492
    assert not {r['family_id'] for r in train}&{r['family_id'] for r in val}
    raw=frames(run/'OLD_SOURCE_RAW_ORDER.csv',['index','logit']);ordered=frames(run/'OLD_SOURCE_FRAMES.csv',['index','logit']);mapping=read(run/'OLD_SOURCE_MAPPING.json')
    historical_source=BASE/'ffpp_boundary_readout_paired_20261103/final_20261103T084508316349Z/run'
    native=read(historical_source/'inputs/source_val.json');assert sha(run/'OLD_SOURCE_RAW_ORDER.csv')==sha(historical_source/'SEMANTIC_VAL.csv')
    assert len(raw)==len(ordered)==len(mapping)==22080 and {r['historical_csv_index'] for r in mapping}==set(range(22080))
    for i,(r,m,z) in enumerate(zip(val,mapping,ordered)):
        assert m['original_index']==i and z==raw[m['historical_csv_index']]
        assert all(r[k]==m[k] for k in ('image_path','sha256','video_id','family_id','label','frame_index'))
        n=native[m['historical_csv_index']];assert (n['prior_aligned_image_path'],n['prior_aligned_sha256'])==(r['image_path'],r['sha256'])
        assert all(n[k]==r[k] for k in ('video_id','family_id','label','frame_index','source_frame_number'))
    result=read(out/'RESULTS.json');allmetrics={}
    for seed in SEEDS:
        epochs={}
        for epoch in ([10] if seed==2027 else range(11)):
            z=[x[0] for x in frames(run/'OLD_SOURCE_FRAMES.csv' if seed==2027 else run/'seeds'/str(seed)/f'frames_{epoch:02}.csv',['index','logit'])]
            videos,metric=replay(val,z)
            if epoch==10:
                draws=bootstrap(videos,7);metric['half_minus_original']=paired(videos,metric['video_AUC'],draws);close(videos,read(out/f'{seed}_VIDEOS.json'),1e-12)
            epochs[str(epoch)]=metric
        expected=dict(endpoint=epochs['10'],epochs=None if seed==2027 else epochs,reused=seed==2027);close(expected,result['seeds'][str(seed)]);allmetrics[str(seed)]=expected
        print(json.dumps(dict(source_verified_seed=seed,AUC=epochs['10']['video_AUC'])),flush=True)
    close({k:mean_sd([allmetrics[str(s)]['endpoint']['video_AUC'][k] for s in SEEDS]) for k in KEYS},result['all3'])
    close({k:mean_sd([allmetrics[str(s)]['endpoint']['video_AUC'][k] for s in NEW]) for k in KEYS},result['new2'])
    assert result['source95']=={str(s):allmetrics[str(s)]['endpoint']['video_AUC']['SemanticHalf']>=.95 for s in SEEDS}
    assert result['new_retained_updates']==14380 and result['disposable_updates']==8 and not result['source_TEST'] and not result['independent_selection_confirmation']
    proof=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,registration_sha256=reg,source_report_sha256=sha(run/'SOURCE_REPORT.json'),
        scope='independent all22 new source epochs plus old2027 saved-frame/mean32/bisect AUC, materialized family bootstrap, checkpoint/cache/gradient/budget receipts',
        source_val_frames=22080,source_val_videos=690,new_retained_updates=14380,actual_neural_replay=False,all3=result['all3'],new2=result['new2'],seconds=time.monotonic()-started)
    save_proof('SOURCE',final,proof)
def old_aucs(domain,videos):
    history={r['video_id']:r for r in read(HISTORY/'evaluation'/(domain+'_VIDEOS.json'))}
    v22={r['video_id']:r for r in read(BASE/'v22_crossdomain_fix2_pathfix1_20261102/final_20261102T024341Z/run/evaluation'/(domain+'_VIDEOS.json'))}
    assert set(history)==set(v22)=={r['video_id'] for r in videos};values={k:[] for k in HISTORICAL}
    for row in videos:
        a,b=history[row['video_id']],v22[row['video_id']];assert (a['family_id'],a['label'])==(row['family_id'],row['label'])==(b['family_id'],b['label'])
        for k,name in (('LegacySemantic','Semantic'),('OldV2','OldV2'),('V18','HistoricalV18Multi'),('OrderedCLIP','OrderedCLIP')):values[k].append(a['scores'][name])
        values['V22FixedFour'].append(b['scores']['FixedFour'])
    return {k:auc([r['label'] for r in videos],v) for k,v in values.items()}
def target():
    started=time.monotonic();final,run,out,lock=inspect('TARGET');reg=lock['registration_sha256'];result=read(out/'RESULTS.json')
    assert read(LOCAL/'SOURCE_LOCAL_VERIFICATION.json')['passed']
    close(read(LOCAL/'SOURCE_LOCAL_VERIFICATION.json'),read(run/'SOURCE_LOCAL_VERIFICATION.json'))
    for name in ('TARGET_PREFLIGHT.json','TARGET_RUNTIME_PREFLIGHT.json'):
        p=read(run/name);assert p['passed'] and p['models_cuda_only'] and p['registration_sha256']==reg and p['owned_nvml_pid']>0
        assert (p['model_frame_forwards'],p['target_encoding_probe_frames'],p['optimizer_updates'])==(336,112,0)
        assert not p['source_TEST'] and not p['target_masks'] and all(v<=2e-7 for v in p['old2027_target_probability_errors'].values())
        for seed in SEEDS:
            assert p['checkpoint_identities'][str(seed)]==['CrossLayerNoImage',10,seed,PARENT_REG if seed==2027 else reg]
        assert all(all(v.values()) for v in p['finite_and_RGB_independent'].values())
    report=read(run/'TARGET_REPORT.json');assert report['full_model_frame_forwards']==446272 and report['preflight_model_frame_forwards_this_process']==336
    assert not report['source_TEST'] and not report['target_masks'] and report['new_training_updates']==0 and report['model_weights_unchanged']
    metrics={};drawsets={s:[] for s in SEEDS};total=0
    for di,domain in enumerate(DOMAINS):
        manifest=read(run/'manifests'/(domain+'.json'));new=frames(run/'target'/(domain+'_FRAMES.csv'),['index','seed3407','seed4099']);old=[]
        with (HISTORY/'evaluation'/(domain+'_FRAMES.jsonl')).open(encoding='utf-8') as f:
            for i,(m,line) in enumerate(itertools.zip_longest(manifest,f)):
                assert m is not None and line is not None;r=json.loads(line);assert r['evaluation_row_index']==i
                assert all(r[k]==m[k] for k in ('image_path','video_id','family_id','label'))
                assert r['source_frame_number']==m.get('source_frame_number') and r['source_frame_number_known']==('source_frame_number' in m)
                assert m['partition']=='target_test' and m['dataset_id']==domain;b=r['scores']['StrongStatic'];old.append(fp(b['s']+b['d']))
        real,fake=COUNTS[di];assert len(old)==len(new)==len(manifest)==32*(real+fake);dm={}
        for seed,z in zip(SEEDS,(old,[r[0] for r in new],[r[1] for r in new])):
            videos,m=replay(manifest,z);assert (m['real_videos'],m['fake_videos'])==(real,fake);draws=bootstrap(videos,di);drawsets[seed].append(draws)
            m['half_minus_original']=paired(videos,m['video_AUC'],draws);dm[str(seed)]=m;close(videos,read(out/f'{domain}_{seed}_VIDEOS.json'),1e-12)
        value=dict(seeds=dm,historical_video_AUC=old_aucs(domain,videos),all3={k:mean_sd([dm[str(s)]['video_AUC'][k] for s in SEEDS]) for k in KEYS},new2={k:mean_sd([dm[str(s)]['video_AUC'][k] for s in NEW]) for k in KEYS})
        close(value,read(out/(domain+'_METRICS.json')));close(value,result['domains'][domain]);metrics[domain]=value
        complete=read(run/'target'/(domain+'_COMPLETE.json'));assert complete['frames']==complete['pixel_SHA_checked']==len(manifest) and complete['model_frame_forwards']==2*len(manifest)
        assert complete['sha256']==sha(run/'target'/(domain+'_FRAMES.csv'));total+=len(manifest);print(json.dumps(dict(target_verified=domain,frames=len(manifest))),flush=True)
    assert total==223136
    macro={str(s):{k:sum(metrics[d]['seeds'][str(s)]['video_AUC'][k] for d in DOMAINS)/7 for k in KEYS} for s in SEEDS};close(macro,result['macro_by_seed'])
    close({k:mean_sd([macro[str(s)][k] for s in SEEDS]) for k in KEYS},result['macro_all3']);close({k:mean_sd([macro[str(s)][k] for s in NEW]) for k in KEYS},result['macro_new2'])
    comparisons={};intervals={}
    for s in SEEDS:
        pairs={}
        for ref in ('SameSeedOriginal',*HISTORICAL):
            delta={d:100*(metrics[d]['seeds'][str(s)]['video_AUC']['SemanticHalf']-(metrics[d]['seeds'][str(s)]['video_AUC']['Semantic'] if ref=='SameSeedOriginal' else metrics[d]['historical_video_AUC'][ref])) for d in DOMAINS};worst=min(delta,key=delta.get)
            pairs[ref]=dict(per_domain_delta_pp=delta,macro_delta_pp=math.fsum(delta.values())/7,improved_domains=sum(x>0 for x in delta.values()),tied_domains=sum(x==0 for x in delta.values()),worst_domain=worst,worst_delta_pp=delta[worst])
        comparisons[str(s)]=pairs;sample=sorted(sum(v)/7 for v in zip(*drawsets[s]) if all(x is not None for x in v))
        intervals[str(s)]=dict(delta_pp=100*(macro[str(s)]['SemanticHalf']-macro[str(s)]['Semantic']),family_descriptive_95pct=[sample[int(.025*(len(sample)-1))],sample[int(.975*(len(sample)-1))]],valid_bootstrap=len(sample))
    close(comparisons,result['comparisons_by_seed']);close(intervals,result['macro_paired'])
    close({k:sum(metrics[d]['historical_video_AUC'][k] for d in DOMAINS)/7 for k in HISTORICAL},result['historical_macro'])
    assert result['target85_domains']=={str(s):sum(metrics[d]['seeds'][str(s)]['video_AUC']['SemanticHalf']>=.85 for d in DOMAINS) for s in SEEDS}
    assert result['macro89699']=={str(s):macro[str(s)]['SemanticHalf']>=.89699 for s in SEEDS}
    assert result['source_report_sha256']==sha(run/'SOURCE_REPORT.json') and result['target_report_sha256']==sha(run/'TARGET_REPORT.json')
    assert result['reused_seed2027'] and result['selection_adaptive'] and not result['independent_selection_confirmation'] and not result['source_TEST']
    proof=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,registration_sha256=reg,target_report_sha256=sha(run/'TARGET_REPORT.json'),
        scope='all3 complete source/target saved-frame FP32 half, mean32, bisect AUC, materialized family bootstrap, repair/damage, sampleSD and historical comparisons',
        target_videos=6973,target_frames=total,new_retained_updates=14380,actual_neural_replay=False,macro_by_seed=macro,macro_all3=result['macro_all3'],macro_new2=result['macro_new2'],seconds=time.monotonic()-started)
    save_proof('TARGET',final,proof)
def save_proof(stage,final,proof):
    assert not any(name in sys.modules for name in ('torch','numpy','PIL'))
    for path in (final/'LOCAL_VERIFICATION.json',LOCAL/(stage+'_LOCAL_VERIFICATION.json')):
        with path.open('x',encoding='utf-8') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('source','target'));args=parser.parse_args();globals()[args.stage]()
