"""Producer for full saved-score analysis; standard library, no local models."""
import argparse
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
import sys
import time

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/guarded_transfer_20261026'
sys.path.insert(0,str(ROOT/'code/dual_stream_20261026/semantic_half_three_seed_fix1_20261104'))
import confirmation_plan as q
LOCAL=BASE/'best_four_three_seed_confirmation_fix1_20261104'
HISTORY=BASE/q.HISTORY/'final_20261102T113604Z/run'
KEYS=('Semantic','SemanticHalf')
HISTORICAL=('LegacySemantic','OldV2','V18','V22FixedFour','OrderedCLIP')


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(path,x):
    with path.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def auc(y,s):
    p=sum(y);n=len(y)-p
    if not p or not n:return None
    below=win=0.
    for _,g in itertools.groupby(sorted(zip(s,y)),key=lambda z:z[0]):
        a=list(g);pp=sum(t for _,t in a);nn=len(a)-pp;win+=pp*(below+.5*nn);below+=nn
    return win/(p*n)
def archive(stage):return ROOT/read(LOCAL/(stage+'_ARCHIVE_RECEIPT.json'))['path']
def files(stage):
    final=archive(stage);run=final/'run';inv=read(final/'REMOTE_INVENTORY.json');lock=read(run/'REGISTRATION.json')
    assert sha(final/'run.tar.gz')==inv['sha256']
    assert lock['registration_sha256']==q.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})
    assert lock['configuration']==q.configuration()
    for name,item in inv['files'].items():assert sha(run/name)==item['sha256']
    for name,value in lock['files'].items():assert sha(run/name)==value
    for name,value in lock['local_evidence_sha256'].items():assert sha(ROOT/name)==value
    assert read(run/(stage+'_COMPLETE.json'))['report_sha256']==sha(run/(stage+'_REPORT.json'))
    return final,run,lock
def logits(path,expected_columns):
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==expected_columns
        values=[]
        for i,row in enumerate(reader):
            assert int(row['index'])==i;zs=[float(row[k]) for k in expected_columns[1:]]
            assert all(math.isfinite(z) and q.f32(z)==z for z in zs);values.append(zs)
    return values
def aggregate(rows,z):
    assert len(rows)==len(z);g=collections.OrderedDict();frame_y=[]
    for i,(row,value) in enumerate(zip(rows,z)):
        v=g.setdefault(row['video_id'],dict(video_id=row['video_id'],family_id=row['family_id'],label=row['label'],indices=[],frames=0,scores={k:[] for k in KEYS}))
        assert (v['family_id'],v['label'])==(row['family_id'],row['label'])
        v['frames']+=1;v['indices'].append(row['frame_index']);frame_y.append(row['label'])
        for key,logit in zip(KEYS,(value,q.f32(value*.5))):v['scores'][key].append(q.sigmoid(logit))
    videos=[]
    for row in g.values():
        assert row['frames']==32 and sorted(row.pop('indices'))==list(range(32))
        row['scores']={key:math.fsum(a)/32 for key,a in row['scores'].items()};videos.append(row)
    frame={}
    for key,values in zip(KEYS,(z,[q.f32(t*.5) for t in z])):
        frame[key]=dict(AUC=auc(frame_y,values),FP=sum(y==0 and a>=0 for y,a in zip(frame_y,values)),FN=sum(y==1 and a<0 for y,a in zip(frame_y,values)))
    return videos,frame
def draw_values(videos,index):
    families=sorted({r['family_id'] for r in videos});y=[r['label'] for r in videos];order={}
    for k in KEYS:
        ids=sorted(range(len(videos)),key=lambda i:videos[i]['scores'][k]);order[k]=[list(g) for _,g in itertools.groupby(ids,key=lambda i:videos[i]['scores'][k])]
    rng=random.Random(20271104+index);out=[]
    for _ in range(1000):
        c=collections.Counter(rng.choice(families) for _ in families);w=[c[r['family_id']] for r in videos];p=sum(a*b for a,b in zip(w,y));n=sum(w)-p
        if not p or not n:out.append(None);continue
        result={}
        for key,groups in order.items():
            below=win=0.
            for group in groups:
                pp=sum(w[i] for i in group if y[i]);nn=sum(w[i] for i in group if not y[i]);win+=pp*(below+.5*nn);below+=nn
            result[key]=win/(p*n)
        out.append(100*(result['SemanticHalf']-result['Semantic']))
    return out
def paired(videos,aa,draws):
    pos=[r for r in videos if r['label']];neg=[r for r in videos if not r['label']];repair=damage=0.
    for a in pos:
        for b in neg:
            w={k:(a['scores'][k]>b['scores'][k])+.5*(a['scores'][k]==b['scores'][k]) for k in KEYS};d=w['SemanticHalf']-w['Semantic'];repair+=max(0.,d);damage+=max(0.,-d)
    sample=sorted(t for t in draws if t is not None);scale=100/(len(pos)*len(neg));net=100*(aa['SemanticHalf']-aa['Semantic'])
    assert abs((repair-damage)*scale-net)<1e-10
    before=[(r['scores']['Semantic']>=.5)==bool(r['label']) for r in videos];after=[(r['scores']['SemanticHalf']>=.5)==bool(r['label']) for r in videos]
    return dict(delta_pp=net,rank_repair_pp=repair*scale,rank_damage_pp=damage*scale,
        repaired_videos=sum(not a and b for a,b in zip(before,after)),damaged_videos=sum(a and not b for a,b in zip(before,after)),
        family_descriptive_95pct=[sample[int(.025*(len(sample)-1))],sample[int(.975*(len(sample)-1))]],valid_bootstrap=len(sample))
def metric(rows,z,index,interval=True):
    v,frame=aggregate(rows,z);y=[r['label'] for r in v];aa={k:auc(y,[r['scores'][k] for r in v]) for k in KEYS};draws=draw_values(v,index) if interval else []
    m=dict(real_videos=len(v)-sum(y),fake_videos=sum(y),real_frames=sum(r['label']==0 for r in rows),fake_frames=sum(r['label']==1 for r in rows),
        families=len({r['family_id'] for r in v}),video_AUC=aa,frame_metrics=frame,
        confusion={k:dict(FP=sum(r['label']==0 and r['scores'][k]>=.5 for r in v),FN=sum(r['label']==1 and r['scores'][k]<.5 for r in v)) for k in KEYS})
    if interval:m['half_minus_original']=paired(v,aa,draws)
    return v,m,draws
def mean_SD(values):return dict(mean=statistics.mean(values),sample_SD=statistics.stdev(values),n=len(values))


def historical(domain,rows):
    previous={r['video_id']:r for r in read(HISTORY/'evaluation'/(domain+'_VIDEOS.json'))}
    v22={r['video_id']:r for r in read(BASE/'v22_crossdomain_fix2_pathfix1_20261102/final_20261102T024341Z/run/evaluation'/(domain+'_VIDEOS.json'))}
    assert {r['video_id'] for r in rows}==set(previous)==set(v22)
    y=[r['label'] for r in rows];values={k:[] for k in HISTORICAL}
    for r in rows:
        a,b=previous[r['video_id']],v22[r['video_id']]
        assert (a['family_id'],a['label'])==(r['family_id'],r['label'])==(b['family_id'],b['label'])
        for k,name in (('LegacySemantic','Semantic'),('OldV2','OldV2'),('V18','HistoricalV18Multi'),('OrderedCLIP','OrderedCLIP')):values[k].append(a['scores'][name])
        values['V22FixedFour'].append(b['scores']['FixedFour'])
    return {k:auc(y,v) for k,v in values.items()}


def comparison_summary(metrics,seed,references):
    result={}
    for reference in references:
        delta={d:100*(metrics[d]['seeds'][str(seed)]['video_AUC']['SemanticHalf']-(metrics[d]['seeds'][str(seed)]['video_AUC']['Semantic'] if reference=='SameSeedOriginal' else metrics[d]['historical_video_AUC'][reference])) for d in q.DOMAINS}
        worst=min(delta,key=delta.get)
        result[reference]=dict(per_domain_delta_pp=delta,macro_delta_pp=math.fsum(delta.values())/7,improved_domains=sum(x>0 for x in delta.values()),tied_domains=sum(x==0 for x in delta.values()),worst_domain=worst,worst_delta_pp=delta[worst])
    return result


def source():
    started=time.monotonic();final,run,lock=files('SOURCE');out=final/'analysis';out.mkdir();val=read(run/'inputs/source_val.json');allmetrics={}
    old=[r[0] for r in logits(run/'OLD_SOURCE_FRAMES.csv',['index','logit'])]
    v,m,draw=metric(val,old,7);dump(out/'2027_VIDEOS.json',v);allmetrics['2027']=dict(endpoint=m,epochs=None,reused=True)
    for seed in q.NEW_SEEDS:
        epochs={}
        for epoch in range(11):
            path=run/'seeds'/str(seed)/f'frames_{epoch:02}.csv';z=[r[0] for r in logits(path,['index','logit'])]
            v,m,_=metric(val,z,7,interval=epoch==10);epochs[str(epoch)]=m
            if epoch==10:dump(out/f'{seed}_VIDEOS.json',v)
        allmetrics[str(seed)]=dict(endpoint=epochs['10'],epochs=epochs,reused=False)
        print(json.dumps(dict(source_seed=seed,endpoint=epochs['10']['video_AUC'])),flush=True)
    result=dict(registration_sha256=lock['registration_sha256'],seeds=allmetrics,all3={k:mean_SD([allmetrics[str(s)]['endpoint']['video_AUC'][k] for s in q.SEEDS]) for k in KEYS},
        new2={k:mean_SD([allmetrics[str(s)]['endpoint']['video_AUC'][k] for s in q.NEW_SEEDS]) for k in KEYS},new_retained_updates=14380,disposable_updates=8,
        source95={str(s):allmetrics[str(s)]['endpoint']['video_AUC']['SemanticHalf']>=.95 for s in q.SEEDS},
        source_TEST=False,independent_selection_confirmation=False,seconds=time.monotonic()-started)
    dump(out/'RESULTS.json',result);dump(out/'COMPLETE.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print(json.dumps(result['all3']))


def target():
    started=time.monotonic();final,run,lock=files('TARGET');out=final/'analysis';out.mkdir();metrics={};drawsets={str(s):[] for s in q.SEEDS}
    for di,domain in enumerate(q.DOMAINS):
        rows=read(run/'manifests'/(domain+'.json'));new=logits(run/'target'/(domain+'_FRAMES.csv'),['index','seed3407','seed4099'])
        old=[]
        with (HISTORY/'evaluation'/(domain+'_FRAMES.jsonl')).open(encoding='utf-8') as f:
            for i,(row,line) in enumerate(itertools.zip_longest(rows,f)):
                assert row is not None and line is not None;entry=json.loads(line);assert entry['evaluation_row_index']==i
                assert all(entry[k]==row[k] for k in ('image_path','video_id','family_id','label'));b=entry['scores']['StrongStatic'];old.append(q.f32(b['s']+b['d']))
        assert len(new)==len(old)==len(rows);domain_metrics={}
        for seed,z in zip(q.SEEDS,(old,[v[0] for v in new],[v[1] for v in new])):
            videos,m,draws=metric(rows,z,di);real,fake=q.COUNTS[di];assert (m['real_videos'],m['fake_videos'])==(real,fake)
            dump(out/f'{domain}_{seed}_VIDEOS.json',videos);domain_metrics[str(seed)]=m;drawsets[str(seed)].append(draws)
        metrics[domain]=dict(seeds=domain_metrics,historical_video_AUC=historical(domain,videos),all3={k:mean_SD([domain_metrics[str(s)]['video_AUC'][k] for s in q.SEEDS]) for k in KEYS},
            new2={k:mean_SD([domain_metrics[str(s)]['video_AUC'][k] for s in q.NEW_SEEDS]) for k in KEYS})
        dump(out/(domain+'_METRICS.json'),metrics[domain]);print(json.dumps(dict(domain=domain,all3=metrics[domain]['all3'])),flush=True)
    macro={str(s):{k:sum(metrics[d]['seeds'][str(s)]['video_AUC'][k] for d in q.DOMAINS)/7 for k in KEYS} for s in q.SEEDS}
    intervals={}
    for seed in q.SEEDS:
        sample=sorted(sum(v)/7 for v in zip(*drawsets[str(seed)]) if all(x is not None for x in v))
        intervals[str(seed)]=dict(delta_pp=100*(macro[str(seed)]['SemanticHalf']-macro[str(seed)]['Semantic']),family_descriptive_95pct=[sample[int(.025*(len(sample)-1))],sample[int(.975*(len(sample)-1))]],valid_bootstrap=len(sample))
    result=dict(registration_sha256=lock['registration_sha256'],configuration=q.configuration(),domains=metrics,macro_by_seed=macro,
        macro_all3={k:mean_SD([macro[str(s)][k] for s in q.SEEDS]) for k in KEYS},macro_new2={k:mean_SD([macro[str(s)][k] for s in q.NEW_SEEDS]) for k in KEYS},
        comparisons_by_seed={str(s):comparison_summary(metrics,s,('SameSeedOriginal',*HISTORICAL)) for s in q.SEEDS},
        historical_macro={k:sum(metrics[d]['historical_video_AUC'][k] for d in q.DOMAINS)/7 for k in HISTORICAL},
        macro_paired=intervals,target85_domains={str(s):sum(metrics[d]['seeds'][str(s)]['video_AUC']['SemanticHalf']>=.85 for d in q.DOMAINS) for s in q.SEEDS},
        macro89699={str(s):macro[str(s)]['SemanticHalf']>=.89699 for s in q.SEEDS},
        reused_seed2027=True,selection_adaptive=True,independent_selection_confirmation=False,source_TEST=False,
        new_retained_updates=14380,source_disposable_updates=8,new_target_model_frame_forwards=446272,seconds=time.monotonic()-started,
        source_report_sha256=sha(run/'SOURCE_REPORT.json'),target_report_sha256=sha(run/'TARGET_REPORT.json'))
    dump(out/'RESULTS.json',result);dump(out/'COMPLETE.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print(json.dumps(dict(macro_all3=result['macro_all3'],macro_new2=result['macro_new2'],macro_by_seed=macro)))


def finalize():
    final,run,lock=files('TARGET');assert read(LOCAL/'TARGET_LOCAL_VERIFICATION.json')['passed'];result=read(final/'analysis/RESULTS.json')
    srcfinal=archive('SOURCE');src=read(srcfinal/'analysis/RESULTS.json')
    lines=['# SemanticHalf：三种子完整源/七域','',
        '固定种子2027/3407/4099，CrossLayerNoImage epoch10；原CLIP及历史语义anchor固定。2027为参与四选一的旧权重复用，新3407/4099独立训练各7190更新。减半只在评价输出，不在训练loss中。七域已自适应使用，不称三个全新盲测或独立选择确认。','',
        '|域|真/假视频|真/假帧|2027 AUC%|3407 AUC%|4099 AUC%|全3均值±样本SD%|新增2均值±样本SD%|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for d,m in result['domains'].items():
        a=m['seeds']['2027'];v='|'.join(f"{m['seeds'][str(s)]['video_AUC']['SemanticHalf']*100:.6f}" for s in q.SEEDS)
        all3=m['all3']['SemanticHalf'];new2=m['new2']['SemanticHalf']
        lines.append(f"|{d}|{a['real_videos']}/{a['fake_videos']}|{a['real_frames']}/{a['fake_frames']}|{v}|{all3['mean']*100:.6f}±{all3['sample_SD']*100:.6f}|{new2['mean']*100:.6f}±{new2['sample_SD']*100:.6f}|")
    lines+=['','macro逐seed：'+str(result['macro_by_seed']),'','全3 macro：'+str(result['macro_all3']),'','新增2 macro：'+str(result['macro_new2']),'',
        '源VAL逐seed：'+str({s:x['endpoint']['video_AUC'] for s,x in src['seeds'].items()}),'',
        '源95%：'+str({s:x['endpoint']['video_AUC']['SemanticHalf']>=.95 for s,x in src['seeds'].items()}),'目标85%域数：'+str(result['target85_domains']),
        'macro89.699%：'+str(result['macro89699']),'',
        '各seed原尺度对照、每域/宏平均家族描述区间、排序修复/误伤及分类修复/误伤均保存analysis/RESULTS.json。种子样本SD是训练变化，家族区间是当前数据抽样描述，二者不能互相替代。无事后挑选种子或最优epoch。','',
        '相对历史旧Semantic、OldV2、V18、V22FixedFour及研究最佳OrderedCLIP的逐域差、macro差、改善域数与最差退化见RESULTS.json/comparisons_by_seed；历史模型均为旧单seed参照，不冒充匹配三seed对照。','',
        '成本：14380保留更新，8一次性预检更新；训练459680帧前向、完整源VAL485760、重载检查352、源预检384；新两seed完整目标446272、目标预检672模型帧前向；新CLIP源23048帧、目标223360帧。2027全目标仅复用保存分数。源GPU进程耗时'+str(read(run/'SOURCE_REPORT.json')['seconds'])+'秒，目标GPU进程耗时'+str(read(run/'TARGET_REPORT.json')['seconds'])+'秒（包含运行内预检，不含独立预检；非纯模型kernel计时）。全部模型加载/前向/训练在region-46 CUDA；本地仅保存数值独立重放。源码/输入/权重身份、进程退出、资源与日志均归档；权重留远端，历史失败/结果不覆盖。','',
        'OldV2默认、历史研究最佳和旧十轮/五seed/automation暂停保持；不自动追加种子或部署。','']
    path=final/'RESULTS_SEMANTIC_HALF_THREE_SEED_CN.md'
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    proof=LOCAL/'DELIVERY.json';dump(proof,dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),report=path.relative_to(ROOT).as_posix(),
        report_sha256=sha(path),verification_sha256=sha(LOCAL/'TARGET_LOCAL_VERIFICATION.json'),results_sha256=sha(final/'analysis/RESULTS.json'),macro_all3=result['macro_all3'],macro_new2=result['macro_new2']))
    from manage_semantic_half_three_seed_fix1_20261104 import update
    update(dict(status='complete_source_target_archived_independently_verified',complete=True,report=path.relative_to(ROOT).as_posix(),
        next_action='Deliver fixed three-seed full experiment; do not add seeds, deploy, resume old tasks or automation'),
        'SemanticHalf_three_seed_complete_verified',proof);print(json.dumps(read(proof),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('source','target','finalize'));args=parser.parse_args()
    try:globals()[args.action]()
    except Exception:
        import traceback
        dump(LOCAL/('ANALYSIS_FAILURE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),dict(traceback=traceback.format_exc()));raise
