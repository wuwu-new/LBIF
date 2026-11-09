"""Independent saved-score/identity audit. Standard library only; no neural execution."""
import argparse
import collections
import csv
from fractions import Fraction
import json
import math
from pathlib import Path,PurePosixPath
import random
import statistics
import manage_patch_region_full_20261105 as m

GROUPS=('SemanticRaw','SemanticHalf','BoundaryFull','PatchClassificationContrast')
CONTRASTS=(('PatchClassificationContrast','BoundaryFull'),('PatchClassificationContrast','SemanticHalf'),('BoundaryFull','SemanticHalf'),('SemanticHalf','SemanticRaw'))

def table(path,count):
    with Path(path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==count and [int(r['index']) for r in rows]==list(range(count));return rows

def sigmoid(x):return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))

def exact_auc(y,scores):
    assert len(y)==len(scores) and set(y)=={0,1} and all(math.isfinite(x) for x in scores)
    ordered=sorted(zip(scores,y),key=lambda r:r[0]);negative=0;numerator=0;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        positive=sum(y for _,y in ordered[i:j]);neg=j-i-positive
        numerator+=positive*(2*negative+neg);negative+=neg;i=j
    return Fraction(numerator,2*sum(y)*(len(y)-sum(y)))

def inspect(rows,logits):
    assert len(rows)==len(logits) and all(math.isfinite(z) for z in logits)
    byvideo=collections.OrderedDict()
    for row,z in zip(rows,logits):
        item=byvideo.setdefault(row['video_id'],dict(video_id=row['video_id'],family_id=row['family_id'],label=row['label'],method=row.get('method'),frames=0,score=0.))
        assert item['label']==row['label'] and item['family_id']==row['family_id']
        item['frames']+=1;item['score']+=sigmoid(z)
    videos=list(byvideo.values())
    for row in videos:assert row['frames']==32;row['score']/=32
    labels=[r['label'] for r in rows];vy=[r['label'] for r in videos]
    auc=exact_auc(vy,[r['score'] for r in videos])
    metrics=dict(frames=len(rows),videos=len(videos),families=len({r['family_id'] for r in videos}),
        real_frames=labels.count(0),fake_frames=labels.count(1),real_videos=vy.count(0),fake_videos=vy.count(1),
        frame_AUC=float(exact_auc(labels,logits)),video_AUC=float(auc),auc_fraction=[auc.numerator,auc.denominator],
        BCE=sum(max(z,0)-z*y+math.log1p(math.exp(-abs(z))) for y,z in zip(labels,logits))/len(rows),
        FP=sum(r['label']==0 and r['score']>=.5 for r in videos),FN=sum(r['label']==1 and r['score']<.5 for r in videos))
    return metrics,videos

def compare_metrics(actual,saved):
    assert actual.keys()==saved.keys()
    for key,value in actual.items():
        if key in ('BCE','frame_AUC','video_AUC'):assert abs(value-saved[key])<=2e-12,(key,value,saved[key])
        else:assert value==saved[key],key

def replay_videos(actual,saved):
    assert len(actual)==len(saved);maximum=0.
    for a,b in zip(actual,saved):
        assert {k:v for k,v in a.items() if k!='score'}=={k:v for k,v in b.items() if k!='score'}
        x,y=a['score'],b['score'];assert all(math.isfinite(t) and 0<=t<=1 for t in (x,y))
        ulp=max(math.ulp(x),math.ulp(y));assert abs(x-y)<=4*ulp
        maximum=max(maximum,abs(x-y)/ulp)
    for i,a in enumerate(actual):
        if not a['label']:continue
        for j,b in enumerate(actual):
            if b['label']:continue
            x,y=a['score'],b['score'];v,w=saved[i]['score'],saved[j]['score']
            assert (x>y)-(x<y)==(v>w)-(v<w),'Cross-class order/tie changed'
    return dict(max_ULPs=maximum,all_cross_class_ranks_and_ties_exact=True)

def correction(frames,base):
    assert [float(r['baseline']) for r in frames]==base
    delta=[float(r['delta']) for r in frames];fused=[float(r['fused']) for r in frames]
    assert all(math.isfinite(d) and abs(d)<=.5 for d in delta)
    assert all(m.p.f32(b+d)==z for b,d,z in zip(base,delta,fused))
    return fused

def localpath(run,path):
    return run/str(PurePosixPath(path).relative_to(PurePosixPath(m.p.REMOTE)/m.p.RUN))

def inventory(folder):
    run=folder/'run';lock=m.read(run/'REGISTRATION.json')
    assert lock['configuration']==m.p.configuration()
    assert lock['registration_sha256']==m.p.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})
    for name,digest in lock['files'].items():assert m.sha(run/name)==digest,('Bundle',name)
    saved=m.read(folder/'REMOTE_INVENTORY.json')
    for name,item in saved['files'].items():
        if name.endswith('.pt'):continue
        file=run/name;assert file.stat().st_size==item['bytes'] and m.sha(file)==item['sha256'],('Inventory',name)
    return lock,saved['files']

def summary(values):return dict(mean=statistics.mean(values),sample_SD=statistics.stdev(values) if len(values)>1 else None)

def paired(left,right):
    assert len(left)==len(right)
    for a,b in zip(left,right):assert (a['video_id'],a['family_id'],a['label'],a['frames'])==(b['video_id'],b['family_id'],b['label'],b['frames'])
    y=[r['label'] for r in left];a=[r['score'] for r in left];b=[r['score'] for r in right]
    ca=[(s>=.5)==bool(t) for s,t in zip(a,y)];cb=[(s>=.5)==bool(t) for s,t in zip(b,y)]
    return dict(delta_pp=100*float(exact_auc(y,a)-exact_auc(y,b)),repaired=sum(x and not z for x,z in zip(ca,cb)),harmed=sum(z and not x for x,z in zip(ca,cb)))

def per_method(videos):
    result={}
    for method in m.p.METHODS[1:]:
        selected=[r for r in videos if r['method'] in ('real',method)]
        result[method]=dict(real_videos=sum(r['label']==0 for r in selected),fake_videos=sum(r['label']==1 for r in selected),
            video_AUC=float(exact_auc([r['label'] for r in selected],[r['score'] for r in selected])))
    return result

def verify_source(folder):
    run=folder/'run';lock,files=inventory(folder);reg=lock['registration_sha256']
    pre=m.read(run/'SOURCE_PREFLIGHT_COMPLETE.json');done=m.read(run/'SOURCE_COMPLETE.json')
    for prefix in ('SOURCE_PREFLIGHT','SOURCE'):
        assert not (run/(prefix+'_FAILURE.json')).exists()
        receipt=m.read(run/(prefix+'_COMPLETE.json'));assert receipt['passed'] and receipt['registration_sha256']==reg
    assert pre['disposable_optimizer_updates']==6 and done['disposable_optimizer_updates']==0
    assert pre['preflight_frame_forwards']==216 and pre['network_frame_forwards']==dict(historical_full_replay=192,
        synthetic_zero_head=6,real_TRAIN_gradient=6,disposable_synthetic=12)
    assert done['new_retained_updates']==21570 and not done['new_semantic_execution']
    assert done['network_frame_forwards']==dict(retained_training=345120,full_source_VAL=662400,selected_reload=192,selected_localization=66240)
    endpointsha=m.sha(run/'SELECTED_ENDPOINT_LOCK.json');assert endpointsha==done['selection_lock_sha256']
    endpoints=m.read(run/'SELECTED_ENDPOINT_LOCK.json');train=m.read(run/'inputs/fit.json');val=m.read(run/'inputs/source_val.json')
    assert len(train)==11492 and len(val)==22080 and not ({r['family_id'] for r in train}&{r['family_id'] for r in val})
    semantic=table(run/'semantic_source/VAL.csv',22080);result={};video_errors=[];six_source={};localizations={}
    for seed in m.p.SEEDS:
        base=[m.p.f32(float(r['s'+str(seed)])/2) for r in semantic];baseline,basevideos=inspect(val,base);result[str(seed)]={}
        raw=[float(r['s'+str(seed)]) for r in semantic]
        six_source[str(seed)]={};localizations[str(seed)]={}
        reused_selected=correction(table(run/'reused_selected_source'/f'{seed}.csv',22080),base)
        reused_epoch10=correction(table(run/'reused_epoch10_source'/f'{seed}.csv',22080),base)
        reused_lock=m.read(run/'REUSED_SELECTED_ENDPOINTS.json')[str(seed)]
        for group,z in (('SemanticRaw',raw),('SemanticHalf',base),('BoundaryFull',reused_selected)):
            met,vid=inspect(val,z)
            if group=='BoundaryFull':
                assert Fraction(*met['auc_fraction'])==Fraction(reused_lock['auc_numerator'],reused_lock['auc_denominator'])
            six_source[str(seed)][group]=dict(selected_metrics=met,source95_met=met['video_AUC']>=.95,
                per_method=per_method(vid),reused=True,selected_epoch=reused_lock['epoch'] if group=='BoundaryFull' else 10,
                epoch10_metrics=inspect(val,reused_epoch10)[0] if group=='BoundaryFull' else met)
        localizations[str(seed)]['BoundaryFull']=dict(reused=True,
            raw_per_frame=m.read(run/'reused_localization'/f'{seed}.json'))
        for arm in m.p.NEW_ARMS:
            directory=run/'source'/str(seed)/arm;records=m.read(directory/'ALL_EPOCHS.json');actual=[]
            assert len(records)==10
            for epoch,record in enumerate(records,1):
                assert (record['seed'],record['arm'],record['epoch'],record['updates'],record['registration_sha256'])==(seed,arm,epoch,719*epoch,reg)
                ck=files[str(PurePosixPath(record['path']).relative_to(PurePosixPath(m.p.REMOTE)/m.p.RUN))]
                assert ck['sha256']==record['sha256']
                exposure=m.read(directory/f'EXPOSURE_{epoch:02}.json')
                assert exposure['updates']==719 and exposure['trajectory_updates']==719*epoch
                schedule=m.p.sampling(train,epoch,seed)
                assert exposure['schedule_sha256']==m.p.fingerprint(schedule) and exposure['unique_frames']==len({i for batch in schedule for i in batch})
                assert pre['shared_initialization_sha256'][str(seed)][arm]==m.read(run/'REUSED_INITIALIZATION.json')['initialization_sha256'][str(seed)]
                assert len(exposure['loss_means'])==5 and exposure['patch_weights']==m.p.variant(arm)
                assert exposure['patch_coverage_totals']['eligible']>exposure['patch_coverage_totals']['fake']>0
                fp=localpath(run,record['frames_path']);assert m.sha(fp)==record['frames_sha256']
                frames=table(fp,22080);logits=correction(frames,base);metrics,videos=inspect(val,logits);compare_metrics(metrics,record['metrics'])
                assert record==m.read(fp.parent/'RESULT.json')
                assert m.sha(fp.parent/'VIDEOS.json')==record['videos_sha256']
                video_errors.append(dict(seed=seed,arm=arm,epoch=epoch,**replay_videos(videos,m.read(fp.parent/'VIDEOS.json'))))
                actual.append((Fraction(*metrics['auc_fraction']),-epoch,record,videos))
            best=max(actual,key=lambda r:r[:2]);assert best[2]==m.read(directory/'SELECTED.json')==endpoints[str(seed)][arm]
            replay=m.read(directory/'SELECTED_REPLAY.json');assert replay['passed'] and replay['max_logit_error']<=2e-7 and replay['epoch']==best[2]['epoch']
            if arm!='UniformNoAux':
                loc=m.read(directory/'LOCALIZATION.json');assert loc['threshold']==.5 and loc['truth_threshold']==.1 and loc['tolerance_pixels']==2
                assert sum(x['frames'] for x in loc['per_method'].values())==sum(r['mask_valid'] for r in val)
                for row in loc['per_method'].values():
                    assert 0<=row['precision_hits']<=row['predicted'] and 0<=row['recall_hits']<=row['GT_pixels']
                    assert 0<=row['response_sum']<=row['frames']
                    assert sum(row[k] for k in ('patch_TP','patch_FP','patch_FN','patch_TN'))==256*row['frames']
                localizations[str(seed)][arm]=loc
            result[str(seed)][arm]=dict(selected_epoch=best[2]['epoch'],selected_metrics=best[2]['metrics'],epoch10_metrics=actual[-1][2]['metrics'],
                baseline_metrics=baseline,selected_minus_baseline=paired(best[3],basevideos),per_method=per_method(best[3]),
                all_epoch_video_AUC=[r[2]['metrics']['video_AUC'] for r in actual])
            six_source[str(seed)][arm]=dict(**result[str(seed)][arm],source95_met=best[2]['metrics']['video_AUC']>=.95,reused=False)
    coverage={part:{method:dict(frames=sum(r['method']==method for r in rows),
        eligible_masks=sum(r['method']==method and r['mask_valid'] for r in rows)) for method in m.p.METHODS}
        for part,rows in (('TRAIN',train),('VAL',val))}
    proof=dict(passed=True,registration_sha256=reg,selection_lock_sha256=endpointsha,source_only=True,new_retained_updates=21570,
        independent_saved_scores_replay=True,neural_rerun=False,all30_epochs_verified=True,exact_ranks_and_earliest_selection=True,
        model_tolerance_unchanged=2e-7,video_probability_replay=video_errors,results=result,
        source_six_groups=six_source,eligible_mask_coverage=coverage,selected_localization=localizations,
        costs=dict(preflight_seconds=pre['seconds'],source_seconds=done['seconds'],network_frame_forwards=done['network_frame_forwards'],
            disposable_frame_forwards=pre['network_frame_forwards']),
        old_tasks_paused=True,automation_paused=True,target_pending=True)
    m.dump(folder/'LOCAL_VERIFICATION.json',proof);m.dump(m.LOCAL/'SOURCE_LOCAL_VERIFICATION.json',proof)
    m.update('Paper_components_source_verified',folder/'LOCAL_VERIFICATION.json',status='source_verified_pending_target',started=False,
        source_complete=True,source_archive=str(folder.relative_to(m.ROOT)),source_verification_sha256=m.sha(folder/'LOCAL_VERIFICATION.json'),
        next_action='Upload source proof then explicit fixed target preflight and complete seven-domain test; no extra training or monitor')
    return proof

def family_intervals(videos):
    """1000 fixed family draws, shared across seeds/groups. Invalid draws disclosed, not replaced."""
    rng=random.Random(20261104);samples={a+'-minus-'+b:[] for a,b in CONTRASTS}
    grouped={}
    for domain in m.p.DOMAINS:
        reference=videos[2027][domain]['SemanticHalf'];families=collections.OrderedDict()
        for i,row in enumerate(reference):families.setdefault(row['family_id'],[]).append(i)
        grouped[domain]=list(families.values())
    for _ in range(1000):
        totals=collections.defaultdict(list);valid=True
        for domain in m.p.DOMAINS:
            families=grouped[domain];ids=[i for group in (rng.choice(families) for _ in families) for i in group]
            y=[videos[2027][domain]['SemanticHalf'][i]['label'] for i in ids]
            if len(set(y))<2:valid=False;break
            for seed in m.p.SEEDS:
                scores={g:float(exact_auc(y,[videos[seed][domain][g][i]['score'] for i in ids])) for g in GROUPS}
                for a,b in CONTRASTS:totals[a+'-minus-'+b].append(100*(scores[a]-scores[b]))
        if valid:
            for key in samples:samples[key].append(statistics.mean(totals[key]))
    result={}
    for key,values in samples.items():
        assert values;values.sort();n=len(values)
        result[key]=dict(attempted_family_draws=1000,valid_macro_draws=n,lower_pp=values[int(.025*(n-1))],upper_pp=values[int(.975*(n-1))],
            descriptive_only=True,shared_family_resampling_across_seeds=True)
    return result

def verify_target(folder):
    run=folder/'run';lock,_=inventory(folder);reg=lock['registration_sha256'];done=m.read(run/'TARGET_COMPLETE.json')
    assert done['passed'] and done['registration_sha256']==reg and done['full_residual_forwards']==669408 and done['new_optimizer_updates']==0
    assert done['network_frame_forwards']==dict(source_reload_probe=192,target_repeat_probe=672,full_target=669408)
    pre=m.read(run/'TARGET_PREFLIGHT_COMPLETE.json');assert pre['passed'] and pre['preflight_forwards']==864
    sourceproof=m.read(run/'SOURCE_LOCAL_VERIFICATION.json')
    assert sourceproof['passed'] and sourceproof['registration_sha256']==reg
    assert sourceproof['selection_lock_sha256']==m.sha(run/'SELECTED_ENDPOINT_LOCK.json')
    results={s:{} for s in m.p.SEEDS};videos={s:{} for s in m.p.SEEDS}
    for domain,counts in zip(m.p.DOMAINS,m.p.COUNTS):
        rows=m.read(run/'manifests'/(domain+'.json'));assert len(rows)==32*sum(counts)
        sem=table(run/'semantic_target'/(domain+'.csv'),len(rows))
        full=table(run/'reused_target'/(domain+'_FRAMES.csv'),len(rows));frames=table(run/'target'/(domain+'_FRAMES.csv'),len(rows))
        for seed in m.p.SEEDS:
            raw=[float(r['s'+str(seed)]) for r in sem];base=[m.p.f32(x/2) for x in raw]
            assert [float(r['s'+str(seed)]) for r in full]==raw
            values=dict(SemanticRaw=raw,SemanticHalf=base,BoundaryFull=[float(r['f'+str(seed)]) for r in full])
            assert all(m.p.f32(b+float(r['d'+str(seed)]))==z for b,r,z in zip(base,full,values['BoundaryFull']))
            for arm in m.p.NEW_ARMS:
                prefix=f'{seed}_{arm}_';sub=[{k:r[prefix+k] for k in ('baseline','delta','fused')} for r in frames]
                values[arm]=correction(sub,base)
            results[seed][domain]={};videos[seed][domain]={}
            for group in GROUPS:
                met,vid=inspect(rows,values[group]);assert (met['real_videos'],met['fake_videos'])==counts
                results[seed][domain][group]=met;videos[seed][domain][group]=vid
    mac={group:summary([statistics.mean(results[s][d][group]['video_AUC'] for d in m.p.DOMAINS) for s in m.p.SEEDS]) for group in GROUPS}
    macro_frame={group:summary([statistics.mean(results[s][d][group]['frame_AUC'] for d in m.p.DOMAINS) for s in m.p.SEEDS]) for group in GROUPS}
    domainsummary={d:{g:summary([results[s][d][g]['video_AUC'] for s in m.p.SEEDS]) for g in GROUPS} for d in m.p.DOMAINS}
    comparisons={}
    for a,b in CONTRASTS:
        byseed={str(s):{d:paired(videos[s][d][a],videos[s][d][b]) for d in m.p.DOMAINS} for s in m.p.SEEDS}
        domainmeans={d:statistics.mean(byseed[str(s)][d]['delta_pp'] for s in m.p.SEEDS) for d in m.p.DOMAINS}
        comparisons[a+'-minus-'+b]=dict(by_seed_domain=byseed,mean_delta_pp=statistics.mean(domainmeans.values()),
            paired_seed_macro_delta_pp=summary([statistics.mean(byseed[str(s)][d]['delta_pp'] for d in m.p.DOMAINS) for s in m.p.SEEDS]),
            improved_domains=sum(x>0 for x in domainmeans.values()),worst_domain=min(domainmeans,key=domainmeans.get),worst_delta_pp=min(domainmeans.values()))
    result=dict(per_seed_domain=results,macro_video=mac,macro_frame=macro_frame,per_domain_mean_sampleSD=domainsummary,
        source_six_groups=sourceproof['source_six_groups'],source_mask_coverage=sourceproof['eligible_mask_coverage'],
        selected_localization=sourceproof['selected_localization'],comparisons=comparisons,family_intervals=family_intervals(videos),
        target85_met={str(s):{d:{g:results[s][d][g]['video_AUC']>=.85 for g in GROUPS} for d in m.p.DOMAINS} for s in m.p.SEEDS},
        costs=dict(source=sourceproof['costs'],target_preflight_seconds=pre['seconds'],target_seconds=done['seconds'],
            target_network_frame_forwards=done['network_frame_forwards'],no_new_semantic_execution=True,
            reused_method_cost_not_zero_in_deployment=True),
        adaptive_exploration=True,literature_CLIP_in_experiment=False,all_groups_reported=True,independent_blind_confirmation=False)
    m.dump(folder/'RESULTS.json',result)
    proof=dict(passed=True,registration_sha256=reg,results_sha256=m.sha(folder/'RESULTS.json'),all_three_endpoints_seven_domains_verified=True,
        four_groups_including_reused=True,full_target_frames=669408,neural_rerun=False)
    m.dump(folder/'LOCAL_VERIFICATION.json',proof)
    lines=['# 真实分块分类加区域对比完整版本结果','','所有组均报告；增强语义不冒称原版CLIP。完整双流与语义为已核验结果复用。','','|组|七域视频AUC均值%|样本SD pp|','|---|---:|---:|']
    for group in GROUPS:lines.append(f"|{group}|{100*mac[group]['mean']:.6f}|{100*mac[group]['sample_SD']:.6f}|")
    lines+=['','|对比|平均变化pp|改善域数|最差退化pp|','|---|---:|---:|---:|']
    for name,c in comparisons.items():lines.append(f"|{name}|{c['mean_delta_pp']:+.6f}|{c['improved_domains']}/7|{c['worst_delta_pp']:+.6f}|")
    lines+=['','逐seed/域真假视频与帧分母、两级AUC、修复误伤和家族描述区间见RESULTS.json。',
        '源四组结果（最佳轮次及固定epoch10）、源95%/目标85%、掩膜覆盖、定位和实际阶段耗时亦见RESULTS.json。',
        '七域已自适应使用，源VAL用于选模，完整模型有事后选臂历史；不能将小幅或不稳定变化称为已证实泛化增益。',
        '没有新CLIP训练、没有目标掩膜或目标梯度；旧任务及监控仍暂停。']
    with (folder/'RESULTS_COMPONENTS_CN.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    m.update('Paper_components_complete_verified',folder/'LOCAL_VERIFICATION.json',status='complete_verified',started=False,complete=True,
        target_archive=str(folder.relative_to(m.ROOT)),verification_sha256=m.sha(folder/'LOCAL_VERIFICATION.json'),
        next_action='Deliver full patch variant and three reused controls including negative results; no automatic new training, rounds, monitor or deployment')
    return proof

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('source','target'));ap.add_argument('archive',type=Path);args=ap.parse_args()
    print(json.dumps((verify_source if args.stage=='source' else verify_target)(args.archive.resolve()),ensure_ascii=False))
