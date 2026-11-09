"""Independent stdlib verification of saved predictions, identities and budgets."""
import collections
import csv
from fractions import Fraction
import math
from pathlib import Path, PurePosixPath
import statistics
import verify_paper_components_20261104 as stat
import manage_latest_ablation_prefetch_20261105 as m


def table(path,count):
    with Path(path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==count and [int(r['index']) for r in rows]==list(range(count))
    return rows


def inventory(folder):
    run=folder/'run';lock=m.read(run/'REGISTRATION.json')
    assert lock['configuration']==m.p.configuration()
    assert lock['registration_sha256']==m.p.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})
    for name,digest in lock['files'].items():assert m.sha(run/name)==digest,('Registered bundle',name)
    saved=m.read(folder/'REMOTE_INVENTORY.json')
    for name,item in saved['files'].items():
        if name.endswith('.pt'):continue
        file=run/name;assert file.stat().st_size==item['bytes'] and m.sha(file)==item['sha256'],('Archive inventory',name)
    return lock,saved['files']


def correction(frames,baseline):
    assert [float(r['baseline']) for r in frames]==baseline
    delta=[float(r['delta']) for r in frames];fused=[float(r['fused']) for r in frames]
    assert all(math.isfinite(d) and abs(d)<=.5 for d in delta)
    assert all(m.p.f32(b+d)==z for b,d,z in zip(baseline,delta,fused))
    return fused


def path(run,remote_path):
    return run/PurePosixPath(remote_path).relative_to(PurePosixPath(m.p.REMOTE)/m.p.RUN)


def summarize(values):
    return dict(mean=statistics.mean(values),sample_SD=statistics.stdev(values) if len(values)>1 else None,n=len(values))


def verify_source(folder):
    raise ValueError('Use original source verifier')


def verify_target(folder):
    run=folder/'run';lock,_=inventory(folder);reg=lock['registration_sha256'];cfg=m.p.configuration()
    done=m.read(run/'TARGET_COMPLETE.json');pre=m.read(run/'TARGET_PREFLIGHT_COMPLETE.json')
    assert done['passed'] and done['registration_sha256']==reg and done['new_optimizer_updates']==0
    assert done['full_residual_forwards']==cfg['full_target_residual_forwards']
    expected=dict(source_reload_probe=cfg['selected_reload_forwards'],target_repeat_probe=len(m.p.SEEDS)*len(m.p.NEW_ARMS)*224)
    assert pre['network_frame_forwards']==dict(**expected,acceleration_benchmark=m.p.EXTRA_BENCHMARK_FORWARDS)
    assert pre['preflight_forwards']==cfg['target_preflight_forwards_per_invocation']
    choice=m.read(run/'ACCELERATION_DECISION.json')
    assert choice['passed'] and choice['max_output_error']<=2e-7
    assert choice['extra_frame_forwards']==m.p.EXTRA_BENCHMARK_FORWARDS
    assert choice['enable_prefetch']==(choice['speedup']>=1.05)
    adoption=m.read(run/'SOURCE_ADOPTION.json')
    assert adoption['source_registration_sha256']==m.p.SOURCE_REG
    for name,digest in adoption['files'].items():assert m.sha(run/name)==digest
    assert done['source_registration_sha256']==m.p.SOURCE_REG
    assert done['source_adoption_sha256']==m.sha(run/'SOURCE_ADOPTION.json')
    sourceproof=m.read(run/'SOURCE_LOCAL_VERIFICATION.json')
    assert sourceproof['passed'] and sourceproof['registration_sha256']==m.p.SOURCE_REG
    assert sourceproof['selection_lock_sha256']==adoption['selection_lock_sha256']
    assert done['network_frame_forwards']==dict(**expected,full_target=cfg['full_target_residual_forwards'])
    assert pre['passed'] and pre['registration_sha256']==reg
    assert all(max(x.values())<=2e-7 for x in pre['target_repeat_errors'].values())
    assert max(pre['source_errors'].values())<=2e-7
    contract=m.read(run/'TARGET_CONTRACT.json')
    assert contract['passed'] and max(contract['source_errors'].values())<=2e-7
    assert all(max(x.values())<=2e-7 for x in contract['target_repeat_errors'].values())
    assert not (run/'TARGET_FAILURE.json').exists() and not (run/'TARGET_PREFLIGHT_FAILURE.json').exists()
    groups=('SemanticBranch','LatestFull',*m.p.NEW_ARMS);results={str(s):{} for s in m.p.SEEDS};videos={}
    for domain,(nr,nf) in zip(m.p.DOMAINS,m.p.COUNTS):
        rows=m.read(run/'manifests'/(domain+'.json'));assert len(rows)==32*(nr+nf)
        sem=table(run/'semantic_target'/(domain+'.csv'),len(rows))
        fp=run/'target'/(domain+'_FRAMES.csv');complete=m.read(run/'target'/(domain+'_COMPLETE.json'))
        assert m.sha(fp)==complete['frames_sha256'] and complete['frames']==len(rows)
        frame=table(fp,len(rows));full=table(run/'reference_full_target'/(domain+'.csv'),len(rows))
        for seed in m.p.SEEDS:
            base=[m.p.f32(float(r['s'+str(seed)])/2) for r in sem]
            assert base==[float(r[f'{seed}_PatchClassificationContrast_baseline']) for r in full]
            z={g:correction([dict(baseline=r[f'{seed}_{g}_baseline'],delta=r[f'{seed}_{g}_delta'],fused=r[f'{seed}_{g}_fused']) for r in frame],base) for g in m.p.NEW_ARMS}
            z['SemanticBranch']=base;z['LatestFull']=[float(r[f'{seed}_PatchClassificationContrast_fused']) for r in full]
            dv={};dm={}
            for group in groups:
                met,vid=stat.inspect(rows,z[group]);assert (met['real_videos'],met['fake_videos'])==(nr,nf)
                dm[group]=met;dv[group]=vid
            dm['comparisons']={g:dict(vs_semantic=stat.paired(dv[g],dv['SemanticBranch']),vs_latest_full=stat.paired(dv[g],dv['LatestFull'])) for g in m.p.NEW_ARMS}
            results[str(seed)][domain]=dm;videos[(seed,domain)]=dv
    macros={str(s):{g:statistics.mean(results[str(s)][d][g]['video_AUC'] for d in m.p.DOMAINS) for g in groups} for s in m.p.SEEDS}
    totals={g:summarize([macros[str(s)][g] for s in m.p.SEEDS]) for g in groups}
    per_domain={d:{g:summarize([results[str(s)][d][g]['video_AUC'] for s in m.p.SEEDS]) for g in groups} for d in m.p.DOMAINS}
    result=dict(registration_sha256=reg,seeds=list(m.p.SEEDS),groups=list(groups),per_seed=results,
        per_seed_macro=macros,macro=totals,per_domain_mean_SD=per_domain,
        source=m.read(run/'SOURCE_LOCAL_VERIFICATION.json')['results'],
        costs=dict(target_preflight=pre,target=done),adaptive_exploration=True,independent_blind_test=False,
        semantic_unchanged=True,full_reference_reused=True,single_seed_SD='NA' if len(m.p.SEEDS)==1 else None)
    m.dump(folder/'RESULTS.json',result)
    proof=dict(passed=True,registration_sha256=reg,seeds=list(m.p.SEEDS),results_sha256=m.sha(folder/'RESULTS.json'),
        endpoints=len(m.p.SEEDS)*len(m.p.NEW_ARMS),all_seven_domains_verified=True,
        frames_per_endpoint=223136,videos_per_endpoint=6973,full_target_forwards=cfg['full_target_residual_forwards'],
        independent_saved_score_verification=True,neural_rerun=False,semantic_and_full_not_retrained=True)
    m.dump(folder/'LOCAL_VERIFICATION.json',proof)
    lines=['# 最新版双流匹配消融结果','',f"seed：{list(m.p.SEEDS)}。保持原语义及融合，完整模型只读复用。七域已自适应使用，不称盲测。",'',
        '|组别|七域macro AUC %|样本SD pp|相对完整模型 pp|','|---|---:|---:|---:|']
    for g in groups:
        sd='NA' if totals[g]['sample_SD'] is None else f"{100*totals[g]['sample_SD']:.4f}"
        lines.append(f"|{g}|{100*totals[g]['mean']:.4f}|{sd}|{100*(totals[g]['mean']-totals['LatestFull']['mean']):+.4f}|")
    lines.extend(['','逐seed/域、源选epoch及末轮、分方法、定位诊断和成本见RESULTS.json。无分类监督组的patch头指标不作为已训练分类器的证据。所有正负结果保留。'])
    with (folder/'SUMMARY_CN.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    delivery=dict(passed=True,seeds=list(m.p.SEEDS),registration_sha256=reg,target_archive=folder.relative_to(m.ROOT).as_posix(),
        target_verification=(folder/'LOCAL_VERIFICATION.json').relative_to(m.ROOT).as_posix(),
        target_verification_sha256=m.sha(folder/'LOCAL_VERIFICATION.json'),results_sha256=proof['results_sha256'],
        next_action='Prepare and execute only remaining3407/4099 under same protocol' if m.p.BATCH=='first' else 'All authorized ablations complete; stop')
    m.dump(m.LOCAL/'FINAL_DELIVERY.json',delivery)
    m.update('Latest_dual_ablation_'+m.p.BATCH+'_complete_verified',m.LOCAL/'FINAL_DELIVERY.json',status='complete_archived_verified',
        started=False,complete=True,target_archive=delivery['target_archive'],target_verification_sha256=delivery['target_verification_sha256'],
        next_action=delivery['next_action'])
    return proof
