"""Target-only fail-closed transformation; original frozen source is not edited."""
import ast
from build_patch_region_full_20261105 import replace


BENCHMARK = '''        if preflight:
            benchmark_rows=val[:512];baseline_results=None;trials=[];maximum=0.
            for mode in (False,True,True,False):
                torch.cuda.synchronize();began=time.monotonic();results=[]
                with closing(ordered_batches(benchmark_rows,lambda rr:np.stack([pixels(r) for r in rr]),parallel=mode)) as batches:
                    with torch.no_grad():
                        for start,chunk,arrays in batches:
                            xx=torch.from_numpy(arrays).cuda().float()/255;batch_results=[]
                            for (seed,arm),m in models.items():
                                bb=torch.tensor(valbase[seed][start:start+len(chunk)],device='cuda')
                                out=forward(m,xx,bb,'acceleration_benchmark')
                                batch_results.extend(out[k].cpu().tolist() for k in ('delta','logit','weight_mass','pooled_norm'))
                            results.append(batch_results)
                torch.cuda.synchronize();elapsed=time.monotonic()-began
                if baseline_results is None:baseline_results=results
                else:
                    for aa,bb in zip(baseline_results,results):
                        assert len(aa)==len(bb)
                        for a,b in zip(aa,bb):
                            assert len(a)==len(b)
                            maximum=max(maximum,max(abs(x-y) for x,y in zip(a,b)))
                    assert maximum<=2e-7
                trials.append(dict(parallel=mode,seconds=elapsed,frames=512,model_forwards=1536))
                guard.check()
            ratio=sum(t['seconds'] for t in trials if not t['parallel'])/sum(t['seconds'] for t in trials if t['parallel'])
            choice=dict(passed=True,trials=trials,speedup=ratio,enable_prefetch=ratio>=1.05,
                max_output_error=maximum,extra_frame_forwards=6144,new_optimizer_updates=0,
                source_registration_sha256=p.SOURCE_REG,selection_lock_sha256=adoption['selection_lock_sha256'])
            dump(run/'ACCELERATION_DECISION.json',choice)
            assert meter['acceleration_benchmark']==6144
            proof.update(preflight_forwards=7008,acceleration=choice)
        else:
            choice=read(run/'ACCELERATION_DECISION.json')
            assert choice['passed'] and choice['max_output_error']<=2e-7
            assert choice['source_registration_sha256']==p.SOURCE_REG
            assert choice['selection_lock_sha256']==adoption['selection_lock_sha256']
            assert choice['enable_prefetch']==(choice['speedup']>=1.05)
'''


def runner(text):
    text=replace(text,'import argparse','import argparse\nfrom contextlib import closing\nfrom prefetch import ordered_batches,target_arrays')
    # The sidecar has no retained/disposable training entrypoint or optimizer.
    begin=text.index('        if source and preflight:')
    end=text.index('        from PIL import Image',begin)
    text=text[:begin]+text[end:]
    text=replace(text,"def execute(run,stage):\n", "def execute(run,stage):\n    assert stage in ('target-preflight','target'),'Target-only sidecar'\n")
    text=replace(text,"done=exited(run,'SOURCE',registration)","adoption=read(run/'SOURCE_ADOPTION.json')\n            assert adoption['source_run']==p.SOURCE_RUN and adoption['source_registration_sha256']==p.SOURCE_REG\n            for name,digest in adoption['files'].items():assert sha(run/name)==digest\n            done=exited(run,'SOURCE',p.SOURCE_REG)")
    text=replace(text,"proof['passed'] and proof['registration_sha256']==registration", "proof['passed'] and proof['registration_sha256']==p.SOURCE_REG")
    text=replace(text,"719*endpoint['epoch'],registration]", "719*endpoint['epoch'],p.SOURCE_REG]")
    # Save actual source identity as metadata, alongside the sidecar execution identity.
    text=replace(text,'network_frame_forwards=dict(meter),resources=resources(run))',
        "network_frame_forwards=dict(meter),resources=resources(run),source_registration_sha256=p.SOURCE_REG,\n                source_adoption_sha256=sha(run/'SOURCE_ADOPTION.json'))")
    text=replace(text,"resources(run,p.configuration()['train_uint8_bytes']+p.GIB if source else 2*p.GIB)","resources(run,2*p.GIB)")
    text=replace(text,"        def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01,betas=(.9,.999),eps=1e-8)\n",'')
    text=replace(text,"rows=read(run/'manifests'/(domain+'.json'));sem=table(run/'semantic_target'/(domain+'.csv'),len(rows));xx=target_batch(rows[:16]);repeats[domain]={}",
        "rows=read(run/'manifests'/(domain+'.json'));sem=table(run/'semantic_target'/(domain+'.csv'),len(rows));xx=target_batch(rows[:16]);repeats[domain]={}\n                with closing(ordered_batches(rows[:16],lambda rr:target_arrays(rr,pixel_hashes))) as batches:\n                    _,_,arrays=next(batches);parallel_xx=torch.from_numpy(arrays).cuda().float()/255\n                assert torch.equal(xx,parallel_xx),'Changed target pixels'")
    text=replace(text,"b=forward(m,xx,bb,'target_repeat_probe')['logit']", "b=forward(m,parallel_xx,bb,'target_repeat_probe')['logit']")
    # Compare all four reported outputs, not only fused logits.
    text=replace(text,"a=forward(m,xx,bb,'target_repeat_probe')['logit'];b=forward(m,parallel_xx,bb,'target_repeat_probe')['logit']\n                    err=float((a-b).abs().max());assert err<=2e-7;repeats[domain][f'{seed}/{arm}']=err",
        "a=forward(m,xx,bb,'target_repeat_probe');b=forward(m,parallel_xx,bb,'target_repeat_probe')\n                    err=max(float((a[k]-b[k]).abs().max()) for k in ('logit','delta','weight_mass','pooled_norm'))\n                    assert err<=2e-7;repeats[domain][f'{seed}/{arm}']=err")
    text=replace(text,"        dump(run/(prefix+'_CONTRACT.json'),proof)",BENCHMARK+"        dump(run/(prefix+'_CONTRACT.json'),proof)")
    start=text.index('                    for start in range(0,len(rows),16):',text.index("evaluation=run/'target'"))
    end=text.index('                dump(evaluation/',start)
    block=text[start:end]
    block=replace(block,'                    for start in range(0,len(rows),16):',
        "                    with closing(ordered_batches(rows,lambda rr:target_arrays(rr,pixel_hashes),parallel=choice['enable_prefetch'])) as batches:\n                        for start,chunk,arrays in batches:")
    block=replace(block,'                        chunk=rows[start:start+16];xx=target_batch(chunk);values={}',
        '                        xx=torch.from_numpy(arrays).cuda().float()/255;values={}')
    lines=block.splitlines(keepends=True)
    # Existing loop body gains one indent under the closing context manager.
    block=''.join(line if i<2 else '    '+line for i,line in enumerate(lines))
    text=text[:start]+block+text[end:]
    text=replace(text,"choices=('source-preflight','source','target-preflight','target')", "choices=('target-preflight','target')")
    ast.parse(text)
    return text


def verifier(text):
    text=replace(text,'import manage_patch_region_full_20261105 as m','import manage_target_prefetch_20261105 as m')
    begin=text.index('def verify_source(')
    end=text.index('def family_intervals(',begin)
    text=text[:begin]+"def verify_source(folder):\n    raise ValueError('Target-only sidecar; original source verifier required')\n\n"+text[end:]
    text=replace(text,"pre['passed'] and pre['preflight_forwards']==864", "pre['passed'] and pre['preflight_forwards']==7008\n    assert pre['network_frame_forwards']==dict(source_reload_probe=192,target_repeat_probe=672,acceleration_benchmark=6144)\n    decision=m.read(run/'ACCELERATION_DECISION.json');assert decision['passed'] and decision['max_output_error']<=2e-7\n    assert decision['enable_prefetch']==(decision['speedup']>=1.05)\n    adoption=m.read(run/'SOURCE_ADOPTION.json')\n    assert adoption['source_registration_sha256']==m.p.SOURCE_REG\n    for name,digest in adoption['files'].items():assert m.sha(run/name)==digest\n    assert done['source_adoption_sha256']==m.sha(run/'SOURCE_ADOPTION.json')\n    assert done['source_registration_sha256']==m.p.SOURCE_REG")
    text=replace(text,"sourceproof['passed'] and sourceproof['registration_sha256']==reg", "sourceproof['passed'] and sourceproof['registration_sha256']==m.p.SOURCE_REG")
    text=replace(text,'target_network_frame_forwards=done[\'network_frame_forwards\'],no_new_semantic_execution=True,',
        "target_network_frame_forwards=done['network_frame_forwards'],acceleration=decision,\n            extra_benchmark_forwards=6144,no_new_semantic_execution=True,")
    text=replace(text,"ap.add_argument('stage',choices=('source','target'))", "ap.add_argument('stage',choices=('target',))")
    ast.parse(text)
    return text
