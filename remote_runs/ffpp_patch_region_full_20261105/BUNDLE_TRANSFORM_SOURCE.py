"""Fail-closed transformations of a previously verified finite CUDA runner."""
import ast
import re


def replace(text, before, after, count=1):
    assert text.count(before) == count, ('Changed transformation context', before, text.count(before))
    return text.replace(before, after)


PREFLIGHT = '''            initial={};shared_initial={};checks={};oldinitial=read(run/'REUSED_INITIALIZATION.json')['initialization_sha256']
            real_index=next(i for i,r in enumerate(train) if r['label']==0 and r['mask_valid'])
            fake_indices=[i for i,r in enumerate(train) if r['label']==1 and r['mask_valid']][:64]
            fake_index=None
            for i in fake_indices:
                mask=torch.from_numpy(mask_pixels(train[i])[None,None]).cuda().float()/255
                target,eligible,_=patch_targets(mask,torch.ones(1,dtype=torch.bool,device='cuda'),torch.ones(1,device='cuda'))
                positives=int(target.sum())
                if 2<=positives<=254:fake_index=i;break
            assert fake_index is not None,'No eligible mixed fake mask among64 frozen TRAIN candidates'
            ids=[real_index,fake_index];realxx=batch([train[i] for i in ids])
            realmm=torch.from_numpy(np.stack([mask_pixels(train[i])[None] for i in ids])).cuda().float()/255
            yy=torch.tensor([0.,1.],device='cuda');valid=torch.ones(2,dtype=torch.bool,device='cuda')
            partial=torch.tensor([True,False],device='cuda');altered=realmm.clone();altered[1]=float('nan')
            torch.manual_seed(101);xx=torch.rand(2,3,256,256,device='cuda')
            synthetic_mask=torch.zeros(2,1,256,256,device='cuda');synthetic_mask[1,:,64:192,64:192]=1
            for seed in p.SEEDS:
                initial[str(seed)]={};shared_initial[str(seed)]={};checks[str(seed)]={}
                semtrain=table(run/'semantic_train'/f'{seed}.csv',11492)
                for i in ids:
                    assert (semtrain[i]['video_id'],int(semtrain[i]['source_frame_number']))==p.identity(train[i])
                    assert float(semtrain[i]['baseline'])==p.f32(float(semtrain[i]['semantic'])/2)
                bb=torch.tensor([float(semtrain[i]['baseline']) for i in ids],device='cuda',requires_grad=True)
                for arm in p.NEW_ARMS:
                    m=make(seed,arm);digest=tensor_sha(m.state_dict());initial[str(seed)][arm]=digest
                    shared=tensor_sha({k:v for k,v in m.state_dict().items() if not k.startswith('patch_classifier.')})
                    assert shared==oldinitial[str(seed)],'Historical shared constructor changed'
                    shared_initial[str(seed)][arm]=shared
                    out=forward(m,xx,bb,'synthetic_zero_head')
                    assert torch.equal(out['logit'],bb) and not bool(out['delta'].any())
                    assert float(m.error_map(torch.ones_like(xx)).abs().max())<=2e-7
                    out=forward(m,realxx,bb,'real_TRAIN_gradient')
                    terms=objective(m,out,yy,realmm,valid,split='TRAIN')
                    params=(m.encoder[0][0].weight,m.patch_classifier.weight,m.decoder[2].weight,bb)
                    pg=torch.autograd.grad(terms['patch_classification'],params,allow_unused=True,retain_graph=True)
                    assert all(g is not None and float(g.abs().sum())>0 for g in pg[:2]) and pg[2:]==(None,None)
                    cg=torch.autograd.grad(terms['patch_contrast'],params,allow_unused=True,retain_graph=True)
                    assert cg[0] is not None and float(cg[0].abs().sum())>0 and cg[1:]==(None,None,None)
                    bg=torch.autograd.grad(terms['boundary'],list(m.decoder.parameters()),retain_graph=True)
                    assert all(g is not None for g in bg) and any(float(g.abs().sum())>0 for g in bg)
                    assert torch.autograd.grad(terms['total'],bb,allow_unused=True,retain_graph=True)[0] is None
                    a=objective(m,out,yy,realmm,partial,split='TRAIN')
                    b=objective(m,out,yy,altered,partial,split='TRAIN')
                    assert all(torch.equal(a[k],b[k]) for k in a)
                    for split in ('VAL','TEST','target'):
                        try:objective(m,out,yy,realmm,valid,split=split)
                        except ValueError:pass
                        else:raise AssertionError('NonTRAIN supervision accepted')
                    del m,out,terms,pg,cg,bg,a,b
                    m=make(seed,arm);opt=optimizer(m)
                    for _ in range(2):
                        opt.zero_grad(set_to_none=True);out=forward(m,xx,bb,'disposable_synthetic')
                        terms=objective(m,out,yy,synthetic_mask,valid,split='TRAIN')
                        assert bool(torch.isfinite(terms['total']));terms['total'].backward();assert bb.grad is None
                        assert all(t.grad is not None for t in m.parameters())
                        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                    assert len(opt.state)==len(list(m.parameters())) and all(int(v['step'])==2 for v in opt.state.values())
                    checks[str(seed)][arm]=dict(passed=True,real_TRAIN_patch_encoder_gradient=True,
                        real_TRAIN_patch_classifier_gradient=True,real_TRAIN_contrast_encoder_gradient=True,
                        semantic_gradient=False,invalid_mask_invariant=True,nonTRAIN_supervision_rejected=True,
                        disposable_updates=2,allocated_parameters=sum(t.numel() for t in m.parameters()))
                    del m,opt,out,terms;gc.collect();torch.cuda.empty_cache();guard.check()
            assert len({v[p.NEW_ARMS[0]] for v in initial.values()})==3
            assert dict(meter)==dict(historical_full_replay=192,synthetic_zero_head=6,
                real_TRAIN_gradient=6,disposable_synthetic=12)
            dump(run/'INITIALIZATION.json',initial)
            finish(dict(passed=True,historical_full_max_errors=anchor_errors,initialization_sha256=initial,
                shared_initialization_sha256=shared_initial,gradient_checks=checks,
                real_TRAIN_fixture_indices=ids,real_TRAIN_fixture_identities=[p.identity(train[i]) for i in ids],
                disposable_optimizer_updates=6,new_retained_updates=0,
                preflight_frame_forwards=216,new_semantic_execution=False));return
'''


def runner(text):
    text = replace(text, 'from model_cuda import ComponentDelta,objective',
                   'from model_cuda import PatchRegionDelta,objective,patch_targets')
    text = replace(text, "with torch.device('cuda'):m=ComponentDelta(arm)",
                   "with torch.device('cuda'):m=PatchRegionDelta(arm)")
    text = replace(text, 'out=m(xx,bb,enabled=True);meter[tag]+=len(bb)',
        "kwargs={'return_patch':tag in ('retained_training','selected_localization','real_TRAIN_gradient','disposable_synthetic')} if isinstance(m,PatchRegionDelta) else {}\n            out=m(xx,bb,enabled=True,**kwargs);meter[tag]+=len(bb)")
    begin = text.index('            initial={};checks={};oldinitial=')
    end = text.index('\n        if source:\n', begin)
    text = text[:begin]+PREFLIGHT+text[end:]
    text = replace(text, "contract['disposable_optimizer_updates']==12", "contract['disposable_optimizer_updates']==6")
    text = replace(text, 'totals=[0.,0.,0.];seen=set()', 'totals=[0.]*5;coverage=[0.,0.,0.];seen=set()')
    text = replace(text, "out=forward(m,xx,bb,'retained_training');loss,bc,aux,penalty=objective(m,out,yy,mm,valid)",
        "out=forward(m,xx,bb,'retained_training');terms=objective(m,out,yy,mm,valid,split='TRAIN')\n                            loss=terms['total'];bc=terms['image'];aux=terms['boundary'];penalty=terms['regularizer']")
    text = replace(text, 'for j,t in enumerate((bc,aux,penalty)):totals[j]+=float(t.detach())',
        "for j,key in enumerate(('image','boundary','regularizer','patch_classification','patch_contrast')):totals[j]+=float(terms[key].detach())\n                            for j,key in enumerate(('eligible_patches','eligible_fake_patches','mixed_patches')):coverage[j]+=float(terms[key])")
    text = replace(text, 'unique_frames=len(seen),loss_means=[x/719 for x in totals]',
        "unique_frames=len(seen),loss_means=[x/719 for x in totals],\n                            loss_names=['image','boundary','regularizer','patch_classification','patch_contrast'],\n                            patch_coverage_totals=dict(zip(('eligible','fake','mixed'),coverage)),patch_weights=p.variant(arm))")
    # The replacement above includes the closing dict parenthesis, not dump's.
    text = replace(text, 'patch_weights=p.variant(arm))))', 'patch_weights=p.variant(arm)))') if 'patch_weights=p.variant(arm))))' in text else text
    text = replace(text, 'response_sum=0.))', 'response_sum=0.,patch_TP=0,patch_FP=0,patch_FN=0,patch_TN=0,patch_mixed=0))')
    text = replace(text, "gt=boundary_target(torch.from_numpy(mask_pixels(r)[None,None]).cuda().float()/255,replay.kernel)>.1",
        "mask=torch.from_numpy(mask_pixels(r)[None,None]).cuda().float()/255\n                                    gt=boundary_target(mask,replay.kernel)>.1\n                                    pt,_,cov=patch_targets(mask,torch.ones(1,dtype=torch.bool,device='cuda'),torch.tensor([r['label']],device='cuda'))\n                                    pp=out['patch_logit'][j:j+1].sigmoid()>=.5;truth=pt>.5")
    text = replace(text, "s['response_sum']+=float(out['boundary_logit'][j].sigmoid().mean())",
        "s['response_sum']+=float(out['boundary_logit'][j].sigmoid().mean())\n                                    s['patch_TP']+=int((pp&truth).sum());s['patch_FP']+=int((pp&~truth).sum())\n                                    s['patch_FN']+=int((~pp&truth).sum());s['patch_TN']+=int((~pp&~truth).sum())\n                                    s['patch_mixed']+=int(((cov>0)&(cov<1)).sum())")
    substitutions = [(690240,345120),(1324800,662400),(88320,66240),(384,192),
                     (43140,21570),(1728,864),(1338816,669408)]
    for before, after in substitutions:
        assert re.search(r'\b'+str(before)+r'\b', text), before
        text = re.sub(r'\b'+str(before)+r'\b', str(after), text)
    text = replace(text, 'residual_frame_forwards=6*len(rows)', 'residual_frame_forwards=3*len(rows)')
    ast.parse(text)
    return text


def verifier(text):
    text = replace(text, 'import manage_paper_components_20261104 as m', 'import manage_patch_region_full_20261105 as m')
    text = replace(text, "GROUPS=('SemanticRaw','SemanticHalf','UniformNoAux','UniformBoundary','BoundaryFull','RGBBoundary')",
        "GROUPS=('SemanticRaw','SemanticHalf','BoundaryFull','PatchClassificationContrast')")
    text = replace(text, "CONTRASTS=(('BoundaryFull','SemanticHalf'),('UniformNoAux','SemanticHalf'),('UniformBoundary','UniformNoAux'),('BoundaryFull','UniformBoundary'),('BoundaryFull','RGBBoundary'),('SemanticHalf','SemanticRaw'))",
        "CONTRASTS=(('PatchClassificationContrast','BoundaryFull'),('PatchClassificationContrast','SemanticHalf'),('BoundaryFull','SemanticHalf'),('SemanticHalf','SemanticRaw'))")
    text = replace(text, "pre['disposable_optimizer_updates']==18", "pre['disposable_optimizer_updates']==6")
    text = replace(text, "pre['preflight_frame_forwards']==264", "pre['preflight_frame_forwards']==216")
    text = replace(text, 'synthetic_zero_head=18,synthetic_gradient=18,disposable_synthetic=36',
                   'synthetic_zero_head=6,real_TRAIN_gradient=6,disposable_synthetic=12')
    text = replace(text, "assert pre['initialization_sha256'][str(seed)][arm]==m.read(run/'REUSED_INITIALIZATION.json')['initialization_sha256'][str(seed)]",
        "assert pre['shared_initialization_sha256'][str(seed)][arm]==m.read(run/'REUSED_INITIALIZATION.json')['initialization_sha256'][str(seed)]\n                assert len(exposure['loss_means'])==5 and exposure['patch_weights']==m.p.variant(arm)\n                assert exposure['patch_coverage_totals']['eligible']>exposure['patch_coverage_totals']['fake']>0")
    text = replace(text, "assert 0<=row['response_sum']<=row['frames']",
        "assert 0<=row['response_sum']<=row['frames']\n                    assert sum(row[k] for k in ('patch_TP','patch_FP','patch_FN','patch_TN'))==256*row['frames']")
    for before, after in [(64710,21570),(1035360,345120),(1987200,662400),(576,192),
                          (132480,66240),(2008224,669408),(2016,672),(2592,864)]:
        assert re.search(r'\b'+str(before)+r'\b', text)
        text=re.sub(r'\b'+str(before)+r'\b',str(after),text)
    text = text.replace('all90_epochs_verified','all30_epochs_verified').replace('all_nine_endpoints_seven_domains_verified','all_three_endpoints_seven_domains_verified')
    text = text.replace('six_groups_including_reused=True','four_groups_including_reused=True')
    text = text.replace('六组内部配对消融结果','真实分块分类加区域对比完整版本结果').replace('源六组结果','源四组结果')
    text = text.replace('Deliver all six groups including negative results','Deliver full patch variant and three reused controls including negative results')
    ast.parse(text)
    return text
