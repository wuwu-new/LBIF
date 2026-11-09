"""Fail-closed edits of the archived full runner/model; never edit the parent."""
import ast
import re


def replace(text, before, after, count=1):
    assert text.count(before) == count, ('Changed frozen context', before, text.count(before))
    return text.replace(before, after)


def model(text):
    text = replace(text, 'f0 = self.encoder[0](self.error_map(rgb))',
                   "front = rgb if cfg.variant(self.patch_variant)['input'] == 'RGB' else self.error_map(rgb)\n        f0 = self.encoder[0](front)")
    text = replace(text, "weight = F.interpolate(boundary.detach().sigmoid(), size=f2.shape[-2:], mode='area')",
                   "weight = (torch.ones_like(f2[:, :1]) if cfg.variant(self.patch_variant)['pooling'] == 'uniform'\n                  else F.interpolate(boundary.detach().sigmoid(), size=f2.shape[-2:], mode='area'))")
    ast.parse(text)
    return text


def patch_config(original):
    text = """# Identical patch geometry/objectives; only registered variant coefficients differ.
from original_patch_config import *
from plan import variant
VARIANTS = ('PatchClassificationContrast', 'RGBFull', 'UniformPoolFull',
            'NoPatchClassification', 'NoRegionContrast', 'NoPatchLearningMatched')
"""
    ast.parse(original)
    ast.parse(text)
    return text


FULL_REPLAY = '''            current_full_errors={};full_selected=read(run/'REFERENCE_FULL_ENDPOINTS.json')
            for seed in p.SEEDS:
                endpoint=full_selected[str(seed)]['PatchClassificationContrast']
                assert sha(endpoint['path'])==endpoint['sha256']
                saved=torch.load(endpoint['path'],map_location='cuda',weights_only=True)
                assert saved['registration_sha256']==p.PARENT_REG and saved['seed']==seed
                assert saved['arm']=='PatchClassificationContrast' and saved['updates']==719*endpoint['epoch']
                m=make(seed,'PatchClassificationContrast');m.load_state_dict(saved['model'],strict=True);m.eval()
                assert tensor_sha(m.state_dict())==endpoint['model_sha256']
                current_full_errors[str(seed)]=source_probe(m,seed,run/'reference_full_source'/f'{seed}.csv','latest_full_replay')
                del m,saved
'''


def runner(text):
    text = replace(text, "            initial={};shared_initial={};checks={};oldinitial=read(run/'REUSED_INITIALIZATION.json')['initialization_sha256']",
                   FULL_REPLAY + "            initial={};shared_initial={};checks={};oldinitial=read(run/'REUSED_INITIALIZATION.json')['initialization_sha256']\n            latest_initial=read(run/'REFERENCE_FULL_INITIALIZATION.json')")
    text = replace(text, "shared_initial[str(seed)][arm]=shared",
                   "shared_initial[str(seed)][arm]=shared\n                    assert digest==latest_initial[str(seed)]['PatchClassificationContrast'],'Full latest initialization changed'")
    text = replace(text, "                    a=objective(m,out,yy,realmm,partial,split='TRAIN')",
                   """                    weights=p.variant(arm)
                    expected=(terms['image']+.2*terms['boundary']+.01*terms['regularizer']+
                        weights['patch_classification']*terms['patch_classification']+
                        weights['patch_contrast']*terms['patch_contrast'])
                    assert torch.equal(terms['total'],expected),'Changed loss mixture'
                    total_patch=torch.autograd.grad(terms['total'],m.patch_classifier.weight,retain_graph=True)[0]
                    assert bool(total_patch.abs().sum()>0)==bool(weights['patch_classification'])
                    if arm=='UniformPoolFull':
                        assert torch.equal(out['weight_mass'],torch.ones_like(out['weight_mass']))
                        detached=torch.autograd.grad(terms['image'],m.decoder[2].weight,allow_unused=True,retain_graph=True)[0]
                        assert detached is None,'Uniform readout accidentally depends on decoder'
                    a=objective(m,out,yy,realmm,partial,split='TRAIN')""")
    text = replace(text, "disposable_updates=2,allocated_parameters=", "declared_variant=p.variant(arm),total_patch_gradient_enabled=bool(weights['patch_classification']),\n                        disposable_updates=2,allocated_parameters=")
    text = replace(text, "            assert len({v[p.NEW_ARMS[0]] for v in initial.values()})==3",
                   "            assert len({v[p.NEW_ARMS[0]] for v in initial.values()})==len(p.SEEDS)\n            assert all(len(set(v.values()))==1 for v in initial.values()),'Unpaired initialization'")
    text = replace(text, "            assert dict(meter)==dict(historical_full_replay=192,synthetic_zero_head=6,\n                real_TRAIN_gradient=6,disposable_synthetic=12)",
                   "            n=len(p.SEEDS)*len(p.NEW_ARMS)\n            assert dict(meter)==dict(historical_full_replay=len(p.SEEDS)*64,latest_full_replay=len(p.SEEDS)*64,\n                synthetic_zero_head=n*2,real_TRAIN_gradient=n*2,disposable_synthetic=n*4)")
    text = replace(text, 'finish(dict(passed=True,historical_full_max_errors=anchor_errors,initialization_sha256=initial,',
                   'finish(dict(passed=True,historical_full_max_errors=anchor_errors,latest_full_max_errors=current_full_errors,initialization_sha256=initial,')
    text = replace(text, 'disposable_optimizer_updates=6,new_retained_updates=0,',
                   'disposable_optimizer_updates=n*2,new_retained_updates=0,')
    text = replace(text, 'preflight_frame_forwards=216,new_semantic_execution=False)',
                   "preflight_frame_forwards=p.configuration()['source_preflight_total_forwards'],new_semantic_execution=False)")
    text = replace(text, "contract['disposable_optimizer_updates']==6", "contract['disposable_optimizer_updates']==p.configuration()['preflight_disposable_updates']")
    replacements = {
        345120: "p.configuration()['retained_training_frame_forwards']",
        662400: "p.configuration()['source_VAL_frame_forwards']",
        66240: "p.configuration()['selected_localization_forwards']",
        192: "p.configuration()['selected_reload_forwards']",
        21570: "p.configuration()['retained_updates']",
        864: "p.configuration()['target_preflight_forwards_per_invocation']",
        669408: "p.configuration()['full_target_residual_forwards']",
    }
    for number, expression in replacements.items():
        assert re.search(r'\b'+str(number)+r'\b', text), number
        text = re.sub(r'\b'+str(number)+r'\b', expression, text)
    text = replace(text, 'residual_frame_forwards=3*len(rows)', 'residual_frame_forwards=len(p.SEEDS)*len(p.NEW_ARMS)*len(rows)')
    ast.parse(text)
    return text
