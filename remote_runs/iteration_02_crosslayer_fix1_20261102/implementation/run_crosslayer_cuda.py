"""Fresh three-arm source experiment, remote CUDA only, immutable stage directories."""
import argparse
from collections import defaultdict
import fcntl
import gc
import hashlib
from pathlib import Path
import sys
import time
import crosslayer_plan as plan
from crosslayer_cache import build_val, digest_read, isolated_rng, train_cache
from gpu_identity_guard import OwnedCudaGuard
from run_multiscale_cuda import dump, resources, stamp, state

ROOT = Path(plan.REMOTE_ROOT)


def verify(run):
    lock = plan.read(run / 'REGISTRATION.json')
    plan.require(lock['registration_sha256'] == plan.fingerprint({k: v for k, v in lock.items() if k != 'registration_sha256'}), 'Registration changed')
    plan.require(lock['configuration'] == plan.configuration(), 'Protocol changed')
    for name, sha in lock['bundle_sha256'].items():
        plan.require(digest_read(run / name) == sha, 'Bundle changed: ' + name)
    for name, sha in plan.read(run / 'EXTERNAL_LOCK.json').items():
        plan.require(digest_read(name) == sha, 'Dependency changed: ' + name)
    train, val = [plan.read(run / 'inputs' / (name + '.json')) for name in ('fit', 'source_val')]
    plan.require(plan.verify_split(train, val) == plan.read(run / 'SPLIT_PROOF.json'), 'Split proof changed')
    plan.require(len({r['family_id'] for r in train}) == 291, 'TRAIN family count changed')
    return lock, train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    prefix = 'PREFLIGHT' if args.preflight_only else 'TRAIN'
    plan.require(run == ROOT / plan.RUN_NAME, 'Wrong run root')
    plan.require(not any((run / n).exists() for n in (prefix + '_STARTED.json', prefix + '_FAILURE.json', 'COMPLETE.json')), 'Existing stage preserved')
    with (ROOT / '.gpu_exclusive.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard = OwnedCudaGuard(on_bind=lambda row: print('CUDA_OWNER', row, flush=True))
        guard.check()
        lock, train, val = verify(run)
        if not args.preflight_only:
            probe = plan.read(run / 'PREFLIGHT.json')
            plan.require(probe['passed'] and probe['registration_sha256'] == lock['registration_sha256'] and
                         not (run / 'PREFLIGHT_FAILURE.json').exists(), 'Missing exact CUDA preflight')
        resources(run, 2 * plan.GIB, plan.cache_estimate()['added_val_bytes'] + 2 * plan.GIB)
        dump(run / (prefix + '_STARTED.json'), dict(time=stamp(), registration_sha256=lock['registration_sha256']))
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import WeightedRandomSampler
        plan.require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        sys.path.insert(0, str(run / 'implementation/v5'))
        sys.path.append(str(ROOT))
        import gpu_pipeline as pipeline
        import full_source_v2 as compact
        import joint_search_v4b as joint
        from multiscale_cuda import upgrade
        from crosslayer_cuda import CrossLayerDual, encode, final_tokens, route_probe
        from crosslayer_cache import DiskLayers
        from disk_val_cache import DiskArray, DiskTokens

        def cuda_state_of(path):
            saved = torch.load(path, map_location='cuda', weights_only=False)
            return saved, saved.get('ema_state') or saved['model_state']

        pipeline.state_of = cuda_state_of
        config = plan.read(run / 'V5_CONFIG.json')
        plan.require((config['batch_size'], config['workers'], config['epochs'], config['dual_lr'], config['weight_decay'],
                      config['residual_aux_weight'], config['consistency_weight']) == (16, 2, 10, .0003, .01, .1, .05), 'Training constants changed')
        spec = {**config['model'], **joint.ARCHITECTURES['IR_LayerScale']}
        anchor = compact.CompactSemantic(config['semantic_checkpoint']).cuda().eval()
        guard.check()
        encoder = compact.CompactEncoder(config)
        guard.check()
        plan.require(encoder.clip.config.num_hidden_layers == 24, 'CLIP layer count changed')
        transformers_source = Path(sys.modules[encoder.clip.__class__.__module__].__file__).resolve()
        plan.require(digest_read(transformers_source) == plan.CLIP_SOURCE_SHA, 'Installed CLIP implementation changed')

        def make(arm):
            pipeline.seed_all(plan.SEED)
            base = upgrade(joint.JointDual(joint.SemanticExit(anchor), spec).cuda(), 'ShallowMultiScale')
            return CrossLayerDual(base, arm).cuda()

        def tensor_hash(values):
            value = hashlib.sha256()
            for key, tensor in sorted(values.items()):
                value.update(key.encode())
                value.update(tensor.detach().contiguous().cpu().numpy().tobytes())
            return value.hexdigest()

        old_folder = ROOT / plan.OLD_VAL_RUN / 'VAL_DISK'
        old_tokens = DiskTokens(old_folder, len(val))
        old_rgb = DiskArray(old_folder / 'rgb.bin', len(val), (3, 224, 224), 'uint8', 1)
        rng_before = (torch.get_rng_state(), torch.cuda.get_rng_state())
        probe_rows = [train[index * len(train) // 16] for index in range(16)]
        with isolated_rng():
            rgb = torch.stack([pipeline.Pixels(probe_rows)[index] for index in range(16)])
            augmented_rgb = torch.stack([pipeline.Pixels(probe_rows, True)[index] for index in range(16)])
            features = encode(encoder, rgb)
            original = compact.compact_encode(encoder, rgb)
            plan.require(torch.equal(final_tokens(features), original), 'New final-layer/logit encoding differs from original')
            for source_rgb in (augmented_rgb, old_rgb[:16].float() / 255):
                fresh = encode(encoder, source_rgb)
                prior = compact.compact_encode(encoder, source_rgb)
                plan.require(torch.equal(final_tokens(fresh), prior), 'Augmented/VAL encoding changed historical final path')
                guard.check()
        plan.require(torch.equal(rng_before[0], torch.get_rng_state()) and torch.equal(rng_before[1], torch.cuda.get_rng_state()), 'Cache CUDA RNG changed')
        rgb_cuda = (rgb * 255).round().to(torch.uint8).cuda().float() / 255
        initializations, counts, audits = {}, {}, {}
        common = {}
        for arm in plan.ARMS:
            model = make(arm).eval()
            for name, module in model.named_modules():
                if name.startswith('base.semantic.'):
                    continue
                if isinstance(module, nn.modules.dropout._DropoutNd):
                    plan.require(module.p == 0, 'Unexpected stochastic dropout')
                if isinstance(module, nn.MultiheadAttention):
                    plan.require(module.dropout == 0, 'Unexpected attention dropout')
                plan.require(not isinstance(module, nn.modules.batchnorm._BatchNorm), 'Unexpected stateful BatchNorm')
            before = tensor_hash(model.state_dict())
            audits[arm] = route_probe(model, rgb_cuda, features)
            y = torch.tensor([r['label'] for r in probe_rows], device='cuda', dtype=torch.float32)
            result = model(rgb_cuda, features)
            loss = F.binary_cross_entropy_with_logits(result['logit'], y) + .1 * F.binary_cross_entropy_with_logits(result['residual_logit'], y)
            targets = [model.base.encoder.adapter.weight, model.base.correction[-1].weight]
            gradients = torch.autograd.grad(loss, targets)
            plan.require(all(bool(g.abs().sum() > 0) for g in gradients), 'Dead encoder adapter/correction preflight gradient')
            plan.require(torch.equal(result['logit'], features[:, 0, 0]), 'Initial zero correction does not match semantic')
            plan.require(all(not v.requires_grad for v in model.base.semantic.parameters()), 'Historical semantic unfrozen')
            plan.require(before == tensor_hash(model.state_dict()), 'Preflight mutated weights')
            initializations[arm] = before
            common[arm] = tensor_hash({key: value for key, value in model.state_dict().items()
                                      if not key.startswith(('base.front.', 'token_input.'))})
            counts[arm] = sum(v.numel() for v in model.parameters() if v.requires_grad)
            del model, result, loss, targets, gradients
            gc.collect()
            torch.cuda.empty_cache()
            guard.check()
        plan.require(initializations[plan.ARMS[0]] == initializations[plan.ARMS[1]], 'Full arms initialization differs')
        plan.require(len(set(common.values())) == 1, 'Shared tensors differ in initialization')
        plan.require(counts[plan.ARMS[0]] == counts[plan.ARMS[1]] and counts[plan.ARMS[2]] - counts[plan.ARMS[1]] == 2272, 'Capacity control parameter mismatch')
        guard.check()
        check = dict(time=stamp(), passed=True, registration_sha256=lock['registration_sha256'], models_cuda_only=True,
                     owned_nvml_pid=guard.owned_nvml_pid, container_pid=__import__('os').getpid(),
                     owned_guard_bound_and_checked=True, original_failed_preflight_preserved=True,
                     final_layer_and_complete_semantic_exact=True, original_compact_probe_views=['clean', 'augmented', 'source_VAL'],
                     full_arm_initialization_equal=True, common_tensor_sha256=common, model_initialization_sha256=initializations,
                     parameters=counts, route_and_image_audits=audits, optimizer_updates=0,
                     original_independent_replay_passed=False, shared_training_prefix=False, resources=resources(run))
        del rgb, augmented_rgb, features, original, source_rgb, fresh, prior, rgb_cuda, y, encoder
        gc.collect()
        torch.cuda.empty_cache()
        if args.preflight_only:
            dump(run / 'PREFLIGHT.json', check)
            state(run, 'cuda_preflight_passed', checks=check)
            return
        dump(run / 'RUNTIME_PREFLIGHT.json', check)

        encoder = compact.CompactEncoder(config)
        state(run, 'source_VAL_early_layer_encoding', done=0, total=len(val))
        caches = {'source_val': build_val(encoder, val, config, run, old_folder, guard, resources, dump)}
        resources(run, plan.cache_estimate()['two_train_views_bytes'] + 2 * plan.GIB)
        for name, augmented in (('fit', False), ('fit_aug', True)):
            state(run, 'source_TRAIN_cache', part=name, done=0, total=len(train))
            caches[name] = train_cache(encoder, train, config, augmented, guard, resources, run)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        labels = torch.tensor([row['label'] for row in train], device='cuda', dtype=torch.float32)
        groups = defaultdict(list)
        for index, row in enumerate(val):
            groups[row['video_id']].append(index)
        ordered = [groups[key] for key in sorted(groups)]
        plan.require(len(ordered) == 690 and all(len(ids) == 32 for ids in ordered), 'Source video frame coverage changed')
        reference = {r['video_id']: r for r in plan.read(run / 'BASELINE_SOURCE_VIDEOS.json')}

        @torch.no_grad()
        def predict(model):
            model.eval()
            probabilities, semantics = [], []
            for start in range(0, len(val), 16):
                result = model(caches['source_val']['rgb'][start:start+16].cuda().float()/255,
                               caches['source_val']['tokens'][start:start+16].cuda().float())
                plan.require(all(bool(torch.isfinite(result[key]).all()) for key in ('logit', 'delta', 'residual_logit')), 'Nonfinite source output')
                probabilities.extend(result['logit'].sigmoid().cpu().tolist())
                semantics.extend(result['semantic_logit'].sigmoid().cpu().tolist())
                if start % 2048 == 0:
                    guard.check()
                    resources(run)
            scores = torch.tensor([[probabilities[i] for i in ids] for ids in ordered], device='cuda', dtype=torch.float32).mean(-1).cpu().tolist()
            bases = torch.tensor([[semantics[i] for i in ids] for ids in ordered], device='cuda', dtype=torch.float32).mean(-1).cpu().tolist()
            return [dict(video_id=val[ids[0]]['video_id'], family_id=val[ids[0]]['family_id'], label=val[ids[0]]['label'],
                         frames=32, score=scores[j], semantic=bases[j]) for j, ids in enumerate(ordered)]

        checkpoints_audit, endpoint_columns, selections = [], {}, {}
        for arm in plan.ARMS:
            folder = run / 'arms' / arm
            folder.mkdir(parents=True)
            model = make(arm)
            plan.require(tensor_hash(model.state_dict()) == initializations[arm], 'Training initialization differs from preflight')
            params = [v for v in model.parameters() if v.requires_grad]
            filters = [model.base.front.delta] if arm != 'CrossLayerNoImage' else []
            filter_ids = {id(value) for value in filters}
            optimizer = torch.optim.AdamW([{'params': [v for v in params if id(v) not in filter_ids], 'lr': config['dual_lr']},
                                          {'params': filters, 'lr': config['dual_lr'] * config['filter_lr_multiplier']}], weight_decay=config['weight_decay'])
            records = []
            for epoch in range(11):
                started = time.monotonic()
                state(run, 'training' if epoch else 'initial_evaluation', arm=arm, epoch=epoch)
                updates, total, audits, sampled = 0, 0., [], []
                if epoch:
                    model.train()
                    sampled = list(WeightedRandomSampler(joint.sampler_weights(train), len(train), True,
                                   generator=torch.Generator().manual_seed(plan.SEED + 10000 * epoch)))
                    for start in range(0, len(sampled), 16):
                        ids = sampled[start:start+16]
                        optimizer.zero_grad(set_to_none=True)
                        outputs, losses = [], []
                        for part in ('fit', 'fit_aug'):
                            result = model(caches[part]['rgb'][ids].cuda().float()/255, caches[part]['tokens'][ids].cuda().float())
                            outputs.append(result)
                            losses.append(F.binary_cross_entropy_with_logits(result['logit'], labels[ids]) + .1 * F.binary_cross_entropy_with_logits(result['residual_logit'], labels[ids]))
                        loss = torch.stack(losses).mean() + .05 * F.mse_loss(outputs[0]['delta'], outputs[1]['delta'])
                        plan.require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                        loss.backward()
                        if updates in (0, 350, 718):
                            gradients = {name: float(value.grad.norm()) if value.grad is not None else 0.
                                         for name, value in (('adapter', model.base.encoder.adapter.weight), ('correction', model.base.correction[-1].weight), ('readout', model.readout.project.weight))}
                            plan.require(gradients['adapter'] > 0 and gradients['correction'] > 0 and
                                         (not (epoch > 1 or updates > 0) or gradients['readout'] > 0), 'Dead actual gradient path')
                            plan.require(all(v.grad is None for v in model.base.semantic.parameters()), 'Semantic gradient leakage')
                            audits.append(dict(step=updates, gradients=gradients, semantic_frozen=True))
                        nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                        optimizer.step()
                        updates += 1
                        total += float(loss.detach())
                        if updates % 100 == 0:
                            guard.check()
                            state(run, 'training', arm=arm, epoch=epoch, updates=updates, total_updates=719, resources=resources(run))
                    outputs = losses = result = loss = None
                    steps = [float(value['step']) for value in optimizer.state.values() if 'step' in value]
                    plan.require(updates == 719 and steps and all(s == 719*epoch for s in steps), 'Optimizer step budget changed')
                videos = predict(model)
                if epoch == 0:
                    error = max(abs(r['score'] - reference[r['video_id']]['scores']['SemanticAnchor']) for r in videos)
                    plan.require(error < 2e-7 and abs(plan.auc([r['label'] for r in videos], [r['score'] for r in videos]) - .9741782188615837) < 1e-10,
                                 'Complete initial semantic reproduction failed')
                dump(folder / f'epoch_{epoch:02}_VIDEOS.json', videos)
                checkpoint = folder / f'epoch_{epoch:02}.pt'
                resources(run, additional_disk=48*1024**2)
                with checkpoint.open('xb') as handle:
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), arm=arm, epoch=epoch, seed=2027,
                                    spec=spec, registration_sha256=lock['registration_sha256']), handle)
                saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
                identity = [saved[key] for key in ('arm', 'epoch', 'seed', 'registration_sha256')]
                model_sha = tensor_hash(model.state_dict())
                plan.require(identity == [arm, epoch, 2027, lock['registration_sha256']] and tensor_hash(saved['model']) == model_sha, 'Saved CUDA checkpoint identity mismatch')
                saved_steps = [float(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v]
                plan.require((not saved_steps and epoch == 0) or (saved_steps and all(s == 719*epoch for s in saved_steps)), 'Saved AdamW steps mismatch')
                checkpoints_audit.append(dict(identity=identity, model_tensor_sha256=model_sha, optimizer_steps=719*epoch, passed=True))
                del saved
                record = dict(arm=arm, epoch=epoch, seed=2027, updates=updates, optimizer_steps=epoch*719,
                              video_auc=plan.auc([r['label'] for r in videos], [r['score'] for r in videos]),
                              checkpoint=checkpoint.name, checkpoint_sha256=digest_read(checkpoint), model_tensor_sha256=model_sha,
                              video_scores=f'epoch_{epoch:02}_VIDEOS.json', video_scores_sha256=digest_read(folder / f'epoch_{epoch:02}_VIDEOS.json'),
                              slot_sampling_sha256=plan.fingerprint(sampled) if epoch else None, mapping_sha256=plan.fingerprint(train),
                              gradient_audits=audits, loss_mean=total/max(updates, 1), seconds=time.monotonic()-started)
                dump(folder / f'epoch_{epoch:02}.json', record)
                records.append(record)
                guard.check()
                state(run, 'epoch_completed', arm=arm, epoch=epoch, auc=record['video_auc'], resources=resources(run))
            best = max(records, key=lambda row: (row['video_auc'], -row['epoch']))
            dump(folder / 'SELECTION.json', best)
            selections[arm] = best
            endpoint_columns[arm] = videos
            params.clear()
            filters.clear()
            model = optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
        schedules = [[plan.read(run / 'arms' / arm / f'epoch_{e:02}.json')['slot_sampling_sha256'] for e in range(1, 11)] for arm in plan.ARMS]
        plan.require(all(s == schedules[0] for s in schedules), 'Three-arm slot sampling mismatch')
        dump(run / 'CHECKPOINT_CUDA_AUDIT.json', dict(passed=True, models_cuda_only=True, records=checkpoints_audit))
        dump(run / 'SELECTION_LOCK.json', dict(primary_endpoint=dict(epoch=10, g=1), secondary_selections=selections, target_selection=False))
        caches = labels = None
        gc.collect()
        torch.cuda.empty_cache()
        state(run, 'source_statistics')
        summary = plan.source_summary(endpoint_columns)
        (run / 'evaluation').mkdir()
        dump(run / 'evaluation/ENDPOINT_COLUMNS.json', endpoint_columns)
        report = dict(time=stamp(), iteration=2, source=summary, source_goal95_met={a: summary['aucs'][a] >= .95 for a in plan.ARMS},
                      seed=2027, seed_standard_deviation=None, shared_semantic=True, shared_training_prefix=False,
                      original_independent_replay_passed=False, per_arm_updates=7190, physical_new_updates=21570,
                      models_cuda_only=True, target_evaluation_required_even_if_source_negative=True,
                      target_inference=False, source_test_access=False, target_training=False, promotion=False,
                      registration_sha256=lock['registration_sha256'])
        dump(run / 'FINAL_REPORT.json', report)
        dump(run / 'COMPLETE.json', dict(time=stamp(), final_report_sha256=digest_read(run / 'FINAL_REPORT.json')))
        state(run, 'completed', report=report, resources=resources(run))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        run = ROOT / plan.RUN_NAME
        prefix = 'PREFLIGHT' if '--preflight-only' in sys.argv else 'TRAIN'
        if run.exists() and not (run / (prefix + '_FAILURE.json')).exists():
            dump(run / (prefix + '_FAILURE.json'), dict(time=stamp(), error=repr(error), history_preserved=True))
        if run.exists():
            state(run, 'failed', stage_kind=prefix, error=repr(error))
        raise
