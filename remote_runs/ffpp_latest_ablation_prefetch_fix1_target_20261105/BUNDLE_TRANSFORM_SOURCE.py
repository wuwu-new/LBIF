"""Fail-closed loading-only transformations; never edit registered source code."""
import ast
from build_patch_region_full_20261105 import replace
from build_target_prefetch_20261105 import runner as old_runner, BENCHMARK


def runner(text):
    # The established sidecar transform is identical except for this experiment's
    # dynamic endpoint count and a smaller, preregistered 256-image benchmark.
    result = old_runner(text)
    benchmark = BENCHMARK.replace('val[:512]', 'val[:256]')
    benchmark = benchmark.replace('frames=512,model_forwards=1536',
        'frames=256,model_forwards=p.ENDPOINTS*256')
    benchmark = benchmark.replace('extra_frame_forwards=6144',
        'extra_frame_forwards=p.EXTRA_BENCHMARK_FORWARDS')
    benchmark = benchmark.replace("meter['acceleration_benchmark']==6144",
        "meter['acceleration_benchmark']==p.EXTRA_BENCHMARK_FORWARDS")
    benchmark = benchmark.replace('preflight_forwards=7008',
        "preflight_forwards=p.configuration()['target_preflight_forwards_per_invocation']")
    result = replace(result, BENCHMARK, benchmark)
    # The check before ABBA is still the unchanged original 1440 probe forwards.
    result = replace(result,
        "assert sum(meter.values())==p.configuration()['target_preflight_forwards_per_invocation']",
        "assert sum(meter.values())==p.configuration()['original_target_probe_forwards']")
    result = replace(result,
        "preflight_forwards=p.configuration()['target_preflight_forwards_per_invocation'],new_optimizer_updates=0)",
        "preflight_forwards=p.configuration()['original_target_probe_forwards'],new_optimizer_updates=0)")
    ast.parse(result)
    return result


def verifier(text):
    text = replace(text, 'import manage_latest_dual_ablation_20261105 as m',
        'import manage_latest_ablation_prefetch_20261105 as m')
    begin = text.index('def verify_source(')
    end = text.index('def verify_target(', begin)
    text = text[:begin]+"def verify_source(folder):\n    raise ValueError('Use original source verifier')\n\n\n"+text[end:]
    text = replace(text, "assert pre['network_frame_forwards']==expected",
        "assert pre['network_frame_forwards']==dict(**expected,acceleration_benchmark=m.p.EXTRA_BENCHMARK_FORWARDS)\n"
        "    assert pre['preflight_forwards']==cfg['target_preflight_forwards_per_invocation']\n"
        "    choice=m.read(run/'ACCELERATION_DECISION.json')\n"
        "    assert choice['passed'] and choice['max_output_error']<=2e-7\n"
        "    assert choice['extra_frame_forwards']==m.p.EXTRA_BENCHMARK_FORWARDS\n"
        "    assert choice['enable_prefetch']==(choice['speedup']>=1.05)\n"
        "    adoption=m.read(run/'SOURCE_ADOPTION.json')\n"
        "    assert adoption['source_registration_sha256']==m.p.SOURCE_REG\n"
        "    for name,digest in adoption['files'].items():assert m.sha(run/name)==digest\n"
        "    assert done['source_registration_sha256']==m.p.SOURCE_REG\n"
        "    assert done['source_adoption_sha256']==m.sha(run/'SOURCE_ADOPTION.json')\n"
        "    sourceproof=m.read(run/'SOURCE_LOCAL_VERIFICATION.json')\n"
        "    assert sourceproof['passed'] and sourceproof['registration_sha256']==m.p.SOURCE_REG\n"
        "    assert sourceproof['selection_lock_sha256']==adoption['selection_lock_sha256']")
    ast.parse(text)
    return text


def remaining_source_runner(text):
    """Preparation only: future source VAL/localization, not training updates.

    Activation needs a separate registration/preflight; not used by this sidecar.
    The exact checked_pixels.pixels retains source-size and byte-SHA assertions.
    """
    text = replace(text, 'import argparse',
        'import argparse\nfrom contextlib import closing\nfrom prefetch import ordered_batches')
    start = text.index('                            for start in range(0,22080,16):')
    end = text.index('            dump(run/\'SELECTED_ENDPOINT_LOCK.json\'', start)
    block = text[start:end]
    # Only the two validation/localization loops read pixels; source training
    # already uses its existing RAM array and remains byte-for-byte unchanged.
    needle = '                            for start in range(0,22080,16):'
    assert block.count(needle)==2
    lines = block.splitlines(keepends=True)
    out=[];inside=False
    for line in lines:
        if line.startswith(needle):
            inside=True
            out.append('                            with closing(ordered_batches(val,lambda rr:np.stack([pixels(r) for r in rr]),parallel=True)) as batches:\n')
            out.append('                                for start,chunk,arrays in batches:\n')
            continue
        if inside and line.strip() and len(line)-len(line.lstrip())<=28:
            inside=False
        if inside:
            line=line.replace('rows=val[start:start+16];xx=batch(rows)',
                'rows=chunk;xx=torch.from_numpy(arrays).cuda().float()/255')
            out.append('    '+line)
        else:out.append(line)
    result=text[:start]+''.join(out)+text[end:]
    ast.parse(result)
    return result
