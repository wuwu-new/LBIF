"""Read-only target sidecar for the five latest ablations, seed2027 first."""
import importlib.util
from pathlib import Path

try:
    import source_plan as _original
except ModuleNotFoundError:
    _path = Path(__file__).resolve().parent.parent/'latest_dual_ablation_20261105/plan.py'
    _spec = importlib.util.spec_from_file_location('latest_prefetch_source_plan', _path)
    _original = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_original)

for _name in dir(_original):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_original, _name)

SOURCE_RUN = 'ffpp_latest_dual_ablation_seed2027_20261105'
SOURCE_REG = 'ca52db99805866c45ac751ce7a1e11be07326cf06b9ffabc7912e8f426388b52'
PARENT, PARENT_REG = SOURCE_RUN, SOURCE_REG
RUN = 'ffpp_latest_ablation_prefetch_fix1_target_20261105'
ENDPOINTS = len(SEEDS)*len(NEW_ARMS)
EXTRA_BENCHMARK_FORWARDS = ENDPOINTS*256*4


def configuration():
    value = _original.configuration().copy()
    original_probe = value['target_preflight_forwards_per_invocation']
    value.update(adopted_source_budget={k:value[k] for k in (
        'retained_updates','retained_training_frame_forwards','source_VAL_frame_forwards',
        'selected_localization_forwards')},
        source_training_authorized=False, retained_updates=0,
        retained_training_frame_forwards=0, source_VAL_frame_forwards=0,
        selected_localization_forwards=0, source_preflight_total_forwards=0,
        preflight_disposable_updates=0, runtime_disposable_updates=0,
        source_run=SOURCE_RUN, source_registration_sha256=SOURCE_REG,
        target_only_sidecar=True, workers=2, pending_batches=2, batch_size=16,
        original_target_probe_forwards=original_probe,
        target_preflight_forwards_per_invocation=original_probe+EXTRA_BENCHMARK_FORWARDS,
        target_preflight_total_forwards=2*original_probe+EXTRA_BENCHMARK_FORWARDS,
        acceleration_extra_forwards=EXTRA_BENCHMARK_FORWARDS,
        CPU_benchmark='one ABBA, source VAL first256,1024 reads, nice19, no torch/CUDA',
        CUDA_benchmark='one ABBA, source VAL first256,5 endpoints,5120 forwards,0 updates',
        equivalence_tolerance=2e-7, benchmark_min_speedup=1.05,
        original_source_registration_preserved=True, source_adoption_required=True,
        selection_rule='pixels and four outputs equal; ABBA speedup>=1.05 else serial',
        CUDA_benchmark_retries=0, training_acceleration=False, no_new_monitor=True)
    return value
