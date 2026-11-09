"""Target-only acceleration: original source model identity is never rewritten."""
import importlib.util
from pathlib import Path

try:
    import source_plan as _original
except ModuleNotFoundError:
    path = Path(__file__).resolve().parent.parent/'patch_region_full_20261105/plan.py'
    spec = importlib.util.spec_from_file_location('prefetch_original_plan', path)
    _original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_original)

for name in dir(_original):
    if not name.startswith('_'):
        globals()[name] = getattr(_original, name)

SOURCE_RUN = 'ffpp_patch_region_full_20261105'
SOURCE_REG = '3c2bbc1cf7b3066ca7d81f212de9f0f33b08befca78a4caeba226ada941368ee'
PARENT, PARENT_REG = SOURCE_RUN, SOURCE_REG
RUN = 'ffpp_patch_region_prefetch_target_20261105'


def configuration():
    value = _original.configuration().copy()
    value.update(source_training_authorized=False, retained_updates=0,
        adopted_source_budget={k:value[k] for k in ('retained_updates','retained_training_frame_forwards','source_VAL_frame_forwards','selected_localization_forwards')},
        retained_training_frame_forwards=0, preflight_disposable_updates=0,
        source_VAL_frame_forwards=0, selected_localization_forwards=0,
        selected_reload_forwards=0, source_preflight_total_forwards=0,
        source_preflight_anchor_forwards=0, source_preflight_synthetic_forwards=0,
        source_preflight_real_TRAIN_forwards=0,
        source_run=SOURCE_RUN, source_registration_sha256=SOURCE_REG,
        target_only_sidecar=True, workers=2, pending_batches=2, batch=16,
        precision='unchanged FP32; no AMP, TF32, batch or CUDA stream change',
        CPU_benchmark='VAL first256, serial/prefetch/prefetch/serial; 1024 image reads; no CUDA',
        CUDA_benchmark='VAL first512, serial/prefetch/prefetch/serial, three endpoints;6144 forwards once',
        benchmark_min_speedup=1.05, equivalence_tolerance=2e-7,
        target_preflight_forwards_per_invocation=7008,
        target_preflight_total_forwards=7872,
        acceleration_extra_forwards=6144,
        full_target_residual_forwards=669408,
        selection_rule='enable prefetch iff all pixels/outputs agree and paired end-to-end speedup>=1.05; otherwise serial',
        benchmark_affects_model_selection=False, source_adoption_required=True,
        original_source_registration_preserved=True, CUDA_benchmark_retries=0,
        training_acceleration=False, no_new_monitor=True)
    return value
