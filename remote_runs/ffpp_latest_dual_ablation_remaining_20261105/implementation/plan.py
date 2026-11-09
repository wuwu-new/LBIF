"""Bounded matched ablations of the latest model; no semantic modification."""
import importlib.util
from pathlib import Path

try:
    import legacy_component_plan as old
except ModuleNotFoundError:
    path = Path(__file__).resolve().parent.parent/'paper_components_20261104/plan.py'
    spec = importlib.util.spec_from_file_location('latest_ablation_original_plan', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)

for name in dir(old):
    if not name.startswith('_'):
        globals()[name] = getattr(old, name)

BATCH = 'remaining'
PARENT = 'ffpp_patch_region_full_20261105'
PARENT_REG = '3c2bbc1cf7b3066ca7d81f212de9f0f33b08befca78a4caeba226ada941368ee'
REFERENCE_TARGET = 'ffpp_patch_region_prefetch_target_20261105'
REFERENCE_TARGET_REG = '707ee19e106a00bb5ef6d9e8c5b2c50f64d42402ef61ef96a482def6ebb0beb8'
NEW_ARMS = ('RGBFull', 'UniformPoolFull', 'NoPatchClassification', 'NoRegionContrast', 'NoPatchLearningMatched')
ALL_SEEDS = (2027, 3407, 4099)


def configure(batch):
    global BATCH, RUN, SEEDS
    assert batch in ('first', 'remaining')
    BATCH = batch
    SEEDS = (2027,) if batch == 'first' else (3407, 4099)
    RUN = 'ffpp_latest_dual_ablation_' + ('seed2027' if batch == 'first' else 'remaining') + '_20261105'


configure(BATCH)


def variant(arm):
    assert arm in NEW_ARMS or arm == 'PatchClassificationContrast'
    return dict(input='RGB' if arm == 'RGBFull' else 'signed_RGB_minus_Gaussian5_sigma1',
                pooling='uniform' if arm == 'UniformPoolFull' else 'predicted_boundary_cells_quarter_floor',
                boundary_coefficient=.2,
                patch_classification=0. if arm in ('NoPatchClassification', 'NoPatchLearningMatched') else .2,
                patch_contrast=0. if arm in ('NoRegionContrast', 'NoPatchLearningMatched') else .1,
                temperature=.5)


def configuration():
    value = old.configuration().copy()
    n = len(SEEDS)*len(NEW_ARMS)
    value.update(primary_question='leave-component-out ablations of the latest complete dual-stream method; semantic branch and fusion unchanged',
        batch=BATCH, new_arms={a:variant(a) for a in NEW_ARMS}, seeds=list(SEEDS),
        all_authorized_seeds=list(ALL_SEEDS), all_authorized_retained_updates=15*7190,
        retained_updates=n*7190, retained_training_frame_forwards=n*7190*16,
        preflight_disposable_updates=n*2, runtime_disposable_updates=0,
        source_VAL_frame_forwards=n*10*22080, selected_localization_forwards=n*22080,
        selected_reload_forwards=n*64, source_preflight_anchor_forwards=len(SEEDS)*128,
        source_preflight_synthetic_forwards=n*6, source_preflight_real_TRAIN_forwards=n*2,
        source_preflight_total_forwards=len(SEEDS)*128+n*8,
        target_preflight_forwards_per_invocation=n*(64+7*32),
        target_preflight_total_forwards=2*n*(64+7*32), full_target_residual_forwards=n*223136,
        architecture='unchanged latest full CNN/decoder/zero delta head/65-parameter auxiliary patch head; only declared ablations',
        semantic='exact same frozen enhanced per-seed CrossLayerNoImage weights/cache and s/2 readout as latest full model; NOT vanilla CLIP',
        fusion='unchanged f32(f32(s_seed/2).detach()+.5*tanh(head(pooled)))',
        patch_geometry='16x16 context-bearing CNN feature grid, authentic mask coverage strictly >0.1',
        patch_loss='within-image class-balanced BCE, image-balanced mean',
        contrast_loss='fake TRAIN only, both classes >=2 patches, no self-pairs, class/image-balanced SupCon',
        inference='no masks or patch head',
        initialization='full initial state identical to latest full model for same seed, including auxiliary head; all ablations share initialization',
        disabled_auxiliary_head='allocated equally; zero coefficient means no supervised gradient, not claimed effective capacity',
        preflight_current_full_replay=True, unchanged_neural_tolerance=2e-7,
        reused_arms=['SemanticBranch_exact_latest_baseline','LatestFull_PatchClassificationContrast'],
        prior_full_retained_updates=21570, prior_full_not_retrained=True,
        old_no_patch_not_reused_as_matched=True, old_selection_history_preserved=True,
        localization='all selected endpoints, eligible source VAL masks only; boundary and patch diagnostics not model selection; unsupervised patch-head metrics labelled accordingly',
        source_training_authorized=True, target_testing_authorized=True,
        source_then_verified_archive_then_target=True, training_seed_order=list(SEEDS),
        first_seed_delivery_before_remaining=True, optional_boundary_loss_ablation_authorized=False,
        first_batch_required_complete_before_remaining=True, automatic_retries=0,
        output_allowance_bytes=3*GIB, total_remaining_output_allowance_bytes=6*GIB,
        old_tasks_paused=True, automation_paused=True, no_new_monitor=True,
        classification_only_deferred=False, future_training_rounds_authorized=False)
    return value


def sampling(rows, epoch, seed):
    assert seed in SEEDS
    return old.sampling(rows, epoch, seed)
