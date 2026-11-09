"""Finite full patch-learning experiment; standard-library configuration only."""
import importlib.util
from pathlib import Path

try:
    import legacy_component_plan as old
except ModuleNotFoundError:
    path = Path(__file__).resolve().parent.parent/'paper_components_20261104/plan.py'
    spec = importlib.util.spec_from_file_location('patch_full_original_plan', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)

for name in dir(old):
    if not name.startswith('_'):
        globals()[name] = getattr(old, name)

RUN = 'ffpp_patch_region_full_20261105'
PARENT = 'ffpp_paper_components_remaining_20261104'
PARENT_REG = '42236ec1ae9e8fa019f512f2625ea89f1db91072c044a8baebea16eca3335745'
IMPLEMENTATION = 'ffpp_patch_region_implementation_20261105'
IMPLEMENTATION_REG = '463954fa31eefa3b7f909a0b467254a357da3b875a579bd12788a7588356f9f3'
SEEDS = (2027, 3407, 4099)
NEW_ARMS = ('PatchClassificationContrast',)


def variant(arm):
    assert arm in NEW_ARMS
    return dict(input='signed_RGB_minus_Gaussian5_sigma1',
                pooling='predicted_boundary_cells_quarter_floor',
                boundary_coefficient=.2, patch_classification=.2,
                patch_contrast=.1, temperature=.5)


def configuration():
    value = old.configuration().copy()
    value.update(primary_question='full authentic-mask patch classification plus intra-image contrast versus reused historical E and semantic controls',
        new_arms={a: variant(a) for a in NEW_ARMS}, seeds=list(SEEDS),
        retained_updates=21570, retained_training_frame_forwards=345120,
        preflight_disposable_updates=6, source_VAL_frame_forwards=662400,
        selected_localization_forwards=66240, selected_reload_forwards=192,
        source_preflight_anchor_forwards=192, source_preflight_synthetic_forwards=18,
        source_preflight_real_TRAIN_forwards=6, source_preflight_total_forwards=216,
        target_preflight_forwards_per_invocation=864, full_target_residual_forwards=669408,
        target_preflight_total_forwards=1728, source_training_authorized=True,
        target_testing_authorized=True, implementation_run=IMPLEMENTATION,
        implementation_registration_sha256=IMPLEMENTATION_REG,
        architecture='unchanged historical E CNN/decoder/delta plus training-only 64to1 patch head (65 parameters)',
        patch_geometry='16x16 grid of context-bearing CNN embeddings, 16x16 input pixels per target; not independent cropped CNNs',
        patch_labels='aligned authentic binary masks, coverage strictly >0.1; missing excluded',
        patch_loss='within-image class-balanced BCE; image-balanced mean',
        contrast_loss='fake TRAIN only, both classes >=2 patches, no diagonal, class/image-balanced SupCon',
        inference='no masks, no patch head, unchanged image aggregation',
        preflight_real_selection='first valid real TRAIN and first of at most64 valid fake TRAIN with >=2 real and >=2 fake patches',
        preflight_shared_state='shared state excluding patch_classifier equals historical initialization; full-state locked separately',
        prior_implementation_disposable_updates=6, prior_implementation_image_forwards=24,
        prior_implementation_retained_updates=0, runtime_disposable_updates=0,
        localization='all three selected endpoints; eligible VAL masks, boundary metrics plus patch confusion at0.5; diagnostic only',
        inactive_decoder=None, reused_arms=['SemanticRaw', 'SemanticHalf', 'NoPatchLearning_historical_E'],
        historical_E_not_renamed_as_new_training=True,
        historical_E_seed2027_pilot_prefix_disclosed=True,
        historical_E_selection_epochs={'2027':10, '3407':1, '4099':10},
        classification_only_deferred=True, full_variant_first=True,
        one_full_variant_cannot_isolate_classification_from_contrast=True,
        output_allowance_bytes=2*GIB, future_training_rounds_authorized=False,
        training_seed_order=list(SEEDS), source_then_verified_archive_then_target=True)
    return value


def sampling(rows, epoch, seed):
    assert seed in SEEDS
    return old.sampling(rows, epoch, seed)
