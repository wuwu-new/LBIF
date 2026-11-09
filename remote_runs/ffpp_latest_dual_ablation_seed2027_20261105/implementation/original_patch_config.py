"""Implementation specification, not permission to start a training experiment."""
import math

RUN = 'ffpp_patch_region_implementation_20261105'
BASE_RUN = 'ffpp_paper_components_remaining_20261104'
BASE_REGISTRATION = '42236ec1ae9e8fa019f512f2625ea89f1db91072c044a8baebea16eca3335745'
REFERENCE_ZIP_SHA256 = 'badb1f51305e48044faa33e988464174ff692654a8dc457f1a65e3773d986bcb'
PATCH_SIZE = 16
IMAGE_SIZE = 256
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
FAKE_COVERAGE_THRESHOLD = 0.1
TEMPERATURE = 0.5
BOUNDARY_WEIGHT = 0.2
DELTA_PENALTY = 0.01
VARIANTS = ('NoPatchLearning', 'PatchClassification', 'PatchClassificationContrast')


def variant(name):
    if name not in VARIANTS:
        raise ValueError('Unknown patch-learning variant')
    return dict(patch_classification=0.0 if name == VARIANTS[0] else 0.2,
                patch_contrast=0.1 if name == VARIANTS[2] else 0.0)


def specification():
    return dict(status='implementation_and_synthetic_validation_only',
                source_training_authorized=False, target_testing_authorized=False,
                semantic_frozen=True, new_CLIP_training=False,
                image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                grid_size=GRID_SIZE, patches_per_image=GRID_SIZE**2,
                patch_label_rule='aligned authentic binary mask coverage strictly > 0.1',
                missing_masks='excluded, never replaced by all-real or all-fake patches',
                patch_embedding='adaptive-average-pool encoder f2 to 16x16, 64 channels',
                patch_classifier='shared trainable 64-to-1 head, 256 logits per image',
                contrast='class-balanced per-image supervised contrastive; fake images only',
                contrast_minimum_per_class=2, temperature=TEMPERATURE,
                boundary_weight=BOUNDARY_WEIGHT, delta_penalty=DELTA_PENALTY,
                variants={name: variant(name) for name in VARIANTS},
                image_forward='unchanged error-map/legacy-boundary-pool bounded delta',
                inference_requires_masks=False, patch_heads_used_for_image_inference=False,
                synthetic_preflight_max_image_forwards=64,
                synthetic_disposable_optimizer_updates=6,
                retained_optimizer_updates=0, dataset_frames_read=0,
                full_experiment_budget=None)


def reference_patch_labels(binary_mask, valid=True):
    """Stdlib geometry oracle, not a CPU model or a dataset loader."""
    if not valid:
        return None
    if len(binary_mask) != IMAGE_SIZE or any(len(row) != IMAGE_SIZE for row in binary_mask):
        raise ValueError('Expected one aligned 256x256 mask')
    if any(pixel not in (0, 1) for row in binary_mask for pixel in row):
        raise ValueError('Mask must already be an authentic binary mask')
    result = []
    for y in range(0, IMAGE_SIZE, PATCH_SIZE):
        for x in range(0, IMAGE_SIZE, PATCH_SIZE):
            count = sum(sum(row[x:x+PATCH_SIZE]) for row in binary_mask[y:y+PATCH_SIZE])
            result.append(int(count / PATCH_SIZE**2 > FAKE_COVERAGE_THRESHOLD))
    return result


def reference_balanced_bce(logits, targets):
    """Small saved-number oracle for static tests only."""
    if len(logits) != len(targets) or any(y not in (0, 1) for y in targets):
        raise ValueError('Invalid classification fixture')
    parts = []
    for label in (0, 1):
        errors = [max(z, 0) - z*y + math.log1p(math.exp(-abs(z)))
                  for z, y in zip(logits, targets) if y == label]
        if errors:
            parts.append(sum(errors) / len(errors))
    return sum(parts) / len(parts) if parts else 0.0
