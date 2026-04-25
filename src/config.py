"""
중앙대 의료AI 해커톤 — 전역 설정.

주의: PathMNIST 워밍업과 본 게임 모두 이 config를 공유한다.
PathMNIST는 9 클래스, 본 게임은 11 클래스이므로 NUM_CLASSES는 환경변수
또는 학습 스크립트에서 override 가능.

데이터 디렉토리 구조 (Stage 2 본 게임 기준):
    data/axial/{class}/*.png         # Axial (full)
    data/coronal/{class}/*.png       # Coronal (full)
    data/sagittal/{class}/*.png      # Sagittal (50/class)
    data/test/*.png                  # 평가용

Stage 1 워밍업 (PathMNIST)에서는 dataset_pathmnist.py가 자체적으로
view A/C/S를 시뮬레이션해서 메모리상에서 처리한다.
"""
import os

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)  # src/의 상위
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# View별 디렉토리 (본 게임)
AXIAL_DIR = os.path.join(DATA_DIR, "axial")
CORONAL_DIR = os.path.join(DATA_DIR, "coronal")
SAGITTAL_DIR = os.path.join(DATA_DIR, "sagittal")
TEST_DIR = os.path.join(DATA_DIR, "test")

# 호환성: 기존 train_fewshot.py의 BASE_TRAIN_DIR 참조용 (deprecated)
BASE_TRAIN_DIR = AXIAL_DIR
BASE_VAL_DIR = AXIAL_DIR

# Checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

# View별 base 모델 + Fisher
def view_ckpt(view, kind="model"):
    """view: 'A' or 'C', kind: 'model' or 'fisher' or 'params'"""
    name = {"A": "axial", "C": "coronal"}.get(view, view.lower())
    suffix = {"model": "model.pth", "fisher": "fisher.pth",
              "params": "params.pth"}[kind]
    return os.path.join(CHECKPOINT_DIR, f"{name}_{suffix}")

BASE_CKPT = os.path.join(CHECKPOINT_DIR, "base_model.pth")  # merged
FISHER_CKPT = os.path.join(CHECKPOINT_DIR, "fisher.pth")    # merged Fisher
BASE_PARAMS_CKPT = os.path.join(CHECKPOINT_DIR, "base_params.pth")
FEWSHOT_CKPT = os.path.join(CHECKPOINT_DIR, "fewshot_model.pth")
FINAL_MODEL = os.path.join(CHECKPOINT_DIR, "model.pth")

# ─── Data ────────────────────────────────────────────────
# NUM_CLASSES는 데이터셋에 따라 달라짐. 환경변수로 override 가능.
NUM_CLASSES = int(os.environ.get("MEDAI_NUM_CLASSES", 11))
IMG_SIZE = 28
CHANNELS = 1

# 본 게임 11 클래스
ORGAN_CLASSES = [
    "liver", "kidney_r", "kidney_l", "spleen", "pancreas",
    "aorta", "ivc", "rag", "lag", "gallbladder", "stomach"
]

# PathMNIST 9 클래스 (워밍업)
PATHMNIST_CLASSES = [
    "ADI", "BACK", "DEB", "LYM", "MUC",
    "MUS", "NORM", "STR", "TUM"
]

# ─── Base Training (Phase 1 — view별) ──────────────────
BASE_EPOCHS = 100
BASE_LR = 1e-3
BASE_WEIGHT_DECAY = 5e-4
BASE_BATCH_SIZE = 128
BASE_LABEL_SMOOTHING = 0.1
BASE_DROPOUT = 0.3

# Cosine Annealing Warm Restarts
COSINE_T0 = 30
COSINE_TMULT = 2

# Mixup / CutMix 확률과 강도
MIXUP_ALPHA = 0.4
MIXUP_PROB = 0.5
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.3

# Gradient clipping
MAX_GRAD_NORM = 5.0

# ─── Few-Shot Fine-Tuning (Phase 2) ────────────────────
FEWSHOT_EPOCHS = 50
FEWSHOT_LR_STAGE2 = 1e-5
FEWSHOT_LR_STAGE3 = 5e-5
FEWSHOT_LR_HEAD = 2e-4
FEWSHOT_WEIGHT_DECAY = 1e-4
FEWSHOT_BATCH_SIZE = 32
FEWSHOT_LABEL_SMOOTHING = 0.15

# EWC
EWC_LAMBDA = 5000.0
FISHER_NUM_SAMPLES = 2000

# Replay buffer (각 view에서 클래스당 N개)
REPLAY_SAMPLES_PER_CLASS = 50

# Early stopping
EARLY_STOP_PATIENCE = 10

# ─── Inference ───────────────────────────────────────────
TTA_NUM_AUGMENTS = 12

# ─── Reproducibility ────────────────────────────────────
SEED = 42
ENSEMBLE_SEEDS = [42, 123, 456]

# ─── Augmentation 강도 ──────────────────────────────────
AUG_ROTATION_RANGE = 30       # degrees
AUG_ELASTIC_ALPHA = 2.0
AUG_ELASTIC_SIGMA = 4.0
AUG_BRIGHTNESS_RANGE = 0.1
AUG_CONTRAST_RANGE = (0.8, 1.2)
AUG_NOISE_STD = 0.03
AUG_CROP_SIZE = 24

FEWSHOT_ELASTIC_ALPHA = 3.0
FEWSHOT_AUG_REPEAT = 20       # epoch당 반복 배율
