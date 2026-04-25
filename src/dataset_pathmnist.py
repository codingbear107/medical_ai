"""
PathMNIST 워밍업 데이터셋.

목적:
  본 게임 데이터(Axial/Coronal/Sagittal CT)가 도착하기 전, PathMNIST를
  사용해 파이프라인 전체(View training → Model Merging → Few-shot →
  Inference)를 검증한다.

전략:
  PathMNIST에는 view 개념이 없다. 대신 다음과 같이 "가짜 view"를 시뮬레이션:

      View A (Axial 시뮬)   = 원본 이미지
      View C (Coronal 시뮬) = 90° 회전 + 약한 elastic deformation
      View S (Sagittal 시뮬)= 270° 회전 + Gaussian blur + 약한 shear

  같은 라벨의 같은 객체를 서로 다른 "관점"으로 보는 셈이 된다.
  본 게임의 A/C 풀데이터 + S 50샷 시나리오를 그대로 흉내내기 위해:
      - View A의 train: 전체 (~89,996장)
      - View A의 val:   PathMNIST val
      - View C의 train: PathMNIST train (변형 적용)
      - View C의 val:   PathMNIST val (변형 적용)
      - View S의 train: 클래스당 50장만 샘플링 (few-shot)
      - View S의 val:   PathMNIST test의 일부 (few-shot 검증용)
      - Test:           PathMNIST test 전체 (A+C+S 세 변형 모두 적용 후 평가)

윤리 가드레일:
  PathMNIST에서 발견한 하이퍼파라미터를 본 게임에 그대로 복사하지 않는다.
  코드 구조 검증과 PyTorch/Colab 사용 감각 확보 용도.

라이브러리:
  - medmnist: 워밍업 전용. inference.py에서는 절대 import 금지.
  - 외 numpy, torch만 사용.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from augmentations import (
    base_augment, fewshot_augment,
    rotate_image, elastic_deformation, random_affine_shear,
    _box_filter_2d,
)

# medmnist는 lazy import (없는 환경에서도 module import는 가능하도록)
_PATHMNIST_CACHE = None


def _load_pathmnist(size=28):
    """PathMNIST 다운로드 및 캐시. 첫 호출 시 자동 다운로드."""
    global _PATHMNIST_CACHE
    if _PATHMNIST_CACHE is not None:
        return _PATHMNIST_CACHE

    from medmnist import PathMNIST  # lazy import
    train = PathMNIST(split='train', download=True, size=size)
    val = PathMNIST(split='val', download=True, size=size)
    test = PathMNIST(split='test', download=True, size=size)

    def to_arrays(ds):
        # ds.imgs: (N, H, W, 3) uint8, ds.labels: (N, 1) int
        imgs = np.asarray(ds.imgs, dtype=np.uint8)
        labels = np.asarray(ds.labels, dtype=np.int64).squeeze(-1)
        return imgs, labels

    _PATHMNIST_CACHE = {
        'train': to_arrays(train),
        'val': to_arrays(val),
        'test': to_arrays(test),
    }
    return _PATHMNIST_CACHE


def rgb_to_gray(rgb_uint8):
    """(H, W, 3) uint8 → (H, W) float32 [0, 1]
    ITU-R BT.601 luma 변환. (의료영상 표준 grayscale 변환)
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.astype(np.float32)


# ─── View 시뮬레이션 변환 (deterministic, view label에 의존) ─────

def _gauss_blur(img, sigma=1.0):
    """간이 가우시안 블러 (box filter 3회 근사)."""
    k = max(3, int(sigma * 2) | 1)
    out = img
    for _ in range(3):
        out = _box_filter_2d(out, k)
    return out.astype(np.float32)


def view_transform(img, view):
    """주어진 view label에 따른 결정적 변환.
    이 변환은 augmentation과 별개로, "같은 객체의 다른 시점"을 만든다.

      view = 'A': 원본
      view = 'C': 90° 회전 + 약한 elastic
      view = 'S': 270° 회전 + 가우시안 blur + 약한 shear
    """
    if view == 'A':
        return img.astype(np.float32)
    elif view == 'C':
        out = np.rot90(img, k=1).copy()
        # 약한 elastic (deterministic하게 만들기 위해 seed 없이 호출 — 학습 시 매번 약간 다름)
        # 워밍업 목적이라 일정 강도 변형이면 충분
        if np.random.random() < 0.5:
            out = elastic_deformation(out, alpha=1.5, sigma=4.0)
        return out
    elif view == 'S':
        out = np.rot90(img, k=3).copy()  # 270°
        out = _gauss_blur(out, sigma=1.0)
        if np.random.random() < 0.5:
            out = random_affine_shear(out, max_shear_deg=8)
        return out.astype(np.float32)
    else:
        raise ValueError(f"Unknown view: {view}")


# ─── Dataset 클래스들 ─────────────────────────────────

class PathMNISTViewDataset(Dataset):
    """단일 view의 PathMNIST. view 변환 + augmentation을 결합."""

    def __init__(self, split, view, augment_fn=None, indices=None,
                 repeat=1):
        """
        Args:
            split: 'train' | 'val' | 'test'
            view:  'A' | 'C' | 'S'
            augment_fn: callable(img) -> img  (또는 None)
            indices: 샘플링 인덱스 (None이면 전체)
            repeat: epoch당 반복 배율 (few-shot에서 사용)
        """
        cache = _load_pathmnist()
        imgs, labels = cache[split]

        if indices is not None:
            imgs = imgs[indices]
            labels = labels[indices]

        self.imgs_rgb = imgs
        self.labels = labels
        self.view = view
        self.augment_fn = augment_fn
        self.repeat = repeat

    def __len__(self):
        return len(self.imgs_rgb) * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % len(self.imgs_rgb)
        rgb = self.imgs_rgb[real_idx]
        label = int(self.labels[real_idx])

        gray = rgb_to_gray(rgb)
        gray = view_transform(gray, self.view)

        if self.augment_fn is not None:
            gray = self.augment_fn(gray)

        # contiguous + (1, H, W) tensor
        gray = np.ascontiguousarray(gray, dtype=np.float32)
        tensor = torch.from_numpy(gray).unsqueeze(0)
        return tensor, label


def sample_balanced_indices(labels, samples_per_class, seed=42):
    """클래스별 균형 인덱스 샘플링 (few-shot용)."""
    rng = np.random.RandomState(seed)
    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = min(samples_per_class, len(cls_idx))
        indices.extend(cls_idx[:n].tolist())
    return np.array(indices)


# ─── Factory functions ─────────────────────────────────

def create_view_datasets(view, augment_train=True):
    """View A 또는 C의 base 학습용 train/val 데이터셋 생성.

    Args:
        view: 'A' or 'C'
        augment_train: train에 base_augment 적용 여부

    Returns:
        (train_ds, val_ds)
    """
    train_ds = PathMNISTViewDataset(
        split='train', view=view,
        augment_fn=base_augment if augment_train else None,
    )
    val_ds = PathMNISTViewDataset(
        split='val', view=view, augment_fn=None,
    )
    return train_ds, val_ds


def create_fewshot_datasets(view='S', samples_per_class=50,
                            val_samples_per_class=30,
                            seed=42):
    """View S few-shot 학습용 train/val.

    워밍업에서는 PathMNIST train에서 50샷, test에서 별도 30샷을 val로 사용.
    """
    cache = _load_pathmnist()
    _, train_labels = cache['train']
    _, test_labels = cache['test']

    train_idx = sample_balanced_indices(train_labels, samples_per_class, seed=seed)
    val_idx = sample_balanced_indices(test_labels, val_samples_per_class, seed=seed + 1)

    train_ds = PathMNISTViewDataset(
        split='train', view=view, augment_fn=fewshot_augment,
        indices=train_idx, repeat=config.FEWSHOT_AUG_REPEAT,
    )
    val_ds = PathMNISTViewDataset(
        split='test', view=view, augment_fn=None,
        indices=val_idx,
    )
    return train_ds, val_ds


def create_replay_dataset(view, samples_per_class=50, seed=42):
    """Few-shot 학습 중 catastrophic forgetting 방어용 replay 버퍼.
    Base view (A 또는 C)의 train에서 클래스 균형 샘플링.
    """
    cache = _load_pathmnist()
    _, train_labels = cache['train']
    indices = sample_balanced_indices(train_labels, samples_per_class, seed=seed)
    return PathMNISTViewDataset(
        split='train', view=view, augment_fn=base_augment,
        indices=indices,
    )


def create_test_datasets(samples_per_class=None):
    """평가용 test 데이터셋 (A, C, S 세 view 모두).

    대회식 평가 (0.7·F1(S) + 0.3·(F1(A)+F1(C))/2)를 시뮬레이션하기 위해
    같은 PathMNIST test를 세 view로 변환하여 각각 평가.
    """
    cache = _load_pathmnist()
    _, test_labels = cache['test']

    if samples_per_class is not None:
        idx = sample_balanced_indices(test_labels, samples_per_class)
    else:
        idx = None

    return {
        'A': PathMNISTViewDataset('test', 'A', augment_fn=None, indices=idx),
        'C': PathMNISTViewDataset('test', 'C', augment_fn=None, indices=idx),
        'S': PathMNISTViewDataset('test', 'S', augment_fn=None, indices=idx),
    }


if __name__ == "__main__":
    # 간단 sanity check (Colab에서 실행)
    print("Loading PathMNIST...")
    cache = _load_pathmnist()
    print(f"Train: {cache['train'][0].shape}, labels unique: {np.unique(cache['train'][1])}")
    print(f"Val:   {cache['val'][0].shape}")
    print(f"Test:  {cache['test'][0].shape}")

    print("\nView A train dataset:")
    ds_a, _ = create_view_datasets('A')
    img, label = ds_a[0]
    print(f"  shape={tuple(img.shape)}, dtype={img.dtype}, label={label}")

    print("\nFewshot S dataset (50 per class):")
    fs_train, fs_val = create_fewshot_datasets()
    print(f"  train (with repeat={config.FEWSHOT_AUG_REPEAT}): {len(fs_train)}")
    print(f"  val: {len(fs_val)}")

    print("\nReplay dataset (view A, 50 per class):")
    rd = create_replay_dataset('A')
    print(f"  size: {len(rd)}")
