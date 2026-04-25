"""
본 게임용 (Stage 2) 데이터셋. 실데이터 도착 후 폴더 구조에 맞춰 조정.

가정하는 디렉토리 구조 (실데이터 받기 전 placeholder):
    data/axial/{class_name}/*.png       # Axial 풀데이터
    data/coronal/{class_name}/*.png     # Coronal 풀데이터
    data/sagittal/{class_name}/*.png    # Sagittal 50샷/클래스
    data/test/*.png                     # 평가용 (분류 안 됨)

train_view.py / train_fewshot.py가 이 모듈에서 다음을 import:
    create_view_datasets(view, augment_train)  # 'A' or 'C'
    create_fewshot_datasets()                  # Sagittal split
    create_replay_dataset(view)                # base view에서 균형 샘플링

실제 데이터 도착 시 확인 사항:
    1. 폴더 구조가 위와 같은지 (다르면 discover_folder_dataset 수정)
    2. 클래스 이름이 config.ORGAN_CLASSES와 일치하는지
    3. 이미지 형식 (PNG/NPY/...)
    4. Sagittal에 train/val split이 별도 제공되는지, 아니면 내부에서 split하는지
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import config
from augmentations import base_augment, fewshot_augment


def load_image(path):
    """grayscale 28×28 float32 [0, 1] 로드."""
    if path.lower().endswith('.npy'):
        img = np.load(path).astype(np.float32)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        if img.max() > 1.5:
            img = img / 255.0
    else:
        pil = Image.open(path).convert('L')
        if pil.size != (config.IMG_SIZE, config.IMG_SIZE):
            pil = pil.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
        img = np.asarray(pil, dtype=np.float32) / 255.0
    return img.astype(np.float32)


def discover_folder_dataset(root_dir):
    """root_dir/{class}/*.png 구조에서 (paths, labels) 발견.

    클래스 디렉토리 이름이 config.ORGAN_CLASSES에 있으면 해당 인덱스 사용,
    아니면 알파벳 순으로 인덱스 부여.
    """
    images, labels = [], []
    if not os.path.exists(root_dir):
        return images, labels

    class_dirs = sorted([d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d))])

    # ORGAN_CLASSES와 매칭 시도, 실패 시 알파벳 순
    name_to_idx = {n: i for i, n in enumerate(config.ORGAN_CLASSES)}
    fallback = {d: i for i, d in enumerate(class_dirs)}

    for class_name in class_dirs:
        cls_id = name_to_idx.get(class_name, fallback[class_name])
        cls_dir = os.path.join(root_dir, class_name)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.npy')):
                images.append(os.path.join(cls_dir, fname))
                labels.append(cls_id)

    return images, labels


class OrganDataset(Dataset):
    """파일 경로 기반 organ classification dataset."""

    def __init__(self, image_paths, labels, augment_fn=None, repeat=1):
        self.image_paths = image_paths
        self.labels = labels
        self.augment_fn = augment_fn
        self.repeat = repeat

    def __len__(self):
        return len(self.image_paths) * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_paths)
        img = load_image(self.image_paths[real_idx])
        if self.augment_fn is not None:
            img = self.augment_fn(img)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return torch.from_numpy(img).unsqueeze(0), int(self.labels[real_idx])


class BalancedReplayDataset(Dataset):
    """클래스별 균형 샘플링 replay 버퍼."""

    def __init__(self, image_paths, labels, samples_per_class=50,
                 augment_fn=None, seed=42):
        rng = np.random.RandomState(seed)
        class_imgs = {}
        for p, l in zip(image_paths, labels):
            class_imgs.setdefault(l, []).append(p)

        self.image_paths, self.labels = [], []
        for cls in sorted(class_imgs.keys()):
            paths = class_imgs[cls]
            n = min(samples_per_class, len(paths))
            replace = len(paths) < n
            sel = rng.choice(len(paths), n, replace=replace)
            for i in sel:
                self.image_paths.append(paths[i])
                self.labels.append(cls)

        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        if self.augment_fn is not None:
            img = self.augment_fn(img)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return torch.from_numpy(img).unsqueeze(0), int(self.labels[idx])


# ─── Factory functions (train_view.py / train_fewshot.py가 호출) ────

def create_view_datasets(view, augment_train=True, val_ratio=0.1, seed=42):
    """View 'A' or 'C'의 train/val 데이터셋.
    실데이터에 별도 val 분할이 없을 경우 클래스별로 비율 split."""
    view_dir = {'A': config.AXIAL_DIR, 'C': config.CORONAL_DIR}[view]
    paths, labels = discover_folder_dataset(view_dir)

    # 별도 val 폴더 (data/axial_val 등)가 있으면 우선 사용 — 데이터 도착 후 결정
    val_dir = view_dir + "_val"
    if os.path.exists(val_dir):
        train_p, train_l = paths, labels
        val_p, val_l = discover_folder_dataset(val_dir)
    else:
        # 클래스별 split
        rng = np.random.RandomState(seed)
        cls_groups = {}
        for p, l in zip(paths, labels):
            cls_groups.setdefault(l, []).append(p)
        train_p, train_l, val_p, val_l = [], [], [], []
        for cls in sorted(cls_groups.keys()):
            ps = cls_groups[cls]
            rng.shuffle(ps)
            n_val = max(1, int(len(ps) * val_ratio))
            val_p.extend(ps[:n_val]); val_l.extend([cls] * n_val)
            train_p.extend(ps[n_val:]); train_l.extend([cls] * (len(ps) - n_val))

    train_ds = OrganDataset(train_p, train_l,
                            augment_fn=base_augment if augment_train else None)
    val_ds = OrganDataset(val_p, val_l, augment_fn=None)
    return train_ds, val_ds


def create_fewshot_datasets(val_ratio=0.2, seed=42):
    """Sagittal 데이터셋. 50샷/클래스에서 val_ratio만큼 val로 분리."""
    paths, labels = discover_folder_dataset(config.SAGITTAL_DIR)
    rng = np.random.RandomState(seed)
    cls_groups = {}
    for p, l in zip(paths, labels):
        cls_groups.setdefault(l, []).append(p)

    train_p, train_l, val_p, val_l = [], [], [], []
    for cls in sorted(cls_groups.keys()):
        ps = cls_groups[cls]
        rng.shuffle(ps)
        n_val = max(1, int(len(ps) * val_ratio))
        val_p.extend(ps[:n_val]); val_l.extend([cls] * n_val)
        train_p.extend(ps[n_val:]); train_l.extend([cls] * (len(ps) - n_val))

    train_ds = OrganDataset(train_p, train_l,
                            augment_fn=fewshot_augment,
                            repeat=config.FEWSHOT_AUG_REPEAT)
    val_ds = OrganDataset(val_p, val_l, augment_fn=None)
    return train_ds, val_ds


def create_replay_dataset(view='A'):
    """Replay buffer: 지정 view에서 클래스당 N개 균형 샘플링."""
    view_dir = {'A': config.AXIAL_DIR, 'C': config.CORONAL_DIR}[view]
    paths, labels = discover_folder_dataset(view_dir)
    return BalancedReplayDataset(
        paths, labels,
        samples_per_class=config.REPLAY_SAMPLES_PER_CLASS,
        augment_fn=base_augment,
        seed=config.SEED,
    )
