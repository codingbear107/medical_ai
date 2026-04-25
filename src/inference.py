"""
대회 제출용 추론 스크립트.

요구사항 (대회 명세):
  - 평가 서버는 폐쇄망. 허용 라이브러리(torch, torchvision, numpy, pandas,
    scikit-learn, nibabel, PIL)와 Python 내장 라이브러리만 사용 가능.
  - 입력: --test_dir 폴더 (재귀적으로 PNG/JPG/NPY 검색)
  - 출력: --output CSV 파일 (image_id, prediction 컬럼)

설계 원칙:
  - 외부 dataset 모듈 의존 최소화. load_image를 인라인 구현.
  - TTA: CT 의료영상의 해부학적 상하 보존을 위해 수직(상하) flip은 사용하지 않음.
    수평 flip + 90° 배수 회전 7가지로 구성.
  - Ensemble 옵션: 같은 아키텍처 여러 체크포인트 평균.

사용:
    # 단일 모델
    python inference.py --model_path model.pth --test_dir test --output submission.csv

    # 앙상블
    python inference.py --model_dir checkpoints/ensemble --ensemble \
                        --test_dir test --output submission.csv
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import config
from model import MiniResNet11


# ─── 이미지 로드 (외부 dataset 모듈 의존 없음) ──────────

def load_image(path):
    """단일 이미지 로드 → (H, W) float32 [0, 1].
    PNG/JPG: PIL로 읽어 grayscale 변환, 28×28 resize.
    NPY: 직접 로드, 0~1 범위로 정규화.
    """
    if path.lower().endswith('.npy'):
        img = np.load(path).astype(np.float32)
        # RGB(H, W, 3)일 경우 grayscale 변환
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


# ─── Argparse ───────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inference + TTA + (optional) Ensemble")
    p.add_argument("--model_path", type=str, default=config.FINAL_MODEL,
                   help="단일 모델 경로 (model.pth)")
    p.add_argument("--model_dir", type=str, default=None,
                   help="앙상블용 모델 디렉토리 (.pth/.pt 파일들)")
    p.add_argument("--test_dir", type=str, required=True,
                   help="테스트 이미지 폴더 (재귀 검색)")
    p.add_argument("--output", type=str, default="submission.csv")
    p.add_argument("--ensemble", action="store_true",
                   help="model_dir의 모든 체크포인트로 앙상블")
    p.add_argument("--no_tta", action="store_true",
                   help="TTA 비활성화 (디버깅용)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=None,
                   help="None이면 config.NUM_CLASSES")
    return p.parse_args()


# ─── 모델 로드 ──────────────────────────────────────────

def load_model(path, device, num_classes):
    """체크포인트 로드 (model_state_dict 또는 raw state_dict 모두 지원)."""
    model = MiniResNet11(num_classes=num_classes,
                         dropout=0.0,
                         use_style_rand=False)  # 추론 시 항상 비활성
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    # style_rand 키 누락 등에 강건하도록 strict=False
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


# ─── TTA: 수평 flip + 90° 회전 7가지 ────────────────────

def tta_augment_batch(images):
    """TTA 변형 생성. 수직 flip은 의도적으로 제외 (CT 해부학적 상하 보존).

    7가지: 원본, hflip, rot90, rot180, rot270, hflip+rot90, hflip+rot270
    """
    out = [images]
    out.append(torch.flip(images, [3]))                       # hflip
    out.append(torch.rot90(images, 1, [2, 3]))                # rot90
    out.append(torch.rot90(images, 2, [2, 3]))                # rot180
    out.append(torch.rot90(images, 3, [2, 3]))                # rot270
    out.append(torch.flip(torch.rot90(images, 1, [2, 3]), [3]))  # hflip+rot90
    out.append(torch.flip(torch.rot90(images, 3, [2, 3]), [3]))  # hflip+rot270
    return out


@torch.no_grad()
def predict_with_tta(model, images, device, use_tta=True):
    """단일 모델 예측. TTA 활성화 시 7가지 변형의 softmax 확률 평균.
    Returns: (B, num_classes) softmax probs
    """
    images = images.to(device)
    if not use_tta:
        return F.softmax(model(images), dim=1)

    aug_list = tta_augment_batch(images)
    probs_list = []
    for aug in aug_list:
        logits = model(aug.to(device))
        probs_list.append(F.softmax(logits, dim=1))
    return torch.stack(probs_list, dim=0).mean(dim=0)


# ─── Test 이미지 탐색 ───────────────────────────────────

def discover_test_images(test_dir):
    """test_dir 하위에서 PNG/JPG/JPEG/NPY 파일을 재귀 검색.
    Returns: [(path, image_id), ...] (정렬됨, 중복 제거)
    """
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.npy',
                  '*.PNG', '*.JPG', '*.JPEG', '*.NPY')
    files = set()
    for ext in extensions:
        files.update(glob.glob(os.path.join(test_dir, ext)))
        files.update(glob.glob(os.path.join(test_dir, '**', ext), recursive=True))

    files = sorted(files)
    out = []
    for f in files:
        rel = os.path.relpath(f, test_dir)
        image_id = os.path.splitext(rel)[0]
        # OS 호환: image_id에 백슬래시(Windows) 있으면 슬래시로 통일
        image_id = image_id.replace('\\', '/')
        out.append((f, image_id))
    return out


def load_batch(paths):
    """이미지 배치 로드 → (B, 1, H, W) tensor."""
    tensors = []
    for p in paths:
        img = load_image(p)
        tensors.append(torch.from_numpy(img).unsqueeze(0))
    return torch.stack(tensors)


# ─── Main ───────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = args.num_classes or config.NUM_CLASSES
    print(f"[inference] device={device}, num_classes={num_classes}")

    # 모델(들) 로드
    if args.ensemble and args.model_dir:
        files = sorted(set(glob.glob(os.path.join(args.model_dir, '*.pth')) +
                           glob.glob(os.path.join(args.model_dir, '*.pt'))))
        if not files:
            print(f"ERROR: {args.model_dir}에 모델 파일 없음")
            return
        models = [load_model(f, device, num_classes) for f in files]
        print(f"[inference] Ensemble: {len(models)}개 모델 로드")
    else:
        models = [load_model(args.model_path, device, num_classes)]
        print(f"[inference] 단일 모델 로드: {args.model_path}")

    # 테스트 이미지 탐색
    test_data = discover_test_images(args.test_dir)
    if not test_data:
        print(f"ERROR: {args.test_dir}에 테스트 이미지 없음")
        return
    print(f"[inference] Test 이미지: {len(test_data)}장")

    # 배치 추론
    all_ids, all_preds = [], []
    use_tta = not args.no_tta

    for i in range(0, len(test_data), args.batch_size):
        batch = test_data[i:i + args.batch_size]
        paths = [b[0] for b in batch]
        ids = [b[1] for b in batch]
        images = load_batch(paths)

        if len(models) > 1:
            ens_probs = []
            for m in models:
                ens_probs.append(predict_with_tta(m, images, device, use_tta))
            probs = torch.stack(ens_probs, dim=0).mean(dim=0)
        else:
            probs = predict_with_tta(models[0], images, device, use_tta)

        preds = probs.argmax(dim=1).cpu().numpy()
        all_ids.extend(ids)
        all_preds.extend(preds.tolist())

        if (i // args.batch_size) % 10 == 0:
            done = min(i + args.batch_size, len(test_data))
            print(f"  진행: {done}/{len(test_data)}")

    # CSV 저장
    df = pd.DataFrame({'image_id': all_ids, 'prediction': all_preds})
    df.to_csv(args.output, index=False)
    print(f"\n[inference] 저장 완료: {args.output} ({len(df)}건)")

    # 분포 출력
    print("\n[inference] 클래스별 예측 분포:")
    for cls in range(num_classes):
        count = int((df['prediction'] == cls).sum())
        pct = count / len(df) * 100 if len(df) else 0
        name = (config.ORGAN_CLASSES[cls]
                if cls < len(config.ORGAN_CLASSES)
                else f"class_{cls}")
        print(f"  {cls:2d} {name:<14} {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
