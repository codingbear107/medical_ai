"""
의료영상 데이터 증강 (numpy + torch only, 외부 라이브러리 0개)

설계 원칙
─────────
1. 수직(상하) flip 금지: CT는 해부학적 상하관계가 의미를 가짐
   (간은 위, 담낭은 아래 등). 수직 flip은 클래스 혼동을 유발한다.
2. 수평 flip은 허용 (50%): kidney_l/r 등 좌우 구분이 라벨에 반영됨.
3. 90° 배수 회전: A/C/S 세 축이 서로 90° 회전 관계 → view-equivariance 유도.
4. FFT amplitude 교란: phase는 보존(=shape) / amplitude만 변형(=texture)
   → 모델이 texture가 아닌 shape에 의존하도록 유도.
5. Elastic deformation: 환자별 자연스러운 장기 변형 시뮬레이션.

구현 함수 (전부 numpy 단일 채널 (H, W) float32 [0,1] 입력 가정)
  rotate_image, _bilinear_sample, elastic_deformation, _box_filter_2d,
  random_crop_resize, random_affine_shear,
  random_brightness, random_contrast, random_noise,
  freq_augment, random_90_rotation,
  base_augment, fewshot_augment,
  mixup, cutmix (배치 단위, torch 텐서)
"""
import numpy as np
import torch


# ─── Geometric Augmentations ────────────────────────────

def _bilinear_sample(img, y_map, x_map):
    """Bilinear interpolation 샘플링.
    img: (H, W) float32, y_map/x_map: (H, W) float32
    범위 밖 좌표는 가장자리로 clip (zero-padding이 아님 → 의료영상 외곽 보존)
    """
    h, w = img.shape
    x0 = np.floor(x_map).astype(np.int32)
    y0 = np.floor(y_map).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    dx = x_map - x0.astype(np.float32)
    dy = y_map - y0.astype(np.float32)

    val = (img[y0, x0] * (1 - dx) * (1 - dy) +
           img[y0, x1] * dx * (1 - dy) +
           img[y1, x0] * (1 - dx) * dy +
           img[y1, x1] * dx * dy)
    return val.astype(np.float32)


def rotate_image(img, angle_deg):
    """임의 각도 회전 (bilinear). 28×28 작은 이미지에 적합."""
    h, w = img.shape
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    cy, cx = h / 2, w / 2
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    x_c = x_coords - cx
    y_c = y_coords - cy
    x_src = cos_a * x_c + sin_a * y_c + cx
    y_src = -sin_a * x_c + cos_a * y_c + cy

    return _bilinear_sample(img, y_src, x_src)


def random_90_rotation(img):
    """0/90/180/270도 중 균등 랜덤 회전.
    A/C/S 축 사이 90° 회전 관계를 모델이 학습하도록 강제.
    np.rot90는 메모리 효율적(view 반환)이지만 .copy()로 contiguous 보장.
    """
    k = np.random.randint(0, 4)
    return np.rot90(img, k=k).copy()


def elastic_deformation(img, alpha=2.0, sigma=4.0):
    """탄성 변형. 환자별 자연 변형 시뮬레이션.
    저주파 displacement field (dx, dy)를 만들어 픽셀을 휨.
    """
    h, w = img.shape
    dx = np.random.randn(h, w).astype(np.float32)
    dy = np.random.randn(h, w).astype(np.float32)

    # box filter 3회 = 가우시안 근사
    kernel_size = max(3, int(sigma * 2) | 1)
    for _ in range(3):
        dx = _box_filter_2d(dx, kernel_size)
        dy = _box_filter_2d(dy, kernel_size)

    dx = dx * alpha
    dy = dy * alpha

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    x_new = x_coords + dx
    y_new = y_coords + dy

    return _bilinear_sample(img, y_new, x_new)


def _box_filter_2d(arr, k):
    """2D box filter (cumsum trick, 빠름). reflect padding으로 입력과 같은 크기 유지.

    수식 검증 (k 홀수일 때):
        pad = (k-1)/2
        padded shape  = (H + k-1, W + k-1)
        cs (앞에 0열 추가) shape = (H + k-1, W + k)
        cs[:, k:] - cs[:, :-k]  → shape (H + k-1, W)  ← 정확히 W 폭
        세로도 동일하게 → 최종 shape (H, W)
    """
    if k % 2 == 0:
        k += 1  # 홀수 보장
    pad = k // 2
    padded = np.pad(arr, pad, mode='reflect').astype(np.float32)
    H_p, W_p = padded.shape

    # 가로 box: prefix sum 앞에 0열 추가 → 정확히 입력 W 만큼 출력
    cs = np.concatenate(
        [np.zeros((H_p, 1), dtype=np.float32),
         np.cumsum(padded, axis=1)], axis=1
    )  # (H_p, W_p + 1)
    h_filtered = (cs[:, k:] - cs[:, :-k]) / k  # (H_p, W_p + 1 - k) = (H_p, W)

    # 세로 box: 같은 방식
    cs2 = np.concatenate(
        [np.zeros((1, h_filtered.shape[1]), dtype=np.float32),
         np.cumsum(h_filtered, axis=0)], axis=0
    )  # (H_p + 1, W)
    v_filtered = (cs2[k:, :] - cs2[:-k, :]) / k  # (H, W)

    return v_filtered.astype(np.float32)


def random_crop_resize(img, crop_size=24, target_size=28):
    """랜덤 크롭 후 다시 target_size로 확대 (zoom-in 효과)."""
    h, w = img.shape
    max_y = h - crop_size
    max_x = w - crop_size
    if max_y <= 0 or max_x <= 0:
        return img

    y0 = np.random.randint(0, max_y + 1)
    x0 = np.random.randint(0, max_x + 1)
    cropped = img[y0:y0 + crop_size, x0:x0 + crop_size]

    scale_y = crop_size / target_size
    scale_x = crop_size / target_size
    y_map = np.arange(target_size, dtype=np.float32) * scale_y
    x_map = np.arange(target_size, dtype=np.float32) * scale_x
    y_map, x_map = np.meshgrid(y_map, x_map, indexing='ij')

    return _bilinear_sample(cropped, y_map, x_map)


def random_affine_shear(img, max_shear_deg=10):
    """수평 shear (찌그러뜨림)."""
    h, w = img.shape
    shear_rad = np.deg2rad(np.random.uniform(-max_shear_deg, max_shear_deg))

    cy, cx = h / 2, w / 2
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    x_c = x_coords - cx
    y_c = y_coords - cy

    x_src = x_c + np.tan(shear_rad) * y_c + cx
    y_src = y_c + cy

    return _bilinear_sample(img, y_src, x_src)


# ─── Intensity Augmentations ────────────────────────────

def random_brightness(img, max_delta=0.1):
    delta = np.random.uniform(-max_delta, max_delta)
    return np.clip(img + delta, 0, 1).astype(np.float32)


def random_contrast(img, low=0.8, high=1.2):
    factor = np.random.uniform(low, high)
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0, 1).astype(np.float32)


def random_noise(img, max_std=0.03):
    std = np.random.uniform(0, max_std)
    noise = np.random.randn(*img.shape).astype(np.float32) * std
    return np.clip(img + noise, 0, 1).astype(np.float32)


# ─── Frequency Domain Augmentation ──────────────────────

def freq_augment(img, amplitude_range=0.2):
    """FFT amplitude만 ±amplitude_range 비율로 교란, phase는 고정.
    수학:  F = FFT(x) = |F| * exp(i*phase)
           |F|' = |F| * (1 + U(-r, r))   ← amplitude 교란
           x'   = IFFT(|F|' * exp(i*phase))
    Phase가 shape 정보, Amplitude가 texture 정보를 담는다는 푸리에 분석 사실에서
    출발 → 모델이 shape에 의존하도록 유도.
    """
    fft = np.fft.fft2(img)
    amplitude = np.abs(fft)
    phase = np.angle(fft)

    noise = 1.0 + np.random.uniform(-amplitude_range, amplitude_range,
                                    size=amplitude.shape).astype(np.float32)
    amplitude_aug = amplitude * noise
    fft_aug = amplitude_aug * np.exp(1j * phase)
    img_aug = np.real(np.fft.ifft2(fft_aug))
    return np.clip(img_aug, 0, 1).astype(np.float32)


# ─── Composite Augmentation Pipelines ───────────────────

def base_augment(img):
    """Base view (Axial / Coronal) 학습용 augmentation.
    img: (28, 28) float32, [0, 1]
    """
    # 90° 회전 (확률 30%) — view-equivariance 유도
    if np.random.random() < 0.3:
        img = random_90_rotation(img)

    # 임의 각도 회전 ±30°
    angle = np.random.uniform(-30, 30)
    img = rotate_image(img, angle)

    # 수평 flip (좌우는 의미가 있지만 일부 클래스에선 무방, 50%)
    if np.random.random() < 0.5:
        img = img[:, ::-1].copy()

    # 수직 flip — 사용하지 않음 (해부학적 상하관계 보존)

    # 탄성 변형
    if np.random.random() < 0.5:
        img = elastic_deformation(img, alpha=2.0, sigma=4.0)

    # FFT amplitude 교란 (texture bias 깨기)
    if np.random.random() < 0.3:
        img = freq_augment(img, amplitude_range=0.2)

    # 강도 변형
    img = random_brightness(img, 0.1)
    img = random_contrast(img, 0.8, 1.2)

    if np.random.random() < 0.3:
        img = random_noise(img, 0.03)

    # zoom-in
    if np.random.random() < 0.4:
        img = random_crop_resize(img, 24, 28)

    return img


def fewshot_augment(img):
    """Sagittal few-shot (50샷/클래스) 학습용 — base보다 강한 증강.
    데이터가 적으니 더 적극적으로 변형하여 over-fit 억제.
    """
    # 90° 회전 (확률 50%)
    if np.random.random() < 0.5:
        img = random_90_rotation(img)

    # 임의 각도 회전 ±45°
    angle = np.random.uniform(-45, 45)
    img = rotate_image(img, angle)

    # 수평 flip만
    if np.random.random() < 0.5:
        img = img[:, ::-1].copy()

    # 강한 탄성 변형
    if np.random.random() < 0.6:
        img = elastic_deformation(img, alpha=3.0, sigma=4.0)

    # Shear
    if np.random.random() < 0.3:
        img = random_affine_shear(img, 10)

    # FFT amplitude 강화
    if np.random.random() < 0.4:
        img = freq_augment(img, amplitude_range=0.3)

    # 강도
    img = random_brightness(img, 0.15)
    img = random_contrast(img, 0.7, 1.3)

    if np.random.random() < 0.4:
        img = random_noise(img, 0.05)

    if np.random.random() < 0.5:
        img = random_crop_resize(img, 22, 28)

    return img


# ─── Mixup / CutMix (batch-level, torch tensor) ──────────

def mixup(x, y, alpha=0.4):
    """Mixup: 두 샘플을 선형 보간.
    x: (B, C, H, W),  y: (B,)
    return: mixed_x, y_a, y_b, lam   (loss = lam*CE(_,y_a) + (1-lam)*CE(_,y_b))
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix(x, y, alpha=1.0):
    """CutMix: 한 샘플의 사각 영역을 다른 샘플로 덮어씀.
    Mixup보다 위치 불변성 강화.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, h, w = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # 실제 잘려나간 면적 비율로 lam 보정
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return mixed_x, y, y[index], lam
