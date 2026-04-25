# PathMNIST 워밍업 결과 (2026-04-25)

본 게임 데이터 도착 전 파이프라인 검증을 위한 PathMNIST 워밍업 사이클 결과.

## 환경

- Colab T4 GPU (15.6 GB VRAM)
- PyTorch 2.2.0+cu121 (또는 2.10.0+cu128, 둘 다 동작)
- Python 3.12 (대회는 3.10 — 호환 코드만 사용)
- Drive symlink로 체크포인트 영구 저장

## 데이터 구성 (워밍업)

PathMNIST (28×28, 9 클래스 의료 영상). View 시뮬레이션:

```
View A (Axial 시뮬):    원본
View C (Coronal 시뮬):  90° 회전
View S (Sagittal 시뮬): 270° 회전 + 50% 가벼운 shear
```

Train: 89,996장 / Val: 10,004장 / Test: 7,180장 (모두 9 클래스)
Sagittal few-shot: 50샷 × 9 클래스 = 450장 (반복 20배 = epoch당 9000장)
Sagittal val: 30샷 × 9 = 270장
Replay: View A/C에서 각 50샷씩 = 합 900장

## 학습 설정

- 모델: MiniResNet-11 (2,776,265 params)
- View A/C: 10 epoch each, AdamW lr=1e-3, mixup+cutmix, CosineAnnealingWarmRestarts
- Few-shot: 20 epoch, AdamW (stage2 1e-5, stage3 5e-5, head 2e-4), Stage1 freeze
- EWC λ=5000, Fisher num_samples=2000, Replay 50/class
- num_workers=2, persistent_workers=True
- 총 소요 시간: 40분

## 결과

### View A 학습 (10 epoch)

| Epoch | Loss | Val F1 | Acc |
|------:|-----:|-------:|----:|
| 1 | 1.7236 | 0.5441 | 0.5402 |
| 3 | 1.3773 | 0.7867 | 0.7855 |
| 5 | 1.3155 | 0.7894 | 0.7915 |
| 7 | 1.2636 | 0.7756 | 0.7732 |
| 9 | 1.2481 | **0.8379** | 0.8353 |
| **10** | 1.2236 | **0.8383** | 0.8375 |

→ Best Val F1 **0.8383**

### View C 학습 (10 epoch)

| Epoch | Loss | Val F1 | Acc |
|------:|-----:|-------:|----:|
| 1 | 1.7255 | 0.6543 | 0.6550 |
| 3 | 1.3851 | 0.7921 | 0.7922 |
| 5 | 1.3175 | 0.7693 | 0.7693 |
| 7 | 1.2643 | 0.7643 | 0.7680 |
| 9 | 1.2493 | 0.7934 | 0.7915 |
| **10** | 1.2245 | **0.8052** | 0.8039 |

→ Best Val F1 **0.8052** (View A보다 약간 낮음, 90° 회전 변형 영향)

### Task Vector 직교성

```
τ_A · τ_C cosine similarity = 0.8819
```

⚠ **거의 같은 방향** — 두 모델이 weight space에서 유사한 학습 궤적.
대회가 강조하는 "직교성"이 PathMNIST 시뮬레이션에선 성립 안 함.
원인: 같은 init + 같은 데이터 + view만 회전 → 비슷한 학습 방향.

### Merge Grid Search (Sagittal val 900샷 평가)

| Method | Hyperparam | Sag F1 | Sag Acc |
|--------|-----------|-------:|--------:|
| **task_arithmetic** | **λ=0.5** | **0.2271** | **0.3200** |
| simple average | — | 0.2258 | 0.3189 |
| fisher-weighted | — | 0.1621 | 0.2433 |
| ties | top_k=0.4, λ=1.0 | 0.0476 | 0.1400 |
| task_arithmetic | λ=0.7 | 0.0222 | 0.1111 (랜덤) |
| task_arithmetic | λ=1.0 | 0.0222 | 0.1111 |
| task_arithmetic | λ=1.3 | 0.0222 | 0.1111 |
| ties (모든 top_k=0.2) | λ=0.7~1.3 | 0.0222 | 0.1111 |
| ties top_k=0.4 | λ=0.7, 1.3 | 0.0222 | 0.1111 |

**핵심**:
- task_arith λ=0.5 ≈ simple average (수학적 등가)
- λ ≥ 0.7 모두 발산 — τ 같은 방향 + 큰 λ → weights 폭발
- TIES는 top_k 0.2~0.4가 너무 sparse → 정보 손실

→ 본 게임용 grid 조정 (commit `0121dc3`):
- task_arithmetic λ ∈ {0.1, 0.3, 0.5, 0.7}
- TIES top_k ∈ {0.3, 0.5, 0.7}, λ ∈ {0.3, 0.5, 0.7}

### Few-shot Fine-tuning (20 epoch)

best_merge (`task_arith λ=0.5`)을 base로, Sagittal 50샷 + EWC + Replay.

| Epoch | CE | EWC | S F1 | A F1 | C F1 | Combined |
|------:|----:|-----:|-----:|-----:|-----:|---------:|
| 1 | 1.4594 | 0.0053 | 0.6989 | 0.6685 | 0.6582 | 0.6882 |
| 3 | 1.3333 | 0.0077 | 0.7789 | 0.7499 | 0.7361 | 0.7681 |
| 9 | 1.2514 | 0.0118 | 0.7812 | 0.7565 | 0.7780 | 0.7770 |
| 10 | 1.2360 | 0.0124 | 0.7838 | 0.7656 | 0.7800 | 0.7805 |
| 12 | 1.2255 | 0.0129 | 0.7885 | 0.7761 | 0.7762 | 0.7848 |
| 15 | 1.2117 | 0.0135 | 0.7813 | 0.7779 | 0.7825 | 0.7809 |
| **18** | 1.2119 | 0.0136 | **0.7958** | **0.7791** | **0.7825** | **0.7913** |
| 20 | 1.2064 | 0.0136 | 0.7936 | 0.7499 | 0.7528 | 0.7809 |

→ Best Combined **0.7913** (epoch 18)

검증식: `0.7 × 0.7958 + 0.3 × (0.7791 + 0.7825) / 2 = 0.5571 + 0.2342 = 0.7913` ✓

### EWC + Replay 효과 (catastrophic forgetting 방어)

| | Base 단독 학습 | Few-shot 후 | 손실 |
|---|--------------:|------------:|-----:|
| View A F1 | 0.8383 | 0.7791 | -7.1% |
| View C F1 | 0.8052 | 0.7783 | -3.3% |
| Sagittal F1 (base merge) | 0.2271 | 0.7958 | **+250%** |

→ Sagittal 적응하면서도 base view 잊지 않음. EWC λ=5000 적정.

## 핵심 시사점 (본 게임 적용)

### 1. Merge보다 Few-shot이 진짜 승부수

```
Base merge Sagittal F1:  0.227
Few-shot Sagittal F1:    0.796   (3.5배 향상)
```

→ 본 게임에서도 Few-shot이 점수 핵심. EWC λ, replay 비중 grid가 우선순위.

### 2. τ 직교성을 본 게임에서 측정해야 함

워밍업 0.88은 PathMNIST view 시뮬레이션의 한계 (단순 회전).
실제 CT의 Axial/Coronal은 진짜 다른 정보 보유 → 더 직교 예상 (추측).
직교성에 따라 merge 효과가 갈림.

### 3. λ는 보수적으로

λ ≥ 0.7면 발산. 본 게임에서도 처음엔 λ=0.3~0.5에서 시작.
직교성 측정 후 λ 늘리는 게 안전.

### 4. EWC는 잘 동작 — 그대로 사용

λ=5000, Fisher samples=2000 → forgetting 거의 없음.
본 게임은 Fisher samples=5000으로 확장 (commit `0121dc3`).

### 5. 파이프라인은 검증 완료

데이터 도착 시 dataset.py만 폴더 구조에 맞춰 미세 조정 → 같은 코드로 끝까지 동작.

## 본 게임 예상 점수 (참고)

| 시나리오 | 가정 | 예상 Combined |
|---------|------|-------------:|
| 비관 | τ 0.85+, 11 클래스 어려움 | 0.65 |
| 현실 | 워밍업 패턴 그대로 | 0.75 |
| 낙관 | τ 0.3~0.5 (진짜 직교적), Few-shot 시너지 | 0.85 |

## 체크포인트

Drive 보관 (https://drive.google.com/drive/MyDrive/medical_ai_ckpts):
- init.pth, axial_model.pth, axial_fisher.pth, axial_params.pth
- coronal_model.pth, coronal_fisher.pth, coronal_params.pth
- base_model.pth (best merge: task_arith λ=0.5)
- fewshot_model.pth, **model.pth** (제출 가능 형태)
- merge_grid_results.csv

⚠ **윤리 가드레일**: 위 가중치는 본 게임에서 절대 재사용 금지. PathMNIST 학습이라 외부 데이터 사용 위반.

## 다음 단계

1. 본 게임 데이터 수령
2. Drive에 `data/axial/`, `data/coronal/`, `data/sagittal/` 업로드
3. PathMNIST 체크포인트 모두 삭제
4. `colab_auto.ipynb` 셀 5의 `--dataset pathmnist` → `--dataset real`, epoch 늘리기
5. 모두 실행 → 약 4시간
6. 결과 보고 grid search 좁히기
