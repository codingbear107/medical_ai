# 중앙대 의료AI 해커톤 2026 — Few-Shot Domain Generalization

CT 영상의 다각도 인지 챌린지: Axial + Coronal로 학습한 모델을 Sagittal에 일반화한다.

## 폴더 구조

```
medical_ai_hackathon/
├── README.md
├── .gitignore
├── requirements_warmup.txt   # Colab 워밍업용
├── requirements_submit.txt   # 대회 제출용
├── colab_main.ipynb          # Colab 마스터 노트북
└── src/
    ├── config.py
    ├── model.py              # MiniResNet-11
    ├── augmentations.py      # numpy/torch only
    ├── ewc.py                # Elastic Weight Consolidation
    ├── utils.py
    ├── merge.py              # Model Merging 4종
    ├── dataset_pathmnist.py  # 워밍업 (PathMNIST + view 시뮬레이션)
    ├── dataset.py            # 본 게임 (실데이터 도착 후 작성)
    ├── train_view.py         # view별 base 학습
    ├── merge_and_eval.py     # merge grid search
    ├── train_fewshot.py      # Sagittal few-shot fine-tuning
    └── inference.py          # 제출용 추론 스크립트
```

## 실행 흐름

```
[Stage 1: 워밍업 (PathMNIST)]
1. dataset_pathmnist.py 로 View A/C/S 시뮬레이션 데이터 생성
2. train_view.py --view A   → axial_model.pth + fisher_axial.pth
3. train_view.py --view C   → coronal_model.pth + fisher_coronal.pth
4. merge_and_eval.py        → 4종 merge 비교 → base_model.pth
5. train_fewshot.py         → fewshot_model.pth → model.pth
6. inference.py             → submission.csv

[Stage 2: 본 게임 (대회 데이터)]
- dataset.py 신설 (실데이터 폴더 구조에 맞게)
- 위 흐름 그대로 재실행
```

## 알고리즘 스택

- **Tier 1**: View-separated training, Model Merging (Simple/Task Arithmetic/Fisher-weighted/TIES), Style Randomization, Shape-centric augmentation
- **Tier 2**: View-equivariant aux loss, Prototype head, EWC + Replay buffer, TTA + Ensemble
- **Tier 3**: Gram-Schmidt 직교화, Self-distillation

자세한 알고리즘 설명: 별도 가이드 문서 참조.

## 평가식

```
Final = 0.7 * F1_macro(Sagittal) + 0.3 * (F1_macro(Axial) + F1_macro(Coronal)) / 2
```

## 윤리 가드레일

- ImageNet 등 pretrained weights 금지 (Random Init만)
- 외부 데이터 금지 (PathMNIST는 워밍업 전용, 본 게임에 가중치/하이퍼파라미터 직접 복사 금지)
- 제출 inference.py 는 medmnist 등 미허용 패키지 import 금지
