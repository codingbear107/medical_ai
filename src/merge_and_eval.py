"""
Merge grid search.

train_view.py로 학습한 view A / view C 모델 (그리고 init.pth)을 불러와
4가지 merge 기법 × 여러 하이퍼파라미터 조합을 평가.
평가는 view S validation 데이터에 대해 수행 (Sagittal 일반화 능력 측정).

선택된 best merge state_dict는 base_model.pth로 저장되어
train_fewshot.py의 출발점이 된다.

사용:
    python merge_and_eval.py --dataset pathmnist
    python merge_and_eval.py --dataset real
"""
import argparse
import os
import sys
import csv
import torch
from torch.utils.data import DataLoader

import config
from model import MiniResNet11
from merge import (simple_average, task_arithmetic,
                   fisher_weighted, ties_merging,
                   task_vector_orthogonality)
from utils import set_seed, evaluate, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Merge grid search")
    p.add_argument("--dataset", type=str, default="pathmnist",
                   choices=['pathmnist', 'real'])
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--save_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--batch_size", type=int, default=256)
    # 평가용 sagittal validation set (워밍업 시 PathMNIST test에서 샘플)
    p.add_argument("--sag_val_per_class", type=int, default=100,
                   help="Sagittal validation 클래스당 샘플 수 (워밍업)")
    return p.parse_args()


def get_sagittal_val_loader(args, num_classes):
    """Sagittal view validation loader."""
    if args.dataset == 'pathmnist':
        from dataset_pathmnist import PathMNISTViewDataset, sample_balanced_indices, _load_pathmnist
        cache = _load_pathmnist()
        _, test_labels = cache['test']
        idx = sample_balanced_indices(test_labels, args.sag_val_per_class,
                                       seed=args.seed + 100)
        ds = PathMNISTViewDataset('test', 'S', augment_fn=None, indices=idx)
    else:
        from dataset import create_view_datasets
        _, ds = create_view_datasets('S', augment_train=False)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


def load_view_state(path, device='cpu'):
    """체크포인트 로드 (model_state_dict 형태 또는 raw state_dict)."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def eval_merged(state, num_classes, sag_loader, device):
    """주어진 state_dict로 모델 만들고 sagittal val에서 F1 측정."""
    model = MiniResNet11(num_classes=num_classes,
                        dropout=0.0, use_style_rand=False).to(device)
    model.load_state_dict(state)
    f1, acc = evaluate(model, sag_loader, device, num_classes=num_classes)
    return f1, acc


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = args.num_classes or (9 if args.dataset == 'pathmnist' else 11)
    print(f"[merge_and_eval] dataset={args.dataset}, num_classes={num_classes}")

    # 체크포인트 경로
    a_model_path = config.view_ckpt('A', 'model')
    c_model_path = config.view_ckpt('C', 'model')
    a_fisher_path = config.view_ckpt('A', 'fisher')
    c_fisher_path = config.view_ckpt('C', 'fisher')
    init_path = os.path.join(args.save_dir, "init.pth")

    for p in [a_model_path, c_model_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} 없음. train_view.py를 먼저 실행하세요.")
            sys.exit(1)

    # State dict 로드 (CPU에서 merge → 평가 시 GPU로 이동)
    state_a = load_view_state(a_model_path, device='cpu')
    state_c = load_view_state(c_model_path, device='cpu')
    print(f"[merge_and_eval] view A, C 로드 완료")

    # init state (옵션)
    init_state = None
    if os.path.exists(init_path):
        init_state = torch.load(init_path, map_location='cpu')
        # init이 raw state_dict 형식
        if isinstance(init_state, dict) and 'model_state_dict' in init_state:
            init_state = init_state['model_state_dict']
        print(f"[merge_and_eval] init.pth 로드 → Task Arithmetic / TIES 사용 가능")
    else:
        print(f"[merge_and_eval] init.pth 없음 → Task Arithmetic / TIES 건너뜀")

    # Fisher 로드 (옵션)
    fishers = None
    if os.path.exists(a_fisher_path) and os.path.exists(c_fisher_path):
        fa = torch.load(a_fisher_path, map_location='cpu')
        fc = torch.load(c_fisher_path, map_location='cpu')
        fishers = [fa, fc]
        print(f"[merge_and_eval] Fisher 로드 → Fisher-weighted merge 사용 가능")
    else:
        print(f"[merge_and_eval] Fisher 없음 → Fisher-weighted 건너뜀")

    # Task vector 직교성 측정 (참고용)
    if init_state is not None:
        cs = task_vector_orthogonality(state_a, state_c, init_state)
        print(f"[merge_and_eval] τ_A · τ_C cosine similarity = {cs:.4f}  "
              f"(0에 가까울수록 직교 — 대회가 강조하는 성질)")

    # Sagittal val loader
    sag_loader = get_sagittal_val_loader(args, num_classes)
    print(f"[merge_and_eval] Sagittal val: {len(sag_loader.dataset)} samples")

    # ─── 평가할 조합들 ──────────────────────────────
    candidates = []  # (method_name, state_dict)

    # 1) Simple Average
    print("\n[1] Simple Average ...")
    s = simple_average([state_a, state_c])
    f1, acc = eval_merged(s, num_classes, sag_loader, device)
    print(f"   Sag F1={f1:.4f}, Acc={acc:.4f}")
    candidates.append(("simple", {}, s, f1, acc))

    # 2) Task Arithmetic (init 필요)
    if init_state is not None:
        for lam in [0.5, 0.7, 1.0, 1.3]:
            print(f"\n[2] Task Arithmetic (λ={lam}) ...")
            s = task_arithmetic(init_state, [state_a, state_c], lam=lam)
            f1, acc = eval_merged(s, num_classes, sag_loader, device)
            print(f"   Sag F1={f1:.4f}, Acc={acc:.4f}")
            candidates.append(("task_arithmetic", {"lam": lam}, s, f1, acc))

    # 3) Fisher-Weighted (fisher 필요)
    if fishers is not None:
        print("\n[3] Fisher-Weighted ...")
        s = fisher_weighted([state_a, state_c], fishers)
        f1, acc = eval_merged(s, num_classes, sag_loader, device)
        print(f"   Sag F1={f1:.4f}, Acc={acc:.4f}")
        candidates.append(("fisher", {}, s, f1, acc))

    # 4) TIES-Merging (init 필요)
    if init_state is not None:
        for top_k in [0.2, 0.4]:
            for lam in [0.7, 1.0, 1.3]:
                print(f"\n[4] TIES (top_k={top_k}, λ={lam}) ...")
                s = ties_merging(init_state, [state_a, state_c],
                                 top_k=top_k, lam=lam)
                f1, acc = eval_merged(s, num_classes, sag_loader, device)
                print(f"   Sag F1={f1:.4f}, Acc={acc:.4f}")
                candidates.append(("ties", {"top_k": top_k, "lam": lam}, s, f1, acc))

    # 결과 정리
    candidates.sort(key=lambda x: -x[3])  # F1 내림차순
    print("\n" + "=" * 70)
    print("Merge Grid Search 결과 (Sagittal F1 내림차순)")
    print("=" * 70)
    print(f"{'Method':<20} {'Hyperparam':<20} {'Sag F1':<10} {'Acc':<10}")
    print("-" * 70)
    for name, hp, _, f1, acc in candidates:
        print(f"{name:<20} {str(hp):<20} {f1:<10.4f} {acc:<10.4f}")

    # CSV 저장
    csv_path = os.path.join(args.save_dir, "merge_grid_results.csv")
    with open(csv_path, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["method", "hyperparam", "sag_f1", "sag_acc"])
        for name, hp, _, f1_, acc in candidates:
            w.writerow([name, str(hp), f1_, acc])
    print(f"\n[merge_and_eval] CSV: {csv_path}")

    # Best 선정 → base_model.pth 저장
    best_name, best_hp, best_state, best_f1, best_acc = candidates[0]
    print(f"\n[merge_and_eval] BEST: {best_name} {best_hp} (F1={best_f1:.4f})")
    base_path = os.path.join(args.save_dir, "base_model.pth")
    torch.save({
        'model_state_dict': best_state,
        'merge_method': best_name,
        'merge_hyperparam': best_hp,
        'sag_f1': best_f1,
    }, base_path)
    print(f"[merge_and_eval] Best merged model 저장: {base_path}")


if __name__ == "__main__":
    main()
