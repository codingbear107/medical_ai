"""
Phase 1: View별 base 학습 (Axial 또는 Coronal 단독).

핵심:
  - 두 view를 별도 모델로 학습 (joint training이 아닌 view-separated).
  - 같은 init seed에서 출발해야 task vector(τ = θ_view - θ_init)가 의미를 가짐.
  - 학습 종료 후 Fisher Information Matrix를 계산해 EWC와 Fisher-Weighted
    Merging에 활용.

저장물:
  - {view}_model.pth        : 학습 완료 가중치
  - {view}_fisher.pth       : Fisher Information (대각선 근사)
  - {view}_params.pth       : 학습 완료 시점의 파라미터 (EWC 기준점)
  - init.pth (옵션)          : 첫 view 학습 시 init state 저장
                                (Task Arithmetic / TIES merge에 필요)

사용:
    python train_view.py --view A --seed 42 --dataset pathmnist
    python train_view.py --view C --seed 42 --dataset pathmnist

    Stage 2 (본 게임):
    python train_view.py --view A --seed 42 --dataset real
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from model import MiniResNet11, count_parameters
from augmentations import mixup, cutmix
from ewc import compute_fisher, save_fisher
from utils import set_seed, evaluate, save_checkpoint, AverageMeter


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: View-별 base 학습")
    p.add_argument("--view", type=str, required=True, choices=['A', 'C'],
                   help="학습할 view (Axial 또는 Coronal)")
    p.add_argument("--dataset", type=str, default="pathmnist",
                   choices=['pathmnist', 'real'],
                   help="pathmnist=워밍업, real=본 게임")
    p.add_argument("--num_classes", type=int, default=None,
                   help="None이면 dataset에 따라 자동 (pathmnist=9, real=11)")
    p.add_argument("--epochs", type=int, default=config.BASE_EPOCHS)
    p.add_argument("--lr", type=float, default=config.BASE_LR)
    p.add_argument("--batch_size", type=int, default=config.BASE_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--save_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--save_init", action="store_true",
                   help="θ_init를 init.pth로 저장 (Task Arithmetic용)")
    p.add_argument("--no_style_rand", action="store_true")
    p.add_argument("--no_fisher", action="store_true",
                   help="Fisher 계산 건너뛰기 (디버깅용)")
    return p.parse_args()


def get_datasets(args):
    """dataset 인자에 따라 train/val 데이터셋 + num_classes 반환."""
    if args.dataset == 'pathmnist':
        from dataset_pathmnist import create_view_datasets
        num_classes = args.num_classes or 9
        train_ds, val_ds = create_view_datasets(args.view, augment_train=True)
    elif args.dataset == 'real':
        from dataset import create_view_datasets  # 본 게임용 (데이터 도착 후 작성)
        num_classes = args.num_classes or 11
        train_ds, val_ds = create_view_datasets(args.view, augment_train=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return train_ds, val_ds, num_classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    """1 epoch 학습 (mixup/cutmix 확률적 적용, gradient clipping)."""
    model.train()
    loss_meter = AverageMeter()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Mixup / CutMix / Plain
        r = np.random.random()
        if r < config.MIXUP_PROB:
            x, y_a, y_b, lam = mixup(x, y, config.MIXUP_ALPHA)
        elif r < config.MIXUP_PROB + config.CUTMIX_PROB:
            x, y_a, y_b, lam = cutmix(x, y, config.CUTMIX_ALPHA)
        else:
            y_a, y_b, lam = y, y, 1.0

        logits = model(x)
        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))

    return loss_meter.avg


def main():
    args = parse_args()
    # seed 고정 — view A와 C가 같은 seed로 학습되어야 init이 일치
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train_view] view={args.view}, dataset={args.dataset}, "
          f"seed={args.seed}, device={device}")

    # 데이터
    train_ds, val_ds, num_classes = get_datasets(args)
    print(f"[train_view] num_classes={num_classes}")
    print(f"[train_view] train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=2, pin_memory=True)

    # 모델 (seed 고정 후 생성 → init이 결정적)
    model = MiniResNet11(
        num_classes=num_classes,
        dropout=config.BASE_DROPOUT,
        use_style_rand=not args.no_style_rand,
    ).to(device)
    total, trainable = count_parameters(model)
    print(f"[train_view] params: total={total:,}, trainable={trainable:,}")

    os.makedirs(args.save_dir, exist_ok=True)

    # init state 저장 (Task Arithmetic / TIES merge에 필요)
    if args.save_init:
        init_path = os.path.join(args.save_dir, "init.pth")
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   init_path)
        print(f"[train_view] init state 저장: {init_path}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.BASE_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.COSINE_T0, T_mult=config.COSINE_TMULT,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.BASE_LABEL_SMOOTHING)

    # 학습 루프
    best_f1 = -1.0
    best_path = config.view_ckpt(args.view, kind="model")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        scheduler.step()

        val_f1, val_acc = evaluate(model, val_loader, device,
                                   num_classes=num_classes)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"[view={args.view}] Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | Acc: {val_acc:.4f} | "
              f"LR: {lr_now:.2e}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, best_path,
                            extra={"epoch": epoch, "val_f1": val_f1,
                                   "view": args.view, "seed": args.seed})
            print(f"  -> Best 저장: {best_path} (F1={val_f1:.4f})")

    print(f"\n[view={args.view}] 학습 완료. Best Val F1: {best_f1:.4f}")

    # Fisher Information Matrix 계산
    if args.no_fisher:
        return
    print(f"[view={args.view}] Fisher Information 계산 중...")
    # best 모델 다시 로드
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    fisher = compute_fisher(model, val_loader, device,
                            num_samples=config.FISHER_NUM_SAMPLES)

    # Fisher와 best params 저장
    fisher_path = config.view_ckpt(args.view, kind="fisher")
    params_path = config.view_ckpt(args.view, kind="params")
    old_params = {n: p.data.detach().cpu().clone()
                  for n, p in model.named_parameters()}
    save_fisher({k: v.detach().cpu() for k, v in fisher.items()},
                old_params, fisher_path, params_path)
    print(f"[view={args.view}] Fisher 저장: {fisher_path}")
    print(f"[view={args.view}] Params 저장: {params_path}")


if __name__ == "__main__":
    main()
