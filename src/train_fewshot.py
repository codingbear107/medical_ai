"""
Phase 2: Sagittal few-shot fine-tuning.

전제:
  - merge_and_eval.py가 base_model.pth를 만들어둠 (병합된 view A+C 모델).
  - view A의 Fisher와 view C의 Fisher를 합쳐 EWC penalty의 기준으로 사용.

알고리즘:
  1) base_model.pth 로드
  2) Stage1 + Stem + StyleRandomization 동결
  3) Stage2/Stage3/FC head는 차등 학습률로 학습
  4) Loss = CE(label_smoothing) + EWC penalty
  5) Train data = Sagittal 50샷 × N 클래스 + Replay buffer (A+C 각 50샷)
  6) Early stopping은 대회식 combined metric 기준:
        combined = 0.7 * F1(S) + 0.3 * (F1(A) + F1(C)) / 2

저장:
  - fewshot_model.pth (best by combined metric)
  - model.pth         (제출용 alias, fewshot_model.pth와 동일)
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import config
from model import MiniResNet11, count_parameters
from ewc import EWCLoss
from utils import (set_seed, evaluate, save_checkpoint, load_checkpoint,
                   AverageMeter, EarlyStopping)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Few-shot fine-tuning")
    p.add_argument("--dataset", type=str, default="pathmnist",
                   choices=['pathmnist', 'real'])
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--base_ckpt", type=str, default=None,
                   help="merge된 base_model.pth (None이면 config.BASE_CKPT)")
    p.add_argument("--ewc_lambda", type=float, default=config.EWC_LAMBDA)
    p.add_argument("--epochs", type=int, default=config.FEWSHOT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.FEWSHOT_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--save_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    p.add_argument("--no_ewc", action="store_true",
                   help="EWC penalty 끄기 (디버깅용)")
    p.add_argument("--no_replay", action="store_true",
                   help="Replay buffer 끄기 (디버깅용)")
    return p.parse_args()


def get_datasets(args, num_classes):
    """fewshot train/val, replay(A), replay(C),
    그리고 base val (A, C 각각 — 대회식 combined metric용) 반환."""
    if args.dataset == 'pathmnist':
        from dataset_pathmnist import (create_fewshot_datasets,
                                       create_replay_dataset,
                                       PathMNISTViewDataset,
                                       sample_balanced_indices,
                                       _load_pathmnist)
        fs_train, fs_val = create_fewshot_datasets(view='S')
        replay_a = create_replay_dataset('A', samples_per_class=config.REPLAY_SAMPLES_PER_CLASS)
        replay_c = create_replay_dataset('C', samples_per_class=config.REPLAY_SAMPLES_PER_CLASS)

        cache = _load_pathmnist()
        _, val_labels = cache['val']
        idx = sample_balanced_indices(val_labels, 30, seed=args.seed + 200)
        base_val_a = PathMNISTViewDataset('val', 'A', augment_fn=None, indices=idx)
        base_val_c = PathMNISTViewDataset('val', 'C', augment_fn=None, indices=idx)
    else:
        from dataset import (create_fewshot_datasets, create_replay_dataset,
                             create_view_datasets)
        fs_train, fs_val = create_fewshot_datasets()
        replay_a = create_replay_dataset('A')
        replay_c = create_replay_dataset('C')
        _, base_val_a = create_view_datasets('A', augment_train=False)
        _, base_val_c = create_view_datasets('C', augment_train=False)

    return fs_train, fs_val, replay_a, replay_c, base_val_a, base_val_c


def freeze_stage1(model):
    """Stem + Stage1 + StyleRandomization 동결."""
    for name, param in model.named_parameters():
        if name.startswith(('stem.', 'stage1.')):
            param.requires_grad = False
    for param in model.style_rand.parameters():
        param.requires_grad = False


def get_param_groups(model):
    """Stage2 / Stage3 / FC head에 차등 학습률."""
    stage2 = [p for n, p in model.named_parameters()
              if n.startswith('stage2.') and p.requires_grad]
    stage3 = [p for n, p in model.named_parameters()
              if n.startswith('stage3.') and p.requires_grad]
    head = [p for n, p in model.named_parameters()
            if n.startswith(('fc.',)) and p.requires_grad]
    return [
        {'params': stage2, 'lr': config.FEWSHOT_LR_STAGE2},
        {'params': stage3, 'lr': config.FEWSHOT_LR_STAGE3},
        {'params': head,   'lr': config.FEWSHOT_LR_HEAD},
    ]


def load_combined_ewc(model, device, args):
    """View A와 view C의 Fisher를 합쳐 EWC penalty 기준으로 사용.

    수식:
        F_combined = F_A + F_C
        θ* = base_model.pth (= merged 가중치)

    "A에 중요한 것"과 "C에 중요한 것" 모두 보존.
    """
    a_fisher_path = config.view_ckpt('A', 'fisher')
    c_fisher_path = config.view_ckpt('C', 'fisher')

    if not (os.path.exists(a_fisher_path) and os.path.exists(c_fisher_path)):
        print("[fewshot] WARN: Fisher 파일 없음. EWC 비활성화.")
        return None

    fa = torch.load(a_fisher_path, map_location=device)
    fc = torch.load(c_fisher_path, map_location=device)

    fisher = {}
    for k in fa:
        if k in fc:
            fisher[k] = fa[k].to(device) + fc[k].to(device)
        else:
            fisher[k] = fa[k].to(device)

    # θ*: load 직후 모델의 파라미터 (= merged base_model.pth)
    old_params = {n: p.data.detach().clone()
                  for n, p in model.named_parameters()}

    ewc = EWCLoss.__new__(EWCLoss)
    ewc.fisher = fisher
    ewc.old_params = old_params
    ewc.ewc_lambda = args.ewc_lambda
    return ewc


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = args.num_classes or (9 if args.dataset == 'pathmnist' else 11)
    print(f"[fewshot] dataset={args.dataset}, num_classes={num_classes}, "
          f"device={device}")

    base_ckpt = args.base_ckpt or os.path.join(args.save_dir, "base_model.pth")
    if not os.path.exists(base_ckpt):
        print(f"ERROR: {base_ckpt} 없음. merge_and_eval.py를 먼저 실행하세요.")
        sys.exit(1)

    model = MiniResNet11(num_classes=num_classes,
                         dropout=config.BASE_DROPOUT).to(device)
    model = load_checkpoint(model, base_ckpt, device)
    print(f"[fewshot] Base 모델 로드: {base_ckpt}")

    freeze_stage1(model)
    total, trainable = count_parameters(model)
    print(f"[fewshot] params: total={total:,}, trainable={trainable:,}")

    ewc = None if args.no_ewc else load_combined_ewc(model, device, args)
    if ewc is not None:
        print(f"[fewshot] EWC λ={args.ewc_lambda} (F_A + F_C 사용)")

    fs_train, fs_val, replay_a, replay_c, base_val_a, base_val_c = \
        get_datasets(args, num_classes)
    print(f"[fewshot] fewshot_train (with repeat): {len(fs_train)}")
    print(f"[fewshot] fewshot_val: {len(fs_val)}")
    print(f"[fewshot] replay A/C: {len(replay_a)}/{len(replay_c)}")
    print(f"[fewshot] base val A/C: {len(base_val_a)}/{len(base_val_c)}")

    train_components = [fs_train]
    if not args.no_replay:
        train_components += [replay_a, replay_c]
    combined = ConcatDataset(train_components)

    train_loader = DataLoader(combined, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    fs_val_loader = DataLoader(fs_val, batch_size=args.batch_size * 2,
                               shuffle=False, num_workers=0)
    base_val_a_loader = DataLoader(base_val_a, batch_size=args.batch_size * 2,
                                   shuffle=False, num_workers=0)
    base_val_c_loader = DataLoader(base_val_c, batch_size=args.batch_size * 2,
                                   shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(get_param_groups(model),
                                  weight_decay=config.FEWSHOT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.FEWSHOT_LABEL_SMOOTHING)

    early_stop = EarlyStopping(patience=args.patience)
    best_combined = -1.0
    fewshot_path = os.path.join(args.save_dir, "fewshot_model.pth")
    final_path = os.path.join(args.save_dir, "model.pth")

    for epoch in range(args.epochs):
        model.train()
        ce_meter = AverageMeter()
        ewc_meter = AverageMeter()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            ce_loss = criterion(logits, y)

            if ewc is not None:
                ewc_loss = ewc.penalty(model)
                total_loss = ce_loss + ewc_loss
                ewc_meter.update(ewc_loss.item())
            else:
                total_loss = ce_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            ce_meter.update(ce_loss.item(), x.size(0))

        scheduler.step()

        sag_f1, _ = evaluate(model, fs_val_loader, device,
                             num_classes=num_classes)
        ax_f1, _ = evaluate(model, base_val_a_loader, device,
                            num_classes=num_classes)
        co_f1, _ = evaluate(model, base_val_c_loader, device,
                            num_classes=num_classes)

        # 대회식 combined metric (정확히)
        combined_metric = 0.7 * sag_f1 + 0.3 * (ax_f1 + co_f1) / 2

        print(f"[fewshot] Ep {epoch+1:3d}/{args.epochs} | "
              f"CE: {ce_meter.avg:.4f} | "
              f"EWC: {ewc_meter.avg:.4g} | "
              f"S F1: {sag_f1:.4f} | "
              f"A F1: {ax_f1:.4f} | C F1: {co_f1:.4f} | "
              f"Combined: {combined_metric:.4f}")

        improved = early_stop.step(combined_metric)
        if improved:
            best_combined = combined_metric
            extra = {"epoch": epoch, "sag_f1": sag_f1,
                     "ax_f1": ax_f1, "co_f1": co_f1,
                     "combined": combined_metric}
            save_checkpoint(model, fewshot_path, extra=extra)
            save_checkpoint(model, final_path, extra=extra)
            print(f"  -> Best 저장 (Combined={combined_metric:.4f})")

        if early_stop.should_stop:
            print(f"[fewshot] Early stop at epoch {epoch+1}")
            break

    print(f"\n[fewshot] 학습 완료. Best Combined={best_combined:.4f}")
    print(f"[fewshot] 제출용 model.pth: {final_path}")


if __name__ == "__main__":
    main()
