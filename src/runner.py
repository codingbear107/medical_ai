"""
파이프라인 자동 실행 도우미.

Colab 노트북에서 한 번 호출하면 다음을 자동 진행:
  1. View A 학습 (이미 결과 있으면 skip)
  2. View C 학습 (이미 결과 있으면 skip)
  3. Merge grid search (이미 결과 있으면 skip)
  4. Few-shot fine-tuning (이미 결과 있으면 skip)
  5. 최종 결과 요약 출력

각 단계는 독립적으로 실행 가능하며, 실패 시 명확한 에러 메시지 출력.
체크포인트가 있으면 skip하므로 세션 끊겨도 다음 세션에서 이어 실행 가능.

사용:
    from runner import run_full_pipeline
    run_full_pipeline(dataset='pathmnist', seed=42,
                      view_epochs=20, fewshot_epochs=30)
"""
import os
import sys
import subprocess
import time

import config


# ─── 단계별 결과물 경로 ─────────────────────────────────

CKPT_DIR = config.CHECKPOINT_DIR

def _expected_files():
    """단계별로 생성되어야 할 체크포인트 파일들."""
    return {
        'view_A': [
            os.path.join(CKPT_DIR, 'init.pth'),
            os.path.join(CKPT_DIR, 'axial_model.pth'),
            os.path.join(CKPT_DIR, 'axial_fisher.pth'),
            os.path.join(CKPT_DIR, 'axial_params.pth'),
        ],
        'view_C': [
            os.path.join(CKPT_DIR, 'coronal_model.pth'),
            os.path.join(CKPT_DIR, 'coronal_fisher.pth'),
            os.path.join(CKPT_DIR, 'coronal_params.pth'),
        ],
        'merge': [
            os.path.join(CKPT_DIR, 'base_model.pth'),
            os.path.join(CKPT_DIR, 'merge_grid_results.csv'),
        ],
        'fewshot': [
            os.path.join(CKPT_DIR, 'fewshot_model.pth'),
            os.path.join(CKPT_DIR, 'model.pth'),
        ],
    }


def _all_exist(files):
    return all(os.path.exists(f) for f in files)


# ─── 단계 실행 함수 ─────────────────────────────────────

def _run_step(name, cmd, expected_files, force=False):
    """한 단계 실행. 이미 결과가 있고 force=False면 skip.

    Args:
        name: 표시용 이름 (예: 'View A 학습')
        cmd: subprocess로 실행할 명령 리스트
        expected_files: 이 단계가 만들어야 할 파일 경로들
        force: True면 이미 있어도 재실행

    Returns:
        True if step ran (or already done), False if failed
    """
    print(f"\n{'='*70}")
    print(f"  [STEP] {name}")
    print(f"{'='*70}")

    if not force and _all_exist(expected_files):
        print(f"  ✓ 이미 완료된 단계 (출력 파일 모두 존재)")
        for f in expected_files:
            size = os.path.getsize(f) // 1024
            print(f"     {os.path.basename(f)}: {size} KB")
        print(f"  → SKIP")
        return True

    print(f"  실행: {' '.join(cmd)}")
    print()
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ✗ 실패 (exit code {result.returncode}, {elapsed:.0f}초)")
        return False

    # 결과 파일 검증
    missing = [f for f in expected_files if not os.path.exists(f)]
    if missing:
        print(f"\n  ✗ 명령은 성공했으나 출력 파일 누락:")
        for f in missing:
            print(f"     missing: {f}")
        return False

    print(f"\n  ✓ 완료 ({elapsed:.0f}초)")
    for f in expected_files:
        size = os.path.getsize(f) // 1024
        print(f"     {os.path.basename(f)}: {size} KB")
    return True


# ─── 메인 ────────────────────────────────────────────────

def run_full_pipeline(dataset='pathmnist', seed=42,
                       view_epochs=20, fewshot_epochs=30,
                       force=False):
    """전체 파이프라인 자동 실행.

    Args:
        dataset: 'pathmnist' (워밍업) | 'real' (본 게임)
        seed: 재현용 시드
        view_epochs: View A/C 각각의 epoch 수
        fewshot_epochs: Few-shot fine-tuning epoch 수
        force: 모든 단계 강제 재실행 (이미 결과 있어도)

    Returns:
        True if pipeline 완료, False if 중간 실패
    """
    print(f"\n{'#'*70}")
    print(f"#  파이프라인 자동 실행")
    print(f"#  dataset={dataset}, seed={seed}")
    print(f"#  view_epochs={view_epochs}, fewshot_epochs={fewshot_epochs}")
    print(f"#  force={force}")
    print(f"{'#'*70}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    expected = _expected_files()

    # Step 1: View A 학습 (--save_init 포함)
    ok = _run_step(
        'View A 학습',
        [sys.executable, 'train_view.py',
         '--view', 'A', '--dataset', dataset,
         '--epochs', str(view_epochs), '--seed', str(seed),
         '--save_init'],
        expected['view_A'], force=force
    )
    if not ok:
        print("\n✗ View A 학습 실패. 중단.")
        return False

    # Step 2: View C 학습
    ok = _run_step(
        'View C 학습',
        [sys.executable, 'train_view.py',
         '--view', 'C', '--dataset', dataset,
         '--epochs', str(view_epochs), '--seed', str(seed)],
        expected['view_C'], force=force
    )
    if not ok:
        print("\n✗ View C 학습 실패. 중단.")
        return False

    # Step 3: Merge grid search
    ok = _run_step(
        'Model Merge grid search',
        [sys.executable, 'merge_and_eval.py',
         '--dataset', dataset, '--seed', str(seed)],
        expected['merge'], force=force
    )
    if not ok:
        print("\n✗ Merge 실패. 중단.")
        return False

    # Step 4: Few-shot fine-tuning
    ok = _run_step(
        'Few-shot fine-tuning',
        [sys.executable, 'train_fewshot.py',
         '--dataset', dataset, '--seed', str(seed),
         '--epochs', str(fewshot_epochs)],
        expected['fewshot'], force=force
    )
    if not ok:
        print("\n✗ Few-shot 실패. 중단.")
        return False

    # 최종 요약
    _print_summary()
    return True


def _print_summary():
    """파이프라인 결과 요약 출력."""
    print(f"\n{'#'*70}")
    print(f"#  파이프라인 완료 — 결과 요약")
    print(f"{'#'*70}\n")

    # 1) View A/C Best F1
    import torch
    for view, name in [('A', 'Axial'), ('C', 'Coronal')]:
        path = config.view_ckpt(view, 'model')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            f1 = ckpt.get('val_f1', 'N/A')
            print(f"  {name} (View {view}) Val F1: {f1}")

    # 2) Merge 결과
    csv_path = os.path.join(CKPT_DIR, 'merge_grid_results.csv')
    if os.path.exists(csv_path):
        print(f"\n  Merge Grid 결과 (Sagittal F1 내림차순):")
        with open(csv_path) as f:
            lines = f.readlines()
        for line in lines[:6]:  # 헤더 + top 5
            print(f"     {line.rstrip()}")

    # 3) Few-shot 최종
    fs_path = os.path.join(CKPT_DIR, 'fewshot_model.pth')
    if os.path.exists(fs_path):
        ckpt = torch.load(fs_path, map_location='cpu')
        sag = ckpt.get('sag_f1', 'N/A')
        ax = ckpt.get('ax_f1', 'N/A')
        co = ckpt.get('co_f1', 'N/A')
        comb = ckpt.get('combined', 'N/A')
        print(f"\n  Few-shot 최종:")
        print(f"     Sagittal F1: {sag}")
        print(f"     Axial F1:    {ax}")
        print(f"     Coronal F1:  {co}")
        print(f"     Combined (대회식):  {comb}")

    final = config.FINAL_MODEL
    if os.path.exists(final):
        print(f"\n  제출용 model.pth: {final}")
        print(f"  (크기: {os.path.getsize(final)//1024} KB)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default='pathmnist',
                   choices=['pathmnist', 'real'])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--view_epochs", type=int, default=20)
    p.add_argument("--fewshot_epochs", type=int, default=30)
    p.add_argument("--force", action='store_true',
                   help="이미 완료된 단계도 재실행")
    args = p.parse_args()
    run_full_pipeline(args.dataset, args.seed,
                      args.view_epochs, args.fewshot_epochs,
                      args.force)
