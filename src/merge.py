"""
Model Merging — 이 대회의 핵심 알고리즘.

배경:
  본 대회의 평가식은 Final = 0.7·F1(S) + 0.3·(F1(A)+F1(C))/2.
  주최측은 "파라미터 공간(Weight Space)의 직교성"을 강조한다. 즉,
  Axial 전문가 θ_A와 Coronal 전문가 θ_C를 학습한 뒤 두 모델의 가중치를
  "수학적으로" 합성하여 view-invariant 3D-aware feature를 얻는 것이 핵심.

  Task vector:
      τ_A = θ_A - θ_init      (Axial 학습이 초기값에서 이동시킨 방향)
      τ_C = θ_C - θ_init

  여기서 4가지 merge 기법을 구현:
    1) simple_average        : θ = (θ_A + θ_C) / 2
    2) task_arithmetic       : θ = θ_init + λ·(τ_A + τ_C)
    3) fisher_weighted       : θ_i = (F_A,i·θ_A,i + F_C,i·θ_C,i)/(F_A,i+F_C,i)
    4) ties_merging          : Trim(top-k%) → Sign Election → Disjoint Mean
                              θ = θ_init + λ·τ_merged

  모두 state_dict 기반 (parameter dict). BatchNorm running stats나
  num_batches_tracked는 단순 평균 처리 (Tier 1: BN을 별도로 다루지 않음).

References:
  - Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023
  - Matena & Raffel, "Merging Models with Fisher-Weighted Averaging", NeurIPS 2022
  - Yadav et al., "TIES-Merging: Resolving Interference When Merging Models", NeurIPS 2023
"""
import torch
from collections import OrderedDict


# ─── 공통 유틸 ──────────────────────────────────────────

def _is_mergeable(name, tensor):
    """Merge 대상 파라미터인지 판정.
    BatchNorm running stats는 평균이 의미가 있지만, num_batches_tracked는 정수라
    별도 처리 (마지막 모델 값 사용).
    """
    if 'num_batches_tracked' in name:
        return False
    return tensor.dtype in (torch.float32, torch.float16, torch.float64)


def _state_dict_clone(sd):
    return OrderedDict((k, v.clone()) for k, v in sd.items())


def _check_shapes(state_dicts):
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if set(sd.keys()) != keys:
            raise ValueError("State dicts have different keys")
        for k in keys:
            if sd[k].shape != state_dicts[0][k].shape:
                raise ValueError(f"Shape mismatch at {k}: "
                                 f"{sd[k].shape} vs {state_dicts[0][k].shape}")


# ─── 기법 1: Simple Averaging ────────────────────────────

def simple_average(state_dicts):
    """θ_merged = mean(θ_i for i in views)

    가장 단순한 베이스라인. λ-free.
    두 모델의 weight space가 같은 loss basin에 있다고 가정한다.
    """
    _check_shapes(state_dicts)
    merged = _state_dict_clone(state_dicts[0])
    for k, v in merged.items():
        if not _is_mergeable(k, v):
            continue
        stack = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        merged[k] = stack.mean(dim=0).to(v.dtype)
    return merged


# ─── 기법 2: Task Arithmetic ─────────────────────────────

def task_arithmetic(init_state, view_states, lam=1.0):
    """θ_merged = θ_init + λ·Σ(θ_view - θ_init)

    Task vector의 합을 λ로 스케일링.
    λ=1.0이면 task vector를 그대로 더함, λ<1이면 보수적, λ>1이면 증폭.

    Args:
        init_state: θ_init (학습 시작 가중치)
        view_states: [θ_A, θ_C, ...] (학습 완료 가중치들)
        lam: 스케일 계수
    """
    _check_shapes([init_state] + list(view_states))
    merged = _state_dict_clone(init_state)
    for k, v in merged.items():
        if not _is_mergeable(k, v):
            # num_batches_tracked 등은 마지막 view의 값으로 (또는 init 유지)
            merged[k] = view_states[-1][k]
            continue
        # τ_sum = Σ (θ_view - θ_init)
        tau_sum = torch.zeros_like(v, dtype=torch.float32)
        for sd in view_states:
            tau_sum = tau_sum + (sd[k].float() - v.float())
        merged[k] = (v.float() + lam * tau_sum).to(v.dtype)
    return merged


# ─── 기법 3: Fisher-Weighted Merging ─────────────────────

def fisher_weighted(view_states, fishers, eps=1e-8):
    """파라미터별 Fisher Information으로 가중 평균.

        θ_merged,i = Σ(F_view,i · θ_view,i) / Σ(F_view,i + ε)

    Fisher가 큰 파라미터일수록 그 view의 값을 더 신뢰한다.
    "각 view에서 중요한 파라미터는 그 view 값으로 보존"의 효과.

    Args:
        view_states: [θ_A, θ_C, ...]
        fishers: [F_A, F_C, ...]  (각각 dict of {name: tensor})
    """
    if len(view_states) != len(fishers):
        raise ValueError("view_states and fishers must have same length")
    _check_shapes(view_states)

    merged = _state_dict_clone(view_states[0])
    for k, v in merged.items():
        if not _is_mergeable(k, v):
            continue
        # Fisher가 없는 파라미터(BN 등)는 단순 평균
        if not all(k in f for f in fishers):
            stack = torch.stack([sd[k].float() for sd in view_states], dim=0)
            merged[k] = stack.mean(dim=0).to(v.dtype)
            continue
        num = torch.zeros_like(v, dtype=torch.float32)
        denom = torch.zeros_like(v, dtype=torch.float32)
        for sd, f in zip(view_states, fishers):
            fk = f[k].float()
            num = num + fk * sd[k].float()
            denom = denom + fk
        merged[k] = (num / (denom + eps)).to(v.dtype)
    return merged


# ─── 기법 4: TIES-Merging ────────────────────────────────

def _trim(tensor, top_k_ratio=0.2):
    """Magnitude 상위 top_k% 만 남기고 나머지는 0.
    Task vector에서 "잡음" 파라미터를 제거.
    """
    flat = tensor.flatten()
    n_keep = max(1, int(len(flat) * top_k_ratio))
    if n_keep >= len(flat):
        return tensor
    threshold = torch.topk(flat.abs(), n_keep, largest=True).values.min()
    mask = tensor.abs() >= threshold
    return tensor * mask


def ties_merging(init_state, view_states, top_k=0.2, lam=1.0):
    """TIES-Merging (Trim · Elect Sign · Disjoint Merge).

    Step 1 (Trim): 각 task vector에서 magnitude 상위 top_k% 만 남김.
    Step 2 (Elect Sign): 파라미터별로 trim 후 부호 합의:
                         sign_final = sign(Σ τ_view,i)
    Step 3 (Disjoint Merge): final 부호와 일치하는 vector만 평균.

    최종: θ_merged = θ_init + λ · τ_merged

    Args:
        init_state: θ_init
        view_states: [θ_A, θ_C, ...]
        top_k: 0~1, magnitude 기준 유지 비율
        lam: scale
    """
    _check_shapes([init_state] + list(view_states))
    merged = _state_dict_clone(init_state)

    for k, v in merged.items():
        if not _is_mergeable(k, v):
            merged[k] = view_states[-1][k]
            continue

        v_init = v.float()
        # Step 1: 각 view의 task vector를 trim
        taus = []
        for sd in view_states:
            tau = sd[k].float() - v_init
            tau_trimmed = _trim(tau, top_k_ratio=top_k)
            taus.append(tau_trimmed)

        # Step 2: 부호 합의 — magnitude로 가중하여 sign(Σ τ) 계산
        tau_stack = torch.stack(taus, dim=0)
        sign_sum = tau_stack.sum(dim=0)
        sign_final = torch.sign(sign_sum)
        # 0인 곳은 그대로 0 (변경 없음)

        # Step 3: Disjoint merge — sign_final과 같은 부호의 vector만 평균
        same_sign_mask = (torch.sign(tau_stack) == sign_final.unsqueeze(0)) & (tau_stack != 0)
        # 분자: sign 일치하는 값들의 합
        numerator = (tau_stack * same_sign_mask.float()).sum(dim=0)
        # 분모: sign 일치하는 vector 개수
        denominator = same_sign_mask.float().sum(dim=0).clamp(min=1.0)
        tau_merged = numerator / denominator

        merged[k] = (v_init + lam * tau_merged).to(v.dtype)

    return merged


# ─── 통합 인터페이스 ─────────────────────────────────────

def merge(method, view_states, init_state=None, fishers=None, **kwargs):
    """모든 merge 기법의 통합 진입점.

    Args:
        method: 'simple' | 'task_arithmetic' | 'fisher' | 'ties'
        view_states: [θ_A, θ_C, ...]
        init_state: task arithmetic / ties에 필요
        fishers: fisher_weighted에 필요
        **kwargs: 기법별 추가 파라미터 (lam, top_k 등)

    Returns:
        merged state_dict
    """
    if method == 'simple':
        return simple_average(view_states)
    elif method == 'task_arithmetic':
        if init_state is None:
            raise ValueError("init_state required for task_arithmetic")
        return task_arithmetic(init_state, view_states,
                               lam=kwargs.get('lam', 1.0))
    elif method == 'fisher':
        if fishers is None:
            raise ValueError("fishers required for fisher_weighted")
        return fisher_weighted(view_states, fishers)
    elif method == 'ties':
        if init_state is None:
            raise ValueError("init_state required for ties_merging")
        return ties_merging(init_state, view_states,
                            top_k=kwargs.get('top_k', 0.2),
                            lam=kwargs.get('lam', 1.0))
    else:
        raise ValueError(f"Unknown merge method: {method}")


# ─── 부수 유틸 ──────────────────────────────────────────

def task_vector_orthogonality(state_a, state_c, init_state):
    """두 task vector의 cosine similarity 측정.
    값이 0에 가까울수록 직교 (대회가 강조하는 성질).

    Returns:
        cosine similarity in [-1, 1]
    """
    dot = 0.0
    norm_a_sq = 0.0
    norm_c_sq = 0.0
    for k, v in init_state.items():
        if not _is_mergeable(k, v):
            continue
        tau_a = (state_a[k].float() - v.float()).flatten()
        tau_c = (state_c[k].float() - v.float()).flatten()
        dot += (tau_a * tau_c).sum().item()
        norm_a_sq += (tau_a * tau_a).sum().item()
        norm_c_sq += (tau_c * tau_c).sum().item()
    if norm_a_sq == 0 or norm_c_sq == 0:
        return 0.0
    return dot / (norm_a_sq ** 0.5 * norm_c_sq ** 0.5)


if __name__ == "__main__":
    # Sanity check: 두 임의 state_dict로 merge 테스트
    from model import MiniResNet11
    torch.manual_seed(0)
    init = MiniResNet11(num_classes=9).state_dict()
    torch.manual_seed(1)
    a = MiniResNet11(num_classes=9).state_dict()
    torch.manual_seed(2)
    c = MiniResNet11(num_classes=9).state_dict()

    print("Simple average:", "OK" if simple_average([a, c]) else "FAIL")
    print("Task arithmetic:", "OK" if task_arithmetic(init, [a, c], lam=1.0) else "FAIL")
    print("TIES:", "OK" if ties_merging(init, [a, c], top_k=0.2, lam=1.0) else "FAIL")

    # Fisher: dummy fisher (모두 1)
    fa = {k: torch.ones_like(v) for k, v in a.items() if v.dtype.is_floating_point}
    fc = {k: torch.ones_like(v) for k, v in c.items() if v.dtype.is_floating_point}
    print("Fisher-weighted:", "OK" if fisher_weighted([a, c], [fa, fc]) else "FAIL")

    cs = task_vector_orthogonality(a, c, init)
    print(f"Task vector cosine similarity (random init→random A,C): {cs:.4f}")
