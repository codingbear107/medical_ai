"""
MiniResNet-11: 28×28 grayscale CT 의료영상 분류용 컴팩트 ResNet.

설계 원칙
─────────
1. 28×28에 맞춘 stem (해상도 보존, stride=1).
2. Stage1을 28×28에서 유지 → 얕은 층은 view-invariant 일반 feature 학습.
3. Stage1 뒤에 StyleRandomization 모듈 → channel statistic 교란으로
   texture bias 제거, 모델이 shape에 의존하도록 유도.
4. 약 300K 파라미터 (28×28 작은 입력에 적합한 크기).
5. Conv/BN: Kaiming init (He init, ReLU와 호환).
   Linear classifier head: Xavier uniform init (softmax와 호환).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleRandomization(nn.Module):
    """채널별 평균/표준편차를 학습 중에 랜덤 교란.
    Stage1과 Stage2 사이에 삽입.

    수식:
       mu, sigma = mean/std per channel of feature map
       mu'    = mu    + N(0, mu_std)
       sigma' = sigma * (1 + N(0, sigma_std)).clamp(0.5, 2.0)
       F'     = (F - mu)/sigma * sigma' + mu'

    효과: feature의 "스타일" 통계를 교란해 모델이 통계가 아닌 구조적 특징
    (= shape, topology)에 의존하도록 강제한다.
    추론 시에는 identity로 동작 (model.eval() 호출 시).
    """

    def __init__(self, mu_std=0.5, sigma_std=0.3):
        super().__init__()
        self.mu_std = mu_std
        self.sigma_std = sigma_std

    def forward(self, x):
        if not self.training:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        sigma = x.std(dim=[2, 3], keepdim=True) + 1e-6
        x_norm = (x - mu) / sigma

        new_mu = mu + torch.randn_like(mu) * self.mu_std
        new_sigma = sigma * (1 + torch.randn_like(sigma) * self.sigma_std).clamp(0.5, 2.0)
        return x_norm * new_sigma + new_mu


class BasicBlock(nn.Module):
    """ResNet BasicBlock: 두 개의 3×3 conv + 잔차 연결."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut: stride 또는 채널 수 변화 시 1×1 conv로 차원 맞춤
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class MiniResNet11(nn.Module):
    """
    구조:
        입력 (1, 28, 28)
          ↓ Stem: Conv3x3(1→64) + BN + ReLU       [28×28 보존]
          ↓ Stage1: BasicBlock×2 (64→64)          [28×28]
          ↓ StyleRandomization                    [training only]
          ↓ Stage2: BasicBlock×2 (64→128, s=2)    [14×14]
          ↓ Stage3: BasicBlock×2 (128→256, s=2)   [7×7]
          ↓ Global Average Pool → Dropout → FC
        출력 (num_classes,)

    파라미터: 약 300K.
    """

    def __init__(self, num_classes=11, dropout=0.3, use_style_rand=True,
                 in_channels=1):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 256

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )

        self.style_rand = StyleRandomization() if use_style_rand else nn.Identity()

        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
        )

        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Kaiming for Conv (ReLU 호환), Xavier for FC head (softmax 호환)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.style_rand(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_features(self, x):
        """분류기 직전의 feature 벡터 (256-d) 추출."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.style_rand(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return x


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for nc in [9, 11]:
        m = MiniResNet11(num_classes=nc)
        x = torch.randn(2, 1, 28, 28)
        y = m(x)
        total, trainable = count_parameters(m)
        print(f"num_classes={nc}: out={y.shape}, params={total:,}")
