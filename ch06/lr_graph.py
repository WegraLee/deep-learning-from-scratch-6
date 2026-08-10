import matplotlib.pyplot as plt
import numpy as np

# 한국어 폰트 설정
plt.rcParams['font.family'] = 'Apple SD Gothic Neo'  # macOS
# plt.rcParams['font.family'] = 'Malgun Gothic  # Windows

# 파라미터
warmup_ratio = 0.05  # 워밍업 기간(전체의 5%)
eta_min_ratio = 0.1  # 코사인 어닐링의 최소 학습률(최대의 10%)

# 데이터 생성
t = np.linspace(0, 1, 1000)

# 코사인 어닐링(워밍업 포함)
def cosine_annealing(t, warmup_ratio, eta_min_ratio):
    lr = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < warmup_ratio:
            # 워밍업: 0 -> 1
            lr[i] = ti / warmup_ratio
        else:
            # 코사인 어닐링: 1 -> eta_min
            progress = (ti - warmup_ratio) / (1 - warmup_ratio)
            lr[i] = eta_min_ratio + 0.5 * (1 - eta_min_ratio) * (1 + np.cos(np.pi * progress))
    return lr

# D2Z(워밍업 포함)
def d2z(t, warmup_ratio):
    lr = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < warmup_ratio:
            # 워밍업: 0 -> 1
            lr[i] = ti / warmup_ratio
        else:
            # 선형 감쇠: 1 -> 0
            progress = (ti - warmup_ratio) / (1 - warmup_ratio)
            lr[i] = 1 - progress
    return lr

cosine_lr = cosine_annealing(t, warmup_ratio, eta_min_ratio)
d2z_lr = d2z(t, warmup_ratio)

# 플롯 생성
fig, ax = plt.subplots(figsize=(10, 6))

# 워밍업 구간을 회색으로 채움
ax.axvspan(0, warmup_ratio, color='lightgray', alpha=0.5)
ax.text(0.01, 1.02, '워밍업', fontsize=10, va='bottom')

# 학습률 곡선
ax.plot(t, cosine_lr, 'b-', linewidth=2, label='코사인 어닐링')
ax.plot(t, d2z_lr, color='orange', linestyle='--', linewidth=2, label='D2Z')

# 축 설정
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_xlabel('학습 진행 상황', fontsize=12)
ax.set_ylabel('학습률', fontsize=12)

# 그리드
ax.grid(True, linestyle='--', alpha=0.7)

# 범례
ax.legend(loc='upper right', fontsize=12)

# 여백 조정
plt.tight_layout()

# PNG로 저장
plt.savefig('lr_schedule.png', format='png', bbox_inches='tight')
plt.close()