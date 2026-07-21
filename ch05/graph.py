import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(-3, 3, 500)

# ReLU 함수
relu = np.maximum(0, x)

# GELU 함수(근사식)
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Swish 함수(SiLU)
swish = x * (1 / (1 + np.exp(-x)))  # x * sigmoid(x)

# 플롯 생성
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, relu, 'b-', linewidth=2, label='ReLU')
ax.plot(x, gelu, 'g-', linewidth=2, label='GELU')
ax.plot(x, swish, 'r-', linewidth=2, label='Swish')

# 축 설정
ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)

# 그리드
ax.grid(True, linestyle='--', alpha=0.7)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

# 범례
ax.legend(loc='upper left', fontsize=12)

# 여백 조정
plt.tight_layout()

# PNG로 저장
plt.savefig('activation_comparison.png', format='png', bbox_inches='tight')
plt.close()