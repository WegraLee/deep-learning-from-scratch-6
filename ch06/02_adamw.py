import torch
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                p.data = p.data - lr * p.grad.data


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['t'] += 1
                t = state['t']

                # 1차 모멘트와 2차 모멘트 갱신
                m, v = state['m'], state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                state['m'], state['v'] = m, v

                # 편향 보정
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
                # 파라미터 갱신과 가중치 감쇠 적용
                p.data = p.data - lr * m_hat / (v_hat.sqrt() + eps) - lr * wd * p.data


torch.manual_seed(0)

# 단순한 선형 모델
model = torch.nn.Linear(2, 1)
# optimizer = AdamW(model.parameters(), lr=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
# 학습용 더미 데이터
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[3.0]])

# 몇 스텝 학습하여 손실이 줄어드는지 확인
for step in range(5):
    output = model(x)
    loss = (output - y).pow(2).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Step {step}: loss = {loss.item():.4f}")
