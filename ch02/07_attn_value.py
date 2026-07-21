import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, key_dim, bias=False)  # 밸류 변환 행렬
        self.W_o = nn.Linear(key_dim, embed_dim, bias=False)  # 출력 변환 행렬
        self.key_dim = key_dim

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)  # 저차원 밸류 생성

        # 어텐션 계산
        K_t = K.transpose(-2, -1)
        scores = torch.matmul(Q, K_t)
        scores = scores / (self.key_dim ** 0.5)

        B, C, E = x.shape
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        hidden = torch.matmul(weights, V)

        # 출력 변환
        output = self.W_o(hidden)

        return output

attention = Attention(embed_dim=256, key_dim=64)
x = torch.randn(2, 5, 256)
y = attention(x)

print("입력 형상:", x.shape)
print("출력 형상:", y.shape)